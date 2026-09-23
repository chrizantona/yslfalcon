import test, { beforeEach } from 'node:test';
import assert from 'node:assert/strict';
import { api, ApiError, explanationFromJob } from '../src/api.ts';
import type { Job } from '../src/types.ts';

let calls: { url: string; options: RequestInit }[];
let responses: { status: number; body?: unknown }[];
let events: string[];
beforeEach(() => {
  calls = [];
  responses = [];
  events = [];
  Object.defineProperty(globalThis, 'document', { value: undefined, configurable: true });
  const browserEvents = new EventTarget();
  browserEvents.addEventListener('falcon:unauthorized', (event) => events.push(event.type));
  browserEvents.addEventListener('falcon:connected', (event) => events.push(event.type));
  Object.defineProperty(globalThis, 'window', { value: browserEvents, configurable: true });
  globalThis.fetch = async (input, options = {}) => {
    calls.push({ url: String(input), options });
    const next = responses.shift() || { status: 200, body: {} };
    return new Response(next.status === 204 ? null : JSON.stringify(next.body), {
      status: next.status,
      headers: { 'Content-Type': 'application/json' },
    });
  };
});
test('bootstrap reuses an existing session and keeps its role and CSRF for mutations', async () => {
  responses.push({
    status: 200,
    body: { role: 'viewer', csrf_token: 'existing-csrf', workspace: 'local' },
  });
  const session = await api.bootstrapSession();
  assert.equal(session.role, 'viewer');
  assert.equal(calls.length, 1);
  assert.equal(calls[0].url, '/api/v1/auth/me');
  assert.equal(calls[0].options.credentials, 'same-origin');
  await api.createGallery('Test gallery', 'Description');
  assert.equal(new Headers(calls[1].options.headers).get('X-CSRF-Token'), 'existing-csrf');
});
test('missing session creates one local session without a secret or request body', async () => {
  responses.push(
    { status: 401, body: { error: { code: 'unauthorized', message: 'Expired' } } },
    { status: 200, body: { role: 'admin', csrf_token: 'local-csrf', workspace: 'local' } },
  );
  await api.bootstrapSession();
  assert.deepEqual(
    calls.map((call) => call.url),
    ['/api/v1/auth/me', '/api/v1/auth/local-session'],
  );
  assert.equal(calls[1].options.method, 'POST');
  assert.equal(calls[1].options.body, undefined);
  assert.equal(calls[1].options.credentials, 'same-origin');
  assert.equal(new Headers(calls[1].options.headers).get('Authorization'), null);
  assert.equal(new Headers(calls[1].options.headers).get('X-CSRF-Token'), null);
  assert.deepEqual(events, []);
  await api.createGallery('Test gallery', 'Description');
  assert.equal(new Headers(calls[2].options.headers).get('X-CSRF-Token'), 'local-csrf');
});
test('parallel bootstrap attempts share the entire GET and local-session flow', async () => {
  responses.push(
    { status: 401 },
    { status: 200, body: { role: 'admin', csrf_token: 'shared-csrf' } },
  );
  const first = api.bootstrapSession();
  const second = api.bootstrapSession();
  assert.equal(first, second);
  const [a, b] = await Promise.all([first, second]);
  assert.equal(a, b);
  assert.equal(calls.length, 2);
});
test('failed service and forbidden session responses never trigger local-login fallback', async () => {
  for (const status of [503, 403]) {
    responses.push({ status, body: { error: { code: 'unavailable', message: 'Unavailable' } } });
    const before = calls.length;
    await assert.rejects(
      () => api.bootstrapSession(),
      (cause: unknown) => cause instanceof ApiError && cause.status === status,
    );
    assert.equal(calls.length, before + 1);
    assert.equal(calls.at(-1)?.url, '/api/v1/auth/me');
  }
});
test('local-session rejection emits no expired event and stops after one attempt', async () => {
  responses.push({ status: 401 }, { status: 401 });
  await assert.rejects(
    () => api.bootstrapSession(),
    (cause: unknown) => cause instanceof ApiError && cause.status === 401,
  );
  assert.equal(calls.length, 2);
  assert.deepEqual(events, []);
});
test('network failure surfaces without fallback and a manual retry can connect', async () => {
  const normalFetch = globalThis.fetch;
  globalThis.fetch = async (input, options = {}) => {
    calls.push({ url: String(input), options });
    throw new TypeError('Network unavailable');
  };
  await assert.rejects(() => api.bootstrapSession(), /Network unavailable/);
  assert.equal(calls.length, 1);
  globalThis.fetch = normalFetch;
  responses.push(
    { status: 401 },
    { status: 200, body: { role: 'admin', csrf_token: 'recovered-csrf' } },
  );
  assert.equal((await api.bootstrapSession()).csrf_token, 'recovered-csrf');
  assert.equal(calls.length, 3);
});
test('expired protected request signals recovery and the next session supplies fresh CSRF', async () => {
  responses.push({ status: 401 });
  await assert.rejects(() => api.capabilities());
  assert.deepEqual(events, ['falcon:unauthorized']);
  responses.push(
    { status: 401 },
    { status: 200, body: { role: 'admin', csrf_token: 'renewed-csrf' } },
  );
  await api.bootstrapSession();
  await api.createGallery('Recovered gallery', 'Description');
  assert.equal(new Headers(calls.at(-1)?.options.headers).get('X-CSRF-Token'), 'renewed-csrf');
  assert.deepEqual(events, ['falcon:unauthorized', 'falcon:connected']);
});
test('a delayed rejection from an older session cannot expire the renewed session', async () => {
  const normalFetch = globalThis.fetch;
  let finishOld: ((response: Response) => void) | undefined;
  globalThis.fetch = (input, options = {}) => {
    if (String(input).endsWith('/capabilities')) {
      calls.push({ url: String(input), options });
      return new Promise<Response>((resolve) => {
        finishOld = resolve;
      });
    }
    return normalFetch(input, options);
  };
  const oldRequest = api.capabilities();
  responses.push({ status: 200, body: { role: 'admin', csrf_token: 'new-generation' } });
  await api.bootstrapSession();
  assert.ok(finishOld);
  finishOld(
    new Response(JSON.stringify({ error: { message: 'Old session expired' } }), { status: 401 }),
  );
  await assert.rejects(() => oldRequest);
  assert.deepEqual(events, []);
});
test('mutations use the current browser cookie after another tab renews the shared session', async () => {
  const browser = { cookie: 'falcon_csrf=first-tab-csrf' };
  Object.defineProperty(globalThis, 'document', { value: browser, configurable: true });
  responses.push({ status: 200, body: { role: 'admin', csrf_token: 'first-tab-csrf' } });
  await api.bootstrapSession();
  await api.createGallery('First gallery', 'Before renewal');
  assert.equal(new Headers(calls[1].options.headers).get('X-CSRF-Token'), 'first-tab-csrf');
  // Renewal happened in a second tab; no bootstrap or memory update occurs in this one.
  browser.cookie = 'falcon_csrf_extra=wrong; theme=light; falcon_csrf=second%2Dtab%2Dcsrf';
  await api.galleries();
  await api.createGallery('Second gallery', 'After renewal');
  assert.equal(new Headers(calls[3].options.headers).get('X-CSRF-Token'), 'second-tab-csrf');
  assert.equal(calls.length, 4);
  assert.equal(calls.filter((call) => call.options.method === 'POST').length, 2);
});
test('missing browser CSRF cookie never falls back to stale memory or retries the mutation', async () => {
  const browser = { cookie: 'falcon_csrf=previous-csrf' };
  Object.defineProperty(globalThis, 'document', { value: browser, configurable: true });
  responses.push({ status: 200, body: { role: 'admin', csrf_token: 'previous-csrf' } });
  await api.bootstrapSession();
  browser.cookie = 'theme=light';
  responses.push({
    status: 403,
    body: { error: { code: 'csrf_failed', message: 'CSRF missing' } },
  });
  await assert.rejects(
    () => api.createGallery('Test', ''),
    (cause: unknown) => cause instanceof ApiError && cause.status === 403,
  );
  assert.equal(new Headers(calls[1].options.headers).get('X-CSRF-Token'), null);
  assert.equal(calls.length, 2);
  assert.deepEqual(events, []);
});
test('malformed and unsafe CSRF cookies do not throw or produce an invalid header', async () => {
  for (const value of ['%not-encoded', 'bad%0Avalue', 'bad%0Dvalue', '%E2%98%83']) {
    Object.defineProperty(globalThis, 'document', {
      value: { cookie: `falcon_csrf=${value}` },
      configurable: true,
    });
    await api.createGallery('Test', '');
    assert.equal(new Headers(calls.at(-1)?.options.headers).get('X-CSRF-Token'), null);
  }
  assert.equal(calls.length, 4);
});
test('manifest is sent as multipart JSON text, not an uploaded file', async () => {
  const manifest = new File(['{"vehicle.jpg":{"bbox":[10,20,300,180]}}'], 'manifest.json', {
    type: 'application/json',
  });
  await api.importGallery(
    'gallery-1',
    [new File(['image'], 'vehicle.jpg', { type: 'image/jpeg' })],
    manifest,
  );
  const body = calls[0].options.body as FormData;
  assert.equal(body.getAll('files').length, 1);
  assert.equal(typeof body.get('manifest'), 'string');
  assert.deepEqual(JSON.parse(body.get('manifest') as string), {
    'vehicle.jpg': { bbox: [10, 20, 300, 180] },
  });
  assert.equal(new Headers(calls[0].options.headers).has('Content-Type'), false);
});
test('activation includes the observed version, including explicit null', async () => {
  await api.activate('gallery-1', 'version-2', 'version-1');
  assert.deepEqual(JSON.parse(calls[0].options.body as string), {
    expected_active_version_id: 'version-1',
  });
  await api.activate('gallery-1', 'version-1', null);
  assert.deepEqual(JSON.parse(calls[1].options.body as string), {
    expected_active_version_id: null,
  });
});
test('version diff keeps both immutable snapshot ids and pagination explicit', async () => {
  await api.versionDiff('gallery 1', 'version/old', 'version/new', 100);
  assert.equal(
    calls[0].url,
    '/api/v1/galleries/gallery%201/versions/version%2Fnew/diff?from_version_id=version%2Fold&limit=100&offset=100',
  );
});
test('expert review transports normalized evidence and case export stays explicit', async () => {
  const annotations = [
    {
      image_ref: 'query',
      surface: 'crop' as const,
      kind: 'match' as const,
      x: 0.25,
      y: 0.75,
    },
  ];
  await api.review('search-1', 'candidate-1', 'uncertain', 'Проверить деталь', annotations);
  assert.deepEqual(JSON.parse(calls[0].options.body as string), {
    observation_id: 'candidate-1',
    verdict: 'uncertain',
    note: 'Проверить деталь',
    annotations,
  });
  await api.export('search-1', 'case');
  assert.deepEqual(JSON.parse(calls[1].options.body as string), { format: 'case' });
});
test('search transports the caller-owned idempotency key without generating a replacement', async () => {
  const image = new File(['image'], 'vehicle.png', { type: 'image/png' });
  await api.search(image, [10, 20, 300, 180], 'gallery-1', 10, 'stable-operation-key');
  await api.search(image, [10, 20, 300, 180], 'gallery-1', 10, 'stable-operation-key');
  for (const call of calls)
    assert.equal(new Headers(call.options.headers).get('Idempotency-Key'), 'stable-operation-key');
  assert.equal((calls[0].options.body as FormData).get('bbox'), '[10,20,300,180]');
});
test('server errors preserve request id and stable error code', async () => {
  responses.push({
    status: 409,
    body: {
      error: { code: 'activation_conflict', message: 'Version changed' },
      request_id: 'request-1',
    },
  });
  await assert.rejects(
    () => api.activate('gallery-1', 'version-2', 'version-1'),
    (error: unknown) => {
      assert.ok(error instanceof ApiError);
      assert.equal(error.status, 409);
      assert.equal(error.code, 'activation_conflict');
      assert.equal(error.requestId, 'request-1');
      return true;
    },
  );
});
test('pagination offset reaches the backend', async () => {
  await api.searches(200);
  await api.galleries(100);
  await api.jobs(300);
  assert.deepEqual(
    calls.map((call) => call.url),
    [
      '/api/v1/searches?limit=100&offset=200',
      '/api/v1/galleries?limit=100&offset=100',
      '/api/v1/jobs?limit=100&offset=300',
    ],
  );
});
test('missing explanation maps stay unavailable', () => {
  assert.equal(
    explanationFromJob({ result: { method: 'occlusion_sensitivity' } } as unknown as Job),
    null,
  );
  assert.ok(
    explanationFromJob({ result: { query_map: [[0]], candidate_map: [[0]] } } as unknown as Job),
  );
});
test('staged import preserves the manifest and create key, then uploads files and finalizes explicitly', async () => {
  const csv = 'image_id,x,y,w,h\nvehicle,10,20,300,180';
  await api.createImportSession('gallery-1', csv, 'stable-import-key');
  await api.createImportSession('gallery-1', csv, 'stable-import-key');
  assert.deepEqual(JSON.parse(calls[0].options.body as string), { manifest_csv: csv });
  for (const call of calls)
    assert.equal(new Headers(call.options.headers).get('Idempotency-Key'), 'stable-import-key');
  await api.importSessions('gallery-1');
  await api.importSession('gallery-1', 'import-1');
  await api.uploadImportFiles('gallery-1', 'import-1', [new File(['image'], 'vehicle.jpg')]);
  const body = calls[4].options.body as FormData;
  assert.equal(body.getAll('files').length, 1);
  assert.equal(new Headers(calls[4].options.headers).has('Content-Type'), false);
  assert.equal(
    calls.some((call) => call.url.endsWith('/finalize')),
    false,
  );
  await api.finalizeImport('gallery-1', 'import-1');
  assert.equal(calls[5].url, '/api/v1/galleries/gallery-1/import-sessions/import-1/finalize');
  assert.equal(calls[5].options.method, 'POST');
});
