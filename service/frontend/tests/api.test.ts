import test, { beforeEach } from 'node:test';
import assert from 'node:assert/strict';
import { api, ApiError, explanationFromJob } from '../src/api.ts';
import type { Job } from '../src/types.ts';

let calls: { url: string; options: RequestInit }[];
let responses: { status: number; body?: unknown }[];
beforeEach(() => {
  calls = [];
  responses = [];
  globalThis.fetch = async (input, options = {}) => {
    calls.push({ url: String(input), options });
    const next = responses.shift() || { status: 200, body: {} };
    return new Response(next.status === 204 ? null : JSON.stringify(next.body), {
      status: next.status,
      headers: { 'Content-Type': 'application/json' },
    });
  };
});
test('session uses same-origin cookies and CSRF on subsequent mutations', async () => {
  responses.push({
    status: 200,
    body: { role: 'admin', csrf_token: 'session-csrf', workspace: 'test' },
  });
  await api.login('user-token');
  assert.equal(calls[0].options.credentials, 'same-origin');
  assert.deepEqual(JSON.parse(calls[0].options.body as string), { token: 'user-token' });
  await api.createGallery('Test gallery', 'Description');
  assert.equal(new Headers(calls[1].options.headers).get('X-CSRF-Token'), 'session-csrf');
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
