import type {
  BBox,
  Capabilities,
  Gallery,
  GalleryVersion,
  GalleryVersionDiff,
  ImportSession,
  Job,
  ModelInfo,
  PageResult,
  Review,
  ReviewAnnotationDraft,
  SearchResult,
  Session,
  SystemInfo,
  Verdict,
} from './types';

const base = '/api/v1';
let csrfToken = '';
let sessionGeneration = 0;
let sessionRequest: Promise<Session> | null = null;
const isSessionRoute = (path: string) => path.startsWith('/auth/');
function currentCsrfToken(): string {
  if (typeof document === 'undefined') return csrfToken;
  const cookie = document.cookie
    .split(';')
    .map((part) => part.trim())
    .find((part) => part.startsWith('falcon_csrf='));
  if (!cookie) return '';
  try {
    const value = decodeURIComponent(cookie.slice('falcon_csrf='.length));
    // Browser cookies are shared by tabs; an absent or malformed cookie must not revive cached credentials.
    return /[\u0000-\u0020\u007f-\uffff]/.test(value) ? '' : value;
  } catch {
    return '';
  }
}
export class ApiError extends Error {
  readonly status: number;
  readonly code: string;
  readonly requestId?: string;
  constructor(message: string, status: number, code: string, requestId?: string) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.code = code;
    this.requestId = requestId;
  }
}
async function request<T>(path: string, options: RequestInit = {}): Promise<T> {
  const generation = sessionGeneration;
  const headers = new Headers(options.headers);
  if (options.body && !(options.body instanceof FormData))
    headers.set('Content-Type', 'application/json');
  const method = options.method || 'GET';
  if (!['GET', 'HEAD', 'OPTIONS'].includes(method)) {
    const currentToken = currentCsrfToken();
    if (currentToken) headers.set('X-CSRF-Token', currentToken);
    else headers.delete('X-CSRF-Token');
  }
  const response = await fetch(`${base}${path}`, {
    ...options,
    headers,
    credentials: 'same-origin',
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    if (
      response.status === 401 &&
      !isSessionRoute(path) &&
      generation === sessionGeneration &&
      typeof window !== 'undefined'
    )
      window.dispatchEvent(new Event('falcon:unauthorized'));
    throw new ApiError(
      body.error?.message || body.detail || `Ошибка запроса (${response.status})`,
      response.status,
      body.error?.code || 'request_failed',
      body.request_id,
    );
  }
  if (!isSessionRoute(path) && generation === sessionGeneration && typeof window !== 'undefined')
    window.dispatchEvent(new Event('falcon:connected'));
  return response.status === 204 ? (undefined as T) : response.json();
}
const json = (value: unknown) => JSON.stringify(value);
const id = encodeURIComponent;
export const api = {
  async session() {
    const session = await request<Session>('/auth/me');
    csrfToken = session.csrf_token;
    sessionGeneration += 1;
    return session;
  },
  async localSession() {
    const session = await request<Session>('/auth/local-session', { method: 'POST' });
    csrfToken = session.csrf_token;
    sessionGeneration += 1;
    return session;
  },
  bootstrapSession(): Promise<Session> {
    // StrictMode and concurrent expired requests share one bounded recovery attempt.
    if (!sessionRequest) {
      sessionRequest = api
        .session()
        .catch((cause: unknown) => {
          if (!(cause instanceof ApiError) || cause.status !== 401) throw cause;
          csrfToken = '';
          sessionGeneration += 1;
          return api.localSession();
        })
        .finally(() => {
          sessionRequest = null;
        });
    }
    return sessionRequest;
  },
  capabilities: () => request<Capabilities>('/capabilities'),
  galleries: (offset = 0) => request<PageResult<Gallery>>(`/galleries?limit=100&offset=${offset}`),
  createGallery: (name: string, description: string) =>
    request<Gallery>('/galleries', { method: 'POST', body: json({ name, description }) }),
  versions: (galleryId: string) =>
    request<{ items: GalleryVersion[] }>(`/galleries/${id(galleryId)}/versions`),
  versionDiff: (galleryId: string, fromVersion: string, toVersion: string, offset = 0) =>
    request<GalleryVersionDiff>(
      `/galleries/${id(galleryId)}/versions/${id(toVersion)}/diff?from_version_id=${id(fromVersion)}&limit=100&offset=${offset}`,
    ),
  async importGallery(galleryId: string, files: File[], manifest?: File) {
    const body = new FormData();
    files.forEach((file) => body.append('files', file));
    if (manifest) body.append('manifest', await manifest.text());
    return request<Job>(`/galleries/${id(galleryId)}/imports`, { method: 'POST', body });
  },
  importSessions: (galleryId: string) =>
    request<{ items: ImportSession[] }>(`/galleries/${id(galleryId)}/import-sessions`),
  importSession: (galleryId: string, sessionId: string) =>
    request<ImportSession>(`/galleries/${id(galleryId)}/import-sessions/${id(sessionId)}`),
  createImportSession: (galleryId: string, manifestCsv: string, operationKey: string) =>
    request<ImportSession>(`/galleries/${id(galleryId)}/import-sessions`, {
      method: 'POST',
      headers: { 'Idempotency-Key': operationKey },
      body: json({ manifest_csv: manifestCsv }),
    }),
  uploadImportFiles(galleryId: string, sessionId: string, files: File[]) {
    const body = new FormData();
    files.forEach((file) => body.append('files', file));
    return request<ImportSession>(
      `/galleries/${id(galleryId)}/import-sessions/${id(sessionId)}/files`,
      { method: 'POST', body },
    );
  },
  finalizeImport: (galleryId: string, sessionId: string) =>
    request<Job>(`/galleries/${id(galleryId)}/import-sessions/${id(sessionId)}/finalize`, {
      method: 'POST',
    }),
  activate: (galleryId: string, version: string, activeVersion: string | null) =>
    request<Gallery>(`/galleries/${id(galleryId)}/versions/${id(version)}/activate`, {
      method: 'POST',
      body: json({ expected_active_version_id: activeVersion }),
    }),
  jobs: (offset = 0) => request<PageResult<Job>>(`/jobs?limit=100&offset=${offset}`),
  job: (jobId: string) => request<Job>(`/jobs/${id(jobId)}`),
  search(image: File, bbox: BBox, galleryId: string, topN: number, operationKey: string) {
    const body = new FormData();
    body.append('image', image);
    body.append('bbox', json(bbox));
    body.append('gallery_id', galleryId);
    body.append('top_n', String(topN));
    return request<SearchResult>('/searches', {
      method: 'POST',
      body,
      headers: { 'Idempotency-Key': operationKey },
    });
  },
  searches: (offset = 0) =>
    request<PageResult<SearchResult>>(`/searches?limit=100&offset=${offset}`),
  searchResult: (searchId: string) => request<SearchResult>(`/searches/${id(searchId)}`),
  replay: (searchId: string) =>
    request<SearchResult>(`/searches/${id(searchId)}/replays`, { method: 'POST' }),
  review: (
    searchId: string,
    observationId: string,
    verdict: Verdict,
    note: string,
    annotations: ReviewAnnotationDraft[],
  ) =>
    request<Review>(`/searches/${id(searchId)}/reviews`, {
      method: 'POST',
      body: json({ observation_id: observationId, verdict, note, annotations }),
    }),
  export: (searchId: string, format: 'json' | 'csv' | 'pdf' | 'case') =>
    request<Job>(`/searches/${id(searchId)}/exports`, { method: 'POST', body: json({ format }) }),
  models: () => request<{ items: ModelInfo[] }>('/models'),
  calibrations: () => request<{ items: Record<string, unknown>[] }>('/calibrations'),
  system: () => request<SystemInfo>('/system'),
};
