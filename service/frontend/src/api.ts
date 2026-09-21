import type {
  BBox,
  Capabilities,
  Explanation,
  Gallery,
  GalleryVersion,
  Job,
  ModelInfo,
  PageResult,
  Review,
  SearchResult,
  Session,
  SystemInfo,
  Verdict,
} from './types';

const base = '/api/v1';
let csrfToken = '';
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
  const headers = new Headers(options.headers);
  if (options.body && !(options.body instanceof FormData))
    headers.set('Content-Type', 'application/json');
  const method = options.method || 'GET';
  if (!['GET', 'HEAD', 'OPTIONS'].includes(method) && csrfToken)
    headers.set('X-CSRF-Token', csrfToken);
  const response = await fetch(`${base}${path}`, {
    ...options,
    headers,
    credentials: 'same-origin',
  });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    if (response.status === 401 && path !== '/auth/session' && path !== '/auth/me')
      window.dispatchEvent(new Event('falcon:unauthorized'));
    throw new ApiError(
      body.error?.message || body.detail || `Ошибка запроса (${response.status})`,
      response.status,
      body.error?.code || 'request_failed',
      body.request_id,
    );
  }
  return response.status === 204 ? (undefined as T) : response.json();
}
const json = (value: unknown) => JSON.stringify(value);
const id = encodeURIComponent;
export const api = {
  async session() {
    const session = await request<Session>('/auth/me');
    csrfToken = session.csrf_token;
    return session;
  },
  async login(token: string) {
    const session = await request<Session>('/auth/session', {
      method: 'POST',
      body: json({ token }),
    });
    csrfToken = session.csrf_token;
    return session;
  },
  async logout() {
    await request<void>('/auth/session', { method: 'DELETE' });
    csrfToken = '';
  },
  capabilities: () => request<Capabilities>('/capabilities'),
  galleries: (offset = 0) => request<PageResult<Gallery>>(`/galleries?limit=100&offset=${offset}`),
  createGallery: (name: string, description: string) =>
    request<Gallery>('/galleries', { method: 'POST', body: json({ name, description }) }),
  versions: (galleryId: string) =>
    request<{ items: GalleryVersion[] }>(`/galleries/${id(galleryId)}/versions`),
  async importGallery(galleryId: string, files: File[], manifest?: File) {
    const body = new FormData();
    files.forEach((file) => body.append('files', file));
    if (manifest) body.append('manifest', await manifest.text());
    return request<Job>(`/galleries/${id(galleryId)}/imports`, { method: 'POST', body });
  },
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
  review: (searchId: string, observationId: string, verdict: Verdict, note: string) =>
    request<Review>(`/searches/${id(searchId)}/reviews`, {
      method: 'POST',
      body: json({ observation_id: observationId, verdict, note }),
    }),
  explain: (searchId: string, observationId: string) =>
    request<Job>(`/searches/${id(searchId)}/explanations`, {
      method: 'POST',
      body: json({ observation_id: observationId }),
    }),
  export: (searchId: string, format: 'json' | 'csv' | 'pdf') =>
    request<Job>(`/searches/${id(searchId)}/exports`, { method: 'POST', body: json({ format }) }),
  models: () => request<{ items: ModelInfo[] }>('/models'),
  calibrations: () => request<{ items: Record<string, unknown>[] }>('/calibrations'),
  system: () => request<SystemInfo>('/system'),
};
export function explanationFromJob(job: Job): Explanation | null {
  const result = job.result;
  return result && Array.isArray(result.query_map) && Array.isArray(result.candidate_map)
    ? (result as unknown as Explanation)
    : null;
}
