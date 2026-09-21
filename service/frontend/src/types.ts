export type Role = 'viewer' | 'operator' | 'admin';
export type BBox = [number, number, number, number];
export interface PageResult<T> {
  items: T[];
  total: number;
  next_offset: number | null;
}
export interface Session {
  role: Role;
  workspace: string;
  workspace_id: string;
  csrf_token: string;
}
export interface Engine {
  mode: string;
  model_id: string;
  dimension: number;
  ready: boolean;
}
export interface Capabilities {
  service_version: string;
  engine: Engine;
  limits: { max_upload_bytes: number; max_import_files: number; max_request_bytes: number };
  profiles: { id: string; name: string; top_n_max: number }[];
}
export interface Gallery {
  id: string;
  name: string;
  description?: string;
  active_version_id: string | null;
  observation_count: number;
  created_at: string;
}
export interface GalleryVersion {
  id: string;
  number: number;
  status: string;
  count: number;
  model_id: string;
  created_at: string;
}
export interface Job {
  id: string;
  kind: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  progress: number;
  message?: string;
  result?: Record<string, unknown>;
  error?: string | { message?: string; code?: string };
  created_at: string;
}
export interface Candidate {
  observation_id: string;
  rank: number;
  cosine: number;
  score: number;
  accepted: boolean;
  image_url: string;
  crop_url: string;
  external_id?: string;
}
export type Verdict = 'match' | 'different' | 'uncertain';
export interface Review {
  id: string;
  search_id: string;
  observation_id?: string;
  verdict: Verdict;
  note?: string;
  created_at: string;
}
export interface SearchResult {
  id: string;
  decision: string;
  query: { image_url: string; crop_url: string; bbox: BBox; width: number; height: number };
  gallery_id: string;
  gallery_version_id: string;
  model_id: string;
  query_confidence: number | null;
  threshold: number | null;
  confidence_kind: string;
  calibration_reason?: string | null;
  ranked_candidates: Candidate[];
  accepted_candidates: Candidate[];
  timings: Record<string, number>;
  created_at: string;
  reviews: Review[];
}
export interface ModelInfo {
  id: string;
  mode: string;
  dimension: number;
  status: string;
  manifest?: Record<string, unknown>;
}
export interface Explanation {
  method: string;
  score_kind: string;
  baseline_score: number;
  query_map: number[][];
  candidate_map: number[][];
  grid: number;
  fill: string;
  semantics: string;
  model_id: string;
  mode: string;
}
export interface SystemInfo {
  status: string;
  dependencies: Record<string, { status: string; mode?: string; model_id?: string }>;
  jobs: Record<string, number>;
  timings: {
    sample_count: number;
    total_ms?: { p50: number | null; p95: number | null } | null;
    [key: string]: unknown;
  };
}
export type Page = 'search' | 'galleries' | 'history' | 'quality' | 'integration' | 'system';
