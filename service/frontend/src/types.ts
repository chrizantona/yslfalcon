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
  real_model_connected: boolean;
  device: string | null;
  precision: string | null;
  weights_bytes: number | null;
}
export interface Capabilities {
  service_version: string;
  engine: Engine;
  limits: { max_upload_bytes: number; max_import_files: number; max_request_bytes: number };
  profiles: { id: string; name: string; top_n_max: number }[];
  features: {
    immutable_snapshots: boolean;
    replay: boolean;
    occlusion_explanations: boolean;
    durable_jobs: boolean;
    exports: ('json' | 'csv' | 'pdf' | 'case')[];
    expert_board: boolean;
    gallery_version_diff: boolean;
    offline_case_export: boolean;
    trained_model_connected: boolean;
  };
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
export interface GalleryVersionDiffItem {
  identity: string;
  external_id: string | null;
  change: 'added' | 'removed' | 'changed';
  from_observation_id: string | null;
  to_observation_id: string | null;
  image_changed: boolean;
  bbox_changed: boolean;
  from_bbox: BBox | null;
  to_bbox: BBox | null;
}
export interface GalleryVersionDiff {
  gallery_id: string;
  from_version_id: string;
  to_version_id: string;
  from_number: number;
  to_number: number;
  added: number;
  removed: number;
  changed: number;
  unchanged: number;
  total_changes: number;
  items: GalleryVersionDiffItem[];
  next_offset: number | null;
}
export interface ImportSession {
  id: string;
  gallery_id: string;
  status: 'uploading' | 'finalized';
  expected_count: number;
  uploaded_count: number;
  missing_image_ids: string[];
  received_image_ids: string[];
  validation: {
    ready: boolean;
    issues: { code: string; message: string; image_id: string | null }[];
  };
  job_id: string | null;
  version_id: string | null;
  created_at: string;
  limits: { max_chunk_files: number; max_chunk_bytes: number; max_upload_bytes: number };
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
  bbox: BBox;
  bbox_source: 'explicit' | 'full_frame';
}
export type Verdict = 'match' | 'different' | 'uncertain';
export type AnnotationKind = 'match' | 'conflict' | 'unclear';
export interface ReviewAnnotationDraft {
  image_ref: string;
  surface: 'crop' | 'original';
  kind: AnnotationKind;
  x: number;
  y: number;
}
export interface ReviewAnnotation extends ReviewAnnotationDraft {
  id: string;
}
export interface Review {
  id: string;
  search_id: string;
  observation_id: string | null;
  verdict: Verdict;
  note: string;
  actor: string;
  created_at: string;
  annotations: ReviewAnnotation[];
}
export interface SearchResult {
  id: string;
  decision: string;
  query: { image_url: string; crop_url: string; bbox: BBox; width: number; height: number };
  gallery_id: string;
  gallery_version_id: string;
  gallery_manifest_hash: string;
  model_id: string;
  query_confidence: number | null;
  threshold: number | null;
  confidence_kind: string;
  calibration_reason?: string | null;
  ranked_candidates: Candidate[];
  accepted_candidates: Candidate[];
  timings: Record<string, number>;
  created_at: string;
  replay_of?: string | null;
  reviews: Review[];
}
export interface ModelInfo {
  id: string;
  mode: string;
  dimension: number;
  status: string;
  trained_model_connected: boolean;
  device: string | null;
  precision: string | null;
  weights_bytes: number | null;
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
