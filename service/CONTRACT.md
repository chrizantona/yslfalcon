# Service implementation contract

This file coordinates implementation. Public API lives under `/api/v1`; internal runtime under `/internal/v1`. All timestamps UTC ISO-8601; IDs opaque strings; all data scoped to the authenticated workspace. Images are uploads, never arbitrary URLs. Errors: `{error: {code, message}, request_id}`. Auth uses `Authorization: Bearer <token>` or HttpOnly session cookie from login.

**User update:** trained model integration is intentionally disabled. Only the explicit stub/preview adapter runs. RepositoryEncoder is an unavailable-adapter guard; no weights are loaded or downloaded. All service operations remain real, with uncalibrated placeholder feature computation.

## Python inference port (owned by inference implementation)

Module `falcon_service.inference` exports:

- `FeatureBundle(embedding: numpy.ndarray, tokens: numpy.ndarray | None, model_id: str)`.
- `PreviewEncoder()`; attributes `model_id`, `mode="preview"`, `dimension`; `encode(image: bytes, bbox: tuple[int,int,int,int] | None) -> FeatureBundle`.
- `RepositoryEncoder(manifest_path: str, device: str="cpu")` is reserved for future integration and raises a clear unavailable error in this release.
- `RemoteEncoder(url: str, token: str, timeout: float=60)` with the same interface; runtime capabilities supply identity.
- Each encoder has `rerank(query: FeatureBundle, candidates: list[FeatureBundle], similarities: numpy.ndarray) -> numpy.ndarray`; preview returns unchanged cosine.
- `rank_gallery(query: FeatureBundle, gallery: list[FeatureBundle], encoder, top_n: int=10, policy: dict | None=None) -> dict`: `decision`, `ranked` list of `{index, cosine, score, accepted}`, `query_confidence`, `threshold`, `confidence_kind`, `calibration_reason`. Only top-1 accepted, all others alternatives. No valid policy => `calibration_unavailable`; preview always uncalibrated. Ranking independent of other queries. A policy pins model_id, score definition and gallery-size range; no guessed threshold.
- `explain_pair(encoder, query_image: bytes, query_bbox, candidate_image: bytes, candidate_bbox, grid: int=6) -> dict`: real occlusion sensitivity, numeric maps and method/score metadata; separate from main request.
- `validate_image(image: bytes, bbox=None) -> tuple[bytes, tuple[int,int,int,int], int, int]`: canonical safe PNG/JPEG original bytes, validated/clipped bbox, width, height. Reject corrupt/decompression bombs/nonfinite/outside boxes; no silent EXIF coordinate changes. `crop_image(image: bytes, bbox) -> bytes`.
- Invalid image/bbox => ValueError. Missing/unusable model => explicit RuntimeError.

## Public HTTP contract (backend owns implementation)

`POST /api/v1/auth/session {token}` sets HttpOnly SameSite cookie; DELETE clears it. GET `/api/v1/auth/me` returns role/workspace. Bearer works for integrations; roles viewer/operator/admin. Health live unauthenticated; business endpoints authenticated.

- GET `/capabilities`: engine mode/model_id/dimension, profiles, limits, service version.
- GET `/system`: dependencies, jobs counts, model status, timing summaries (real measured values only).
- GET/POST `/galleries`: list `{items:[...]}`; create `{name,description?}` -> gallery `{id,name,description,active_version_id,observation_count,created_at}`.
- GET `/galleries/{id}`; GET `/galleries/{id}/versions` -> `{items:[{id,number,status,count,model_id,created_at,...}]}`.
- POST `/galleries/{id}/imports`: multipart `files` repeated JPEG/PNG, optional `manifest` JSON mapping filename to `{bbox:[x,y,w,h],external_id?}`. Return 202 job. Import is replacement snapshot. Missing bbox means full frame, displayed explicitly.
- POST `/galleries/{id}/versions/{version}/activate`: admin, atomic CAS compatible snapshot activation. Import prepares ready snapshot, activation explicit.
- GET `/jobs`, GET `/jobs/{id}` -> `{id,kind,status,progress,message,result,error,created_at}`; statuses pending/running/completed/failed. Jobs durable/recoverable, bounded retry.
- POST `/searches`: multipart `image`, `bbox` JSON array optional, `gallery_id`, `top_n` (1..50). Optional Idempotency-Key. Return saved search directly: `{id,decision,query:{image_url,crop_url,bbox,width,height},gallery_id,gallery_version_id,model_id,query_confidence,threshold,confidence_kind,calibration_reason,ranked_candidates:[{observation_id,rank,cosine,score,accepted,image_url,crop_url}],accepted_candidates:[],timings,created_at,reviews:[]}`. Empty/invalid gallery is typed error, not refusal. Only completed saved result changes UI ranking.
- GET `/searches` -> `{items:[...]}`; GET `/searches/{id}` -> same immutable response plus reviews; POST `/searches/{id}/replays` -> new result with pinned version (or explicit unavailable).
- POST `/searches/{id}/reviews` `{observation_id?, verdict:"match"|"different"|"uncertain",note?}` -> review, independent of model.
- POST `/searches/{id}/explanations` `{observation_id}` -> job. Job result occlusion maps with method metadata.
- POST `/searches/{id}/exports` `{format:"json"|"csv"|"pdf"}` -> job with result `{download_url}`. GET `/exports/{id}/download`, scoped.
- GET `/observations/{id}/image`, `/observations/{id}/crop`; query originals through `/searches/{id}/image`, `/searches/{id}/crop`; scoped cookie or bearer auth, never query-string secrets.
- GET `/models`, `/calibrations` -> real local configured data, no invented performance.

Frontend React/TypeScript consumes these routes with same-origin cookies (`/api` proxy). Screens: search/comparison, galleries+jobs, history/export, quality+model status, integration, system. Russian product language. No fabricated gallery, camera or performance data. Explicit persistent preview badge. Main API may consolidate Catalog; worker and model runtime run separately in Compose. PostgreSQL in Compose, SQLite supported for tests/local development. Shared artifact volume behind a storage adapter, immutable files with server-generated paths. Atomic snapshots and persisted idempotency must hold across API restarts.

Implementation ownership: backend agent owns `backend/falcon_service/` except `inference/`, `model_runtime.py`, `cli.py`; inference agent owns those exclusions and inference tests; frontend agent owns `frontend/`. Root owns packaging/Compose/docs/scripts/integration checks. Test filenames must stay distinct.
