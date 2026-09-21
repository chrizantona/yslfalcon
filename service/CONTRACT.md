# Service implementation contract

This file describes the service boundaries. Public API lives under `/api/v1`; internal runtime under `/internal/v1`. All timestamps are UTC ISO-8601; IDs are opaque strings; business data is scoped to the authenticated workspace. Images are uploads, never arbitrary URLs. Public errors use `{error: {code, message}, request_id}` and return the same ID in `X-Request-ID`; a safe supplied request ID is preserved. Auth uses `Authorization: Bearer <token>` or an HttpOnly session cookie from login. Cookie mutations also require an allowed `Origin` and `X-CSRF-Token`.

The exact request/response models, status codes and security schemes are published in `/api/v1/openapi.json` and the local `/docs` reference. Shapes below summarize those models; [response_schemas.py](backend/falcon_service/response_schemas.py) is their implementation. The internal runtime uses its separate typed `{detail: ...}` error contract from [runtime_schemas.py](backend/falcon_service/runtime_schemas.py).

**User update:** trained model integration is intentionally disabled. Only the explicit stub/preview adapter runs. RepositoryEncoder is an unavailable-adapter guard; no weights are loaded or downloaded. All service operations remain real, with uncalibrated placeholder feature computation.

## Python inference port

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

## Public HTTP contract

`POST /api/v1/auth/session {token}` sets HttpOnly SameSite cookie; DELETE clears it. GET `/api/v1/auth/me` returns role/workspace. Bearer works for integrations; roles viewer/operator/admin. Health live unauthenticated; business endpoints authenticated.

- GET `/capabilities`: engine mode/model_id/dimension, profiles, limits, service version.
- GET `/system`: dependencies, jobs counts, model status, timing summaries (real measured values only).
- GET/POST `/galleries`: list `{items:[...],total,next_offset}` with `limit`/`offset`; create `{name,description?}` -> gallery `{id,name,description,active_version_id,observation_count,created_at}`.
- GET `/galleries/{id}`; GET `/galleries/{id}/versions` -> `{items:[{id,number,status,count,model_id,created_at,...}]}`.
- POST `/galleries/{id}/imports`: legacy multipart `files` repeated JPEG/PNG, optional `manifest` JSON mapping filename to `{bbox:[x,y,w,h],external_id?}`. Return 202 job. One complete replacement snapshot per request, up to 500 files by default; this legacy route does not provide a resumable session or import idempotency key. Missing bbox means full frame, displayed explicitly.
- POST/GET `/galleries/{id}/import-sessions`, GET `/galleries/{id}/import-sessions/{session}`, POST `.../{session}/files`, POST `.../{session}/finalize`: resumable official CSV import described below; admin role required.
- POST `/galleries/{id}/versions/{version}/activate`: admin; required JSON `{expected_active_version_id: string|null}` performs atomic compare-and-set activation of a compatible ready snapshot. Import never activates the snapshot automatically.
- GET `/jobs` -> paginated `{items:[...],total,next_offset}`; GET `/jobs/{id}` -> `{id,kind,status,progress,message,result,error,attempts,created_at,completed_at}`. Statuses pending/running/completed/failed. Jobs are durable/recoverable with bounded retry; result is a typed import, explanation or export result.
- POST `/searches`: multipart `image`, `bbox` JSON array optional, `gallery_id`, `top_n` (1..50). Optional Idempotency-Key. Return saved search directly: `{id,decision,query:{image_url,crop_url,bbox,width,height},gallery_id,gallery_version_id,model_id,query_confidence,threshold,confidence_kind,calibration_reason,ranked_candidates:[{observation_id,rank,cosine,score,accepted,image_url,crop_url}],accepted_candidates:[],timings,created_at,reviews:[]}`. Empty/invalid gallery is typed error, not refusal. Only completed saved result changes UI ranking.
- GET `/searches` -> paginated `{items:[...],total,next_offset}`; GET `/searches/{id}` -> same immutable response plus reviews; POST `/searches/{id}/replays` -> new result with pinned version (or explicit unavailable).
- POST `/searches/{id}/reviews` `{observation_id?, verdict:"match"|"different"|"uncertain",note?}` -> review, independent of model.
- POST `/searches/{id}/explanations` `{observation_id}` -> job. Job result occlusion maps with method metadata.
- POST `/searches/{id}/exports` `{format:"json"|"csv"|"pdf"}` -> job with result `{download_url}`. GET `/exports/{id}/download`, scoped.
- GET `/observations/{id}/image`, `/observations/{id}/crop`; query originals through `/searches/{id}/image`, `/searches/{id}/crop`; scoped cookie or bearer auth, never query-string secrets.
- GET `/models`, `/calibrations` -> real local configured data, no invented performance.

## Resumable CSV import

All paths in this section are under `/api/v1/galleries/{gallery_id}/import-sessions`.

| Method and suffix | Request | Response |
|---|---|---|
| `POST /` | JSON `{manifest_csv: string}` and required `Idempotency-Key` of 1–128 characters | 201 `ImportSessionResponse`. The same key and parsed manifest return the same session; changed input returns 409. |
| `GET /` | None | `{items: ImportSessionResponse[]}`, newest first. The list is workspace/gallery scoped and currently unpaginated. |
| `GET /{session_id}` | None | `ImportSessionResponse`, including missing and received image IDs. |
| `POST /{session_id}/files` | Multipart `files` repeated; bounded to the session's advertised limits | Updated `ImportSessionResponse`; the complete chunk is validated before its receipts are committed. |
| `POST /{session_id}/finalize` | No body required | 202 `JobResponse`; repeated and concurrent calls return the same job and version. Missing files produce 409 `import_incomplete`. |

The collection URL is `/import-sessions` without a required trailing slash. CSV requires `image_id,x,y,w,h`; additional columns are ignored. UTF-8 BOM is accepted. IDs have surrounding whitespace trimmed and must be unique plain basenames or stems, at most 255 characters, with no path separators or control characters. JPEG/PNG extensions are matched without case sensitivity; stems remain case-sensitive. A stem expands to `.jpg`, `.jpeg` or `.png`. Ambiguous IDs such as `car` together with `car.jpg`, or multiple uploaded filenames for one ID, are rejected. No ID, filename, hash, CSV position or camera field is used as a ranking feature.

Coordinates are finite integer-valued numbers with absolute value at most 10 000 000; `w`/`h` must be positive. Negative `x`/`y` are allowed when the box intersects the image; the validated box is clipped to the actual dimensions during upload. The original CSV box remains in `import_items.requested_bbox`. A complete box outside the image is rejected. The server validates decoded image contents rather than trusting extension or MIME type.

`ImportSessionResponse` contains:

```text
id, gallery_id, status: "uploading" | "finalized",
expected_count, uploaded_count,
missing_image_ids: string[], received_image_ids: string[],
validation: {ready: boolean, issues: [{code, message, image_id: string|null}]},
job_id: string|null, version_id: string|null, created_at,
limits: {max_chunk_files, max_chunk_bytes, max_upload_bytes}
```

`validation.ready` means every expected file has been received; it does not mean that features have been calculated or that the version is active. When files are missing, `issues` contains a `missing_files` summary. Invalid CSV or image chunks return the public error envelope and leave previously committed receipts intact. A repeated filename with identical bytes is acknowledged without duplication, including after finalization. Changed bytes or a different filename for a received ID produce 409 `upload_conflict`.

Creation stores only the upload session and manifest items. Finishing a complete session atomically creates its immutable `GalleryVersion` in `preparing`, its durable import job and their session references. The worker publishes all observation rows, all database feature artifacts and `ready` status together. No chunk publishes a partial searchable gallery. The active version changes only through the separate activation route.

Limits reflect the actual implementation:

- A manifest contains 1–10 000 images. The browser accepts a CSV file up to 2 MiB; the API string field allows up to 2 097 152 characters. JSON request bodies are capped at `min(FALCON_MAX_REQUEST_BYTES, 4 MiB)` in bytes, including JSON encoding overhead.
- Each chunk contains at most `min(FALCON_MAX_IMPORT_FILES, 100)` files. A session can exceed the legacy default of 500 files by using multiple chunks.
- Total multipart body size defaults to 256 MiB (`FALCON_MAX_REQUEST_BYTES`), including multipart overhead. Nginx independently caps request bodies at 256 MiB. Clients must leave room for multipart framing.
- `FALCON_MAX_UPLOAD_BYTES` defaults to 20 MiB. The image decoder independently enforces a maximum of 20 MiB and 24 megapixels; raising the environment setting alone does not change the decoder limit. JPEG/PNG must be single-frame and already use the supplied EXIF orientation.

## Database features and compatibility

PostgreSQL is used in Compose; SQLite supports tests/local development. New gallery vectors are authoritative rows in `feature_artifacts`, not files in the shared volume. `FeatureStore.prepare` validates a numeric archive and returns a row for the caller's snapshot transaction. `FeatureStore.read_many` loads the selected workspace/version, validates SHA-256, model identity, dimension and encoding, and yields feature bundles for exact ranking. Each archive uses `dtype=float32`, `encoding=numeric-npz-v1` and `allow_pickle=False`. A corrupt existing database row fails with 503 `feature_corrupt`; there is no fallback to a different cached vector.

The added schema consists of:

- `feature_artifacts`: primary key `key`, index on `workspace`, numeric NPZ `payload` (`BYTEA`/`BLOB`), `model_id`, `dimension`, `dtype`, `encoding`, `sha256`, `created_at`. `observations.feature_key` remains the application-level reference; no new foreign key is imposed on legacy observations.
- `import_sessions`: primary key `id`, indexes on `workspace` and `gallery_id`, unique `(workspace, idempotency_key)`, gallery/job/version foreign keys, request hash, status and revision.
- `import_items`: primary key `id`, index and foreign key on `session_id`, unique `(session_id, image_id)`, original bbox/order, received filename/checksum/image key and validated image geometry. Workspace access is enforced through the owning session.

Startup creates the three missing tables with their indexes through `create_all`; existing tables are not rebuilt or altered. PostgreSQL startup DDL is protected by the existing advisory transaction lock. When a legacy observation has no database feature row, `FeatureStore` reads and validates the bounded old NPZ, retains the original key and inserts a database record using conflict-safe insertion. **Old NPZ files must remain available until every required historical version has been read and backfilled.** Starting the new image does not backfill all versions. A missing legacy file before backfill returns 503 `feature_unavailable`.

New snapshots need no NPZ cache. Gallery features are loaded into memory for exact dot-product search; relational indexes and the optional exact `FAISS IndexFlatIP` are not ANN. Query feature vectors remain transient. ANN scale or million-observation performance is not claimed.

Back up the database and original/report/legacy-feature volume consistently before upgrading. **Reverting only to the previous Docker image cannot read newly created database-only feature artifacts.** A rollback requires restoration of the matching pre-upgrade database and file backup, or a separately prepared compatible data conversion; automatic downgrade is not implemented. A pre-upgrade restore does not retain operations written after that backup. Future changes to existing columns require explicit versioned migrations.

## Browser and verification

React/TypeScript consumes these routes with same-origin cookies (`/api` proxy). Screens cover search/comparison, galleries with resumable CSV uploads and jobs, history/export, quality/model status, integration and system state. Product language is Russian. The UI has no fabricated gallery, camera or performance data and carries a persistent preview badge. API, worker and runtime run separately in Compose. The shared file adapter stores original images, reports and legacy migration input using server-generated paths.

Behavioral coverage includes 750-file chunked import, restart/resume, repeated uploads, concurrent finalization, validation and workspace isolation; database vectors are tested across missing caches, legacy migration, corruption, worker failure, replay and export. See [test_import_sessions.py](tests/test_import_sessions.py), [test_feature_store.py](tests/test_feature_store.py) and [IMPLEMENTATION](docs/IMPLEMENTATION.md). Exact commands, counts and verified deployment checks belong to [VALIDATION](docs/VALIDATION.md).
