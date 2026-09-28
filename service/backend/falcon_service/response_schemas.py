"""Public wire contracts, independent of database rows and private artifact keys."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

BBox = tuple[int, int, int, int]
Role = Literal["viewer", "operator", "admin"]
Timestamp = Annotated[str, Field(json_schema_extra={"format": "date-time"})]


class ResponseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ErrorDetail(ResponseModel):
    code: str = Field(description="Stable machine-readable error code.")
    message: str = Field(description="Human-readable description; never echoes credentials.")


class ErrorResponse(ResponseModel):
    error: ErrorDetail
    request_id: str = Field(description="Correlation ID also returned in X-Request-ID.")


ERROR_RESPONSES = {
    status: {"model": ErrorResponse, "description": description}
    for status, description in {
        400: "Invalid image, manifest or operation input",
        401: "Authentication required or credential revoked",
        403: "Insufficient role, invalid origin or CSRF token",
        404: "Object does not exist in this workspace",
        405: "HTTP method is not supported",
        409: "State, version or idempotency conflict; inspect error.code",
        410: "Referenced artifact is no longer available",
        413: "Request, file or batch exceeds an advertised limit",
        422: "Request fields do not conform to this API contract",
        500: "Unexpected service error; use request_id for diagnosis",
        503: "A required dependency is unavailable",
    }.items()
}


class Items[T](ResponseModel):
    items: list[T]


class Page[T](Items[T]):
    total: int = Field(ge=0)
    next_offset: int | None


class IdentityResponse(ResponseModel):
    role: Role
    workspace: str
    workspace_id: str
    csrf_token: str | None


class GalleryResponse(ResponseModel):
    id: str
    name: str
    description: str
    active_version_id: str | None
    observation_count: int = Field(ge=0)
    created_at: Timestamp


class GalleryVersionResponse(ResponseModel):
    id: str
    number: int
    status: Literal["preparing", "ready", "failed"]
    count: int = Field(ge=0)
    model_id: str
    active: bool
    manifest_hash: str | None
    created_at: Timestamp


GalleryList = Page[GalleryResponse]
GalleryVersionList = Items[GalleryVersionResponse]


class ImportIssue(ResponseModel):
    code: str
    message: str
    image_id: str | None


class ImportValidation(ResponseModel):
    ready: bool
    issues: list[ImportIssue]


class ImportLimits(ResponseModel):
    max_chunk_files: int
    max_chunk_bytes: int
    max_upload_bytes: int


class ImportSessionResponse(ResponseModel):
    id: str
    gallery_id: str
    status: Literal["uploading", "finalized"]
    expected_count: int = Field(ge=1)
    uploaded_count: int = Field(ge=0)
    missing_image_ids: list[str]
    received_image_ids: list[str]
    validation: ImportValidation
    job_id: str | None
    version_id: str | None
    created_at: Timestamp
    limits: ImportLimits


ImportSessionList = Items[ImportSessionResponse]


class ReviewAnnotationResponse(ResponseModel):
    id: str
    image_ref: str
    surface: Literal["crop", "original"]
    kind: Literal["match", "conflict", "unclear"]
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)


class ReviewResponse(ResponseModel):
    id: str
    search_id: str
    observation_id: str | None
    verdict: Literal["match", "different", "uncertain"]
    note: str
    actor: str
    created_at: Timestamp
    annotations: list[ReviewAnnotationResponse]


class QueryImage(ResponseModel):
    bbox: BBox = Field(description="Validated natural-image pixels: x, y, width, height.")
    width: int
    height: int
    bbox_source: Literal["explicit", "full_frame"]
    sha256: str
    image_url: str
    crop_url: str


class Candidate(ResponseModel):
    observation_id: str
    rank: int = Field(ge=1)
    cosine: float = Field(description="Embedding cosine similarity, not a probability.")
    score: float = Field(description="Ranking score, not a calibrated probability.")
    accepted: bool
    image_url: str
    crop_url: str
    bbox: BBox
    external_id: str | None
    bbox_source: Literal["explicit", "full_frame"]


class SearchTimings(ResponseModel):
    load_ms: float = Field(ge=0)
    encode_ms: float = Field(ge=0)
    rank_ms: float = Field(ge=0)
    total_ms: float = Field(ge=0)


class SearchResponse(ResponseModel):
    id: str
    decision: Literal["calibration_unavailable", "candidates", "no_confident_match"]
    query: QueryImage
    gallery_id: str
    gallery_version_id: str
    gallery_manifest_hash: str
    model_id: str
    query_confidence: float = Field(description="Top-1 z-score over this gallery, not a probability.")
    threshold: float | None
    confidence_kind: str
    calibration_reason: str | None
    calibration_policy: dict[str, JsonValue] | None = Field(
        description="Original versioned policy artifact, including any validation metadata. Null when absent."
    )
    ranked_candidates: list[Candidate]
    accepted_candidates: list[Candidate]
    timings: SearchTimings
    created_at: Timestamp
    replay_of: str | None
    reviews: list[ReviewResponse]


SearchList = Page[SearchResponse]


class ImportResult(ResponseModel):
    gallery_id: str
    version_id: str
    count: int
    status: Literal["ready"]
    manifest_hash: str


class AnalysisSizes(ResponseModel):
    query: tuple[int, int]
    candidate: tuple[int, int]


class ExplanationResult(ResponseModel):
    method: Literal["occlusion_sensitivity"]
    score_kind: Literal["analysis_crop_embedding_cosine"]
    analysis_max_edge: int
    analysis_sizes: AnalysisSizes
    grid: int
    baseline_score: float
    query_map: list[list[float]]
    candidate_map: list[list[float]]
    fill: Literal["crop_mean_rgb"]
    model_id: str
    mode: str
    semantics: str
    preview_warning: str | None
    search_id: str
    observation_id: str
    gallery_version_id: str


class ExportResult(ResponseModel):
    export_id: str
    download_url: str
    sha256: str


class JobResponse(ResponseModel):
    id: str
    kind: Literal["import", "explanation", "export"]
    status: Literal["pending", "running", "completed", "failed"]
    progress: int = Field(ge=0, le=100)
    message: str
    result: ImportResult | ExplanationResult | ExportResult | None
    error: str | None
    attempts: int = Field(ge=0)
    created_at: Timestamp
    completed_at: Timestamp | None


JobList = Page[JobResponse]


class GalleryVersionDiffItem(ResponseModel):
    identity: str
    external_id: str | None
    change: Literal["added", "removed", "changed"]
    from_observation_id: str | None
    to_observation_id: str | None
    image_changed: bool
    bbox_changed: bool
    from_bbox: BBox | None
    to_bbox: BBox | None


class GalleryVersionDiffResponse(ResponseModel):
    gallery_id: str
    from_version_id: str
    to_version_id: str
    from_number: int
    to_number: int
    added: int = Field(ge=0)
    removed: int = Field(ge=0)
    changed: int = Field(ge=0)
    unchanged: int = Field(ge=0)
    total_changes: int = Field(ge=0)
    items: list[GalleryVersionDiffItem]
    next_offset: int | None


class LiveResponse(ResponseModel):
    status: Literal["alive"]
    version: str


class ReadyResponse(ResponseModel):
    status: Literal["ready"]
    mode: str


class EngineCapabilities(ResponseModel):
    mode: str
    model_id: str | None
    dimension: int | None
    ready: bool
    real_model_connected: bool
    device: str | None
    precision: str | None
    weights_bytes: int | None = Field(default=None, ge=0, le=2_000_000_000)


class SearchProfile(ResponseModel):
    id: str
    name: str
    top_n_max: int


class ServiceLimits(ResponseModel):
    max_upload_bytes: int
    max_import_files: int
    max_request_bytes: int


class ServiceFeatures(ResponseModel):
    immutable_snapshots: bool
    replay: bool
    occlusion_explanations: bool
    durable_jobs: bool
    exports: list[Literal["json", "csv", "pdf", "case"]]
    expert_board: bool
    gallery_version_diff: bool
    offline_case_export: bool
    trained_model_connected: bool


class CapabilitiesResponse(ResponseModel):
    service_version: str
    engine: EngineCapabilities
    profiles: list[SearchProfile]
    limits: ServiceLimits
    features: ServiceFeatures
    notice: str


class ModelResponse(ResponseModel):
    id: str | None
    mode: str
    dimension: int | None
    status: Literal["ready", "unavailable"]
    trained_model_connected: bool
    device: str | None
    precision: str | None
    weights_bytes: int | None = Field(default=None, ge=0, le=2_000_000_000)
    description: str


ModelList = Items[ModelResponse]


class CalibrationList(ResponseModel):
    items: list[dict[str, JsonValue]]
    notice: str


class DependencyStatus(ResponseModel):
    status: Literal["ready", "unavailable"]


class ModelStatus(EngineCapabilities, DependencyStatus):
    pass


class Dependencies(ResponseModel):
    database: DependencyStatus
    storage: DependencyStatus
    model: ModelStatus


class JobCounts(ResponseModel):
    pending: int
    running: int
    completed: int
    failed: int


class Percentiles(ResponseModel):
    p50: float
    p95: float


class TimingSummary(ResponseModel):
    sample_count: int
    total_ms: Percentiles | None
    load_ms: Percentiles | None
    encode_ms: Percentiles | None
    rank_ms: Percentiles | None


class SystemResponse(ResponseModel):
    status: Literal["ready", "degraded"]
    dependencies: Dependencies
    jobs: JobCounts
    timings: TimingSummary
    timing_scope: str


class ImageDigest(ResponseModel):
    kind: Literal["query", "candidate"]
    observation_id: str | None = None
    sha256: str


class ReportIntegrity(ResponseModel):
    algorithm: Literal["sha256"]
    original_images: list[ImageDigest]


class SearchReport(ResponseModel):
    schema_version: Literal[1]
    purpose: Literal["visual_search_report"]
    search: SearchResponse
    integrity: ReportIntegrity
    notice: str
