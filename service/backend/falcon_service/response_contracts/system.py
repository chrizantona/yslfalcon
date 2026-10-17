"""Health, model capabilities, dependencies and timing summaries."""

from typing import Literal

from pydantic import Field, JsonValue

from .common import Items, ResponseModel


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
    hnsw_index: bool
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
