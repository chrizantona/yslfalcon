"""Durable job responses, including historical explanation results."""

from typing import Literal

from pydantic import Field

from .common import Page, ResponseModel, Timestamp
from .imports import ImportResult
from .reports import ExportResult


class LegacyAnalysisSizes(ResponseModel):
    query: tuple[int, int]
    candidate: tuple[int, int]


class LegacyExplanationResult(ResponseModel):
    """Read-only compatibility for durable jobs created before maps were removed."""

    method: Literal["occlusion_sensitivity"]
    score_kind: Literal["analysis_crop_embedding_cosine"]
    analysis_max_edge: int
    analysis_sizes: LegacyAnalysisSizes
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


class JobResponse(ResponseModel):
    id: str
    kind: Literal["import", "explanation", "export"]
    status: Literal["pending", "running", "completed", "failed"]
    progress: int = Field(ge=0, le=100)
    message: str
    result: ImportResult | LegacyExplanationResult | ExportResult | None
    error: str | None
    attempts: int = Field(ge=0)
    created_at: Timestamp
    completed_at: Timestamp | None


JobList = Page[JobResponse]
