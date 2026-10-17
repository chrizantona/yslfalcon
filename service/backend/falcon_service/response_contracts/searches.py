"""Immutable search results, query images and operator reviews."""

from typing import Literal

from pydantic import Field, JsonValue

from .common import BBox, Page, ResponseModel, Timestamp


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
    query_confidence: float | None = Field(
        description=(
            "Top-1 z-score over the exact gallery population, not a probability; null when "
            "the search profile does not compute exact population statistics."
        )
    )
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
