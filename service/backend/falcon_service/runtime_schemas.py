"""Internal inference HTTP contract; feature matrices travel as safe NPZ bytes."""

from typing import Literal

from pydantic import BaseModel, Field, JsonValue, model_validator


class RuntimeLiveResponse(BaseModel):
    status: Literal["ok"]


class RuntimeCapabilities(BaseModel):
    mode: str = Field(min_length=1, max_length=64)
    model_id: str = Field(min_length=1, max_length=256)
    dimension: int = Field(ge=1, le=16384)
    rerank_window: int = Field(ge=0, le=50)
    calibration_policy: dict[str, JsonValue] | None
    real_model_connected: bool
    device: str = Field(min_length=1, max_length=64)
    precision: str = Field(min_length=1, max_length=64)
    weights_bytes: int = Field(ge=0, le=2_000_000_000)

    @model_validator(mode="after")
    def preview_never_claims_a_model(self):
        if self.mode == "preview" and self.real_model_connected:
            raise ValueError("Preview mode cannot claim a connected trained model")
        if self.mode != "preview" and not self.real_model_connected:
            raise ValueError("A non-preview runtime must declare a connected trained model")
        if self.real_model_connected and not self.weights_bytes:
            raise ValueError("A connected trained model must report nonzero model weights")
        if not self.real_model_connected and self.weights_bytes:
            raise ValueError("A runtime without a connected model cannot report model weights")
        return self


class ValidationIssue(BaseModel):
    loc: list[str | int]
    msg: str
    type: str
    input: JsonValue = None
    ctx: dict[str, JsonValue] | None = None


class RuntimeErrorResponse(BaseModel):
    detail: str | list[ValidationIssue]


RUNTIME_ERRORS = {
    status: {"model": RuntimeErrorResponse, "description": description}
    for status, description in {
        401: "Internal runtime bearer credential required",
        413: "Feature message exceeds 64 MiB",
        422: "Invalid request fields, image, bounding box or feature archive",
        429: "Runtime concurrency capacity exhausted; retry after one second",
    }.items()
}
RUNTIME_ERRORS[429]["headers"] = {"Retry-After": {"schema": {"type": "string", "const": "1"}}}

FEATURE_WIRE_DESCRIPTION = (
    "NPZ archive, maximum 64 MiB including expanded contents; load with allow_pickle=False. "
    "Required arrays: embeddings float32[N,D] (1<=N<=51, 1<=D<=16384), model_id scalar Unicode string. "
    "Optional tokens floating[N,T,C]. All numeric values must be finite. One shared model_id per message. "
)


def feature_response(description: str) -> dict:
    return {
        200: {
            "description": FEATURE_WIRE_DESCRIPTION + description,
            "content": {"application/octet-stream": {"schema": {"type": "string", "format": "binary"}}},
        }
    }


RERANK_REQUEST = {
    "requestBody": {
        "required": True,
        "description": FEATURE_WIRE_DESCRIPTION
        + "Row 0 is the query; rows 1..N-1 are candidates. Required extra: similarities numeric[N-1].",
        "content": {"application/octet-stream": {"schema": {"type": "string", "format": "binary"}}},
    }
}
