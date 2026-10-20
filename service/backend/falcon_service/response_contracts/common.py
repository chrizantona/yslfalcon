"""Shared wire types, strict responses, errors and collection envelopes."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

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
