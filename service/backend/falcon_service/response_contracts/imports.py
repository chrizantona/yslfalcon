"""Import validation, upload session and completed import contracts."""

from typing import Literal

from pydantic import Field

from .common import Items, ResponseModel, Timestamp


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


class ImportResult(ResponseModel):
    gallery_id: str
    version_id: str
    count: int
    status: Literal["ready"]
    manifest_hash: str
