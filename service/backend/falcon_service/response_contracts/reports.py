"""Export results and search reports with source image integrity."""

from typing import Literal

from .common import ResponseModel
from .searches import SearchResponse


class ExportResult(ResponseModel):
    export_id: str
    download_url: str
    sha256: str


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
