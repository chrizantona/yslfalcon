"""Gallery snapshots, version lists and version comparison contracts."""

from typing import Literal

from pydantic import Field

from .common import BBox, Items, Page, ResponseModel, Timestamp


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
