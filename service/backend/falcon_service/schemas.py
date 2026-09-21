"""Small public command schemas; IDs are opaque and have no filesystem meaning."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Command(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class CreateGallery(Command):
    name: str = Field(min_length=1, max_length=160)
    description: str = Field(default="", max_length=5000)


class ActivateVersion(Command):
    expected_active_version_id: str | None


class CreateReview(Command):
    observation_id: str | None = None
    verdict: Literal["match", "different", "uncertain"]
    note: str = Field(default="", max_length=5000)


class CreateExplanation(Command):
    observation_id: str


class CreateExport(Command):
    format: Literal["json", "csv", "pdf"]
