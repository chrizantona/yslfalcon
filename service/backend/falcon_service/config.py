"""Validated configuration; secrets never appear in public capabilities."""

from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FALCON_", extra="ignore")

    database_url: str = "sqlite:///./falcon.db"
    storage_root: Path = Path("./falcon-data")
    auth_tokens: dict[str, dict[str, str]] = Field(default_factory=dict, repr=False)
    engine_mode: Literal["preview", "repository", "remote"] = "preview"
    model_manifest: str | None = None
    expected_model_id: str | None = None
    expected_weights_bytes: int | None = Field(default=None, ge=0, le=2_000_000_000)
    runtime_url: str = "http://model-runtime:8090"
    runtime_token: str = Field(default="", repr=False)
    model_device: str = "cpu"
    calibration_path: Path | None = None
    allowed_origins: list[str] = Field(default_factory=lambda: ["http://localhost:8080"])
    cookie_secure: bool = True
    local_auto_login: bool = False
    session_hours: int = Field(default=12, ge=1, le=168)
    max_upload_bytes: int = Field(default=20 * 1024 * 1024, ge=1024)
    max_request_bytes: int = Field(default=256 * 1024 * 1024, ge=1024)
    max_import_files: int = Field(default=500, ge=1, le=10000)
    job_lease_seconds: int = Field(default=120, ge=5)
    job_max_attempts: int = Field(default=3, ge=1, le=10)
    worker_poll_seconds: float = Field(default=1, gt=0)

    @model_validator(mode="after")
    def validate_security(self):
        for token, identity in self.auth_tokens.items():
            if len(token) < 24:
                raise ValueError("Access tokens must contain at least 24 characters")
            if identity.get("role") not in {"viewer", "operator", "admin"}:
                raise ValueError("Each access token needs a viewer, operator or admin role")
            if not identity.get("workspace") or len(identity["workspace"]) > 128:
                raise ValueError("Each access token needs a workspace of 1..128 characters")
        return self
