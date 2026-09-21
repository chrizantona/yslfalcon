"""Storage schema. Immutable snapshots/results are distinct from mutable jobs."""

from datetime import datetime
from typing import Any

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base, now


class Gallery(Base):
    __tablename__ = "galleries"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    workspace: Mapped[str] = mapped_column(String(128), index=True)
    name: Mapped[str] = mapped_column(String(160))
    description: Mapped[str] = mapped_column(Text, default="")
    active_version_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    next_version: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)


class GalleryVersion(Base):
    __tablename__ = "gallery_versions"
    __table_args__ = (UniqueConstraint("gallery_id", "number"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    workspace: Mapped[str] = mapped_column(String(128), index=True)
    gallery_id: Mapped[str] = mapped_column(ForeignKey("galleries.id"), index=True)
    number: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(24), default="preparing")
    model_id: Mapped[str] = mapped_column(String(256))
    count: Mapped[int] = mapped_column(Integer, default=0)
    manifest_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)


class Observation(Base):
    __tablename__ = "observations"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    workspace: Mapped[str] = mapped_column(String(128), index=True)
    version_id: Mapped[str] = mapped_column(ForeignKey("gallery_versions.id"), index=True)
    ordinal: Mapped[int] = mapped_column(Integer)
    external_id: Mapped[str | None] = mapped_column(String(256), nullable=True)
    image_key: Mapped[str] = mapped_column(String(256))
    feature_key: Mapped[str] = mapped_column(String(256))
    bbox: Mapped[list[int]] = mapped_column(JSON)
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)
    bbox_source: Mapped[str] = mapped_column(String(24), default="explicit")


class Search(Base):
    __tablename__ = "searches"
    __table_args__ = (UniqueConstraint("workspace", "idempotency_key"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    workspace: Mapped[str] = mapped_column(String(128), index=True)
    gallery_id: Mapped[str] = mapped_column(ForeignKey("galleries.id"))
    gallery_version_id: Mapped[str] = mapped_column(ForeignKey("gallery_versions.id"))
    model_id: Mapped[str] = mapped_column(String(256))
    idempotency_key: Mapped[str] = mapped_column(String(128))
    request_hash: Mapped[str] = mapped_column(String(64))
    state: Mapped[str] = mapped_column(String(24), default="pending", index=True)
    query: Mapped[dict[str, Any]] = mapped_column(JSON)
    top_n: Mapped[int] = mapped_column(Integer)
    result: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    claim_token: Mapped[str | None] = mapped_column(String(36), nullable=True)
    lease_until: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    replay_of: Mapped[str | None] = mapped_column(String(36), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)


class Review(Base):
    __tablename__ = "reviews"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    workspace: Mapped[str] = mapped_column(String(128), index=True)
    search_id: Mapped[str] = mapped_column(ForeignKey("searches.id"), index=True)
    observation_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    verdict: Mapped[str] = mapped_column(String(16))
    note: Mapped[str] = mapped_column(Text, default="")
    actor: Mapped[str] = mapped_column(String(16))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)


class Job(Base):
    __tablename__ = "jobs"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    workspace: Mapped[str] = mapped_column(String(128), index=True)
    kind: Mapped[str] = mapped_column(String(24))
    status: Mapped[str] = mapped_column(String(24), default="pending", index=True)
    progress: Mapped[int] = mapped_column(Integer, default=0)
    message: Mapped[str] = mapped_column(Text, default="Ожидает обработки")
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)
    result: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    claim_token: Mapped[str | None] = mapped_column(String(36), nullable=True)
    lease_until: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, index=True)
    available_at: Mapped[datetime] = mapped_column(DateTime, default=now)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class Export(Base):
    __tablename__ = "exports"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    workspace: Mapped[str] = mapped_column(String(128), index=True)
    search_id: Mapped[str] = mapped_column(ForeignKey("searches.id"))
    format: Mapped[str] = mapped_column(String(8))
    storage_key: Mapped[str] = mapped_column(String(256))
    sha256: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)


class BrowserSession(Base):
    __tablename__ = "browser_sessions"
    token_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    credential_hash: Mapped[str] = mapped_column(String(64))
    csrf_hash: Mapped[str] = mapped_column(String(64))
    expires_at: Mapped[datetime] = mapped_column(DateTime, index=True)
