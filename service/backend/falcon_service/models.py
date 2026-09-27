"""Storage schema. Immutable snapshots/results are distinct from mutable jobs."""

from datetime import datetime
from typing import Any

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, LargeBinary, String, Text, UniqueConstraint
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


class ObservationDigest(Base):
    """Image inventory checksum kept outside the legacy observation schema."""

    __tablename__ = "observation_digests"
    observation_id: Mapped[str] = mapped_column(
        ForeignKey("observations.id", ondelete="CASCADE"), primary_key=True
    )
    workspace: Mapped[str] = mapped_column(String(128), index=True)
    sha256: Mapped[str] = mapped_column(String(64))


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


class ReviewAnnotation(Base):
    """Immutable expert evidence pinned to a saved review and image."""

    __tablename__ = "review_annotations"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    workspace: Mapped[str] = mapped_column(String(128), index=True)
    review_id: Mapped[str] = mapped_column(ForeignKey("reviews.id"), index=True)
    image_ref: Mapped[str] = mapped_column(String(36))
    surface: Mapped[str] = mapped_column(String(8))
    kind: Mapped[str] = mapped_column(String(16))
    x: Mapped[float] = mapped_column(Float)
    y: Mapped[float] = mapped_column(Float)
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


class FeatureArtifact(Base):
    """Authoritative vectors; old file keys remain valid during additive backfill."""

    __tablename__ = "feature_artifacts"
    key: Mapped[str] = mapped_column(String(256), primary_key=True)
    workspace: Mapped[str] = mapped_column(String(128), index=True)
    model_id: Mapped[str] = mapped_column(String(256))
    dimension: Mapped[int] = mapped_column(Integer)
    dtype: Mapped[str] = mapped_column(String(16), default="float32")
    encoding: Mapped[str] = mapped_column(String(32), default="numeric-npz-v1")
    sha256: Mapped[str] = mapped_column(String(64))
    payload: Mapped[bytes] = mapped_column(LargeBinary)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)


class GalleryIndexArtifact(Base):
    """Rebuildable immutable HNSW graph for one gallery snapshot."""

    __tablename__ = "gallery_index_artifacts"
    version_id: Mapped[str] = mapped_column(
        ForeignKey("gallery_versions.id", ondelete="CASCADE"), primary_key=True
    )
    workspace: Mapped[str] = mapped_column(String(128), index=True)
    model_id: Mapped[str] = mapped_column(String(256))
    algorithm: Mapped[str] = mapped_column(String(64))
    metric: Mapped[str] = mapped_column(String(32))
    dimension: Mapped[int] = mapped_column(Integer)
    count: Mapped[int] = mapped_column(Integer)
    parameters: Mapped[dict[str, Any]] = mapped_column(JSON)
    artifact_key: Mapped[str] = mapped_column(String(256))
    sha256: Mapped[str] = mapped_column(String(64))
    size_bytes: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)


class ImportSession(Base):
    __tablename__ = "import_sessions"
    __table_args__ = (UniqueConstraint("workspace", "idempotency_key"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    workspace: Mapped[str] = mapped_column(String(128), index=True)
    gallery_id: Mapped[str] = mapped_column(ForeignKey("galleries.id"), index=True)
    idempotency_key: Mapped[str] = mapped_column(String(128))
    request_hash: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(24), default="uploading")
    revision: Mapped[int] = mapped_column(Integer, default=0)
    job_id: Mapped[str | None] = mapped_column(ForeignKey("jobs.id"), nullable=True)
    version_id: Mapped[str | None] = mapped_column(ForeignKey("gallery_versions.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)


class ImportItem(Base):
    __tablename__ = "import_items"
    __table_args__ = (UniqueConstraint("session_id", "image_id"),)
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("import_sessions.id"), index=True)
    ordinal: Mapped[int] = mapped_column(Integer)
    image_id: Mapped[str] = mapped_column(String(256))
    requested_bbox: Mapped[list[int]] = mapped_column(JSON)
    received_filename: Mapped[str | None] = mapped_column(String(256), nullable=True)
    image_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    image_key: Mapped[str | None] = mapped_column(String(256), nullable=True)
    bbox: Mapped[list[int] | None] = mapped_column(JSON, nullable=True)
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)
