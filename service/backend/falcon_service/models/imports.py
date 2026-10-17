"""Resumable gallery import sessions and their uploaded images."""

from datetime import datetime

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from ..db import Base, now


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
