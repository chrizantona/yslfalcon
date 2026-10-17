"""Machine search results, operator reviews, and saved exports."""

from datetime import datetime
from typing import Any

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from ..db import Base, now


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


class Export(Base):
    __tablename__ = "exports"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    workspace: Mapped[str] = mapped_column(String(128), index=True)
    search_id: Mapped[str] = mapped_column(ForeignKey("searches.id"))
    format: Mapped[str] = mapped_column(String(8))
    storage_key: Mapped[str] = mapped_column(String(256))
    sha256: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=now)
