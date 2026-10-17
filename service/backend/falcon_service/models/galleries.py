"""Gallery snapshots and their immutable observations."""

from datetime import datetime

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from ..db import Base, now


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
