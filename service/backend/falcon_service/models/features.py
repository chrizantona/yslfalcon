"""Authoritative feature vectors and rebuildable gallery indexes."""

from datetime import datetime
from typing import Any

from sqlalchemy import JSON, DateTime, ForeignKey, Integer, LargeBinary, String
from sqlalchemy.orm import Mapped, mapped_column

from ..db import Base, now


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
