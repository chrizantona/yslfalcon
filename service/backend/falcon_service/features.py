"""Database-authoritative numeric features with additive legacy-file backfill."""

import hashlib
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as postgres_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from .db import Database
from .errors import ServiceError
from .inference import FeatureBundle
from .inference.wire import pack_features, unpack_features
from .models import FeatureArtifact, Observation
from .storage import ArtifactStore


class FeatureStore:
    """Publish artifacts in the snapshot transaction; files are migration input only."""

    def __init__(self, db: Database, legacy: ArtifactStore):
        self.db, self.legacy = db, legacy

    @staticmethod
    def prepare(feature: FeatureBundle, workspace: str, key: str | None = None) -> FeatureArtifact:
        payload = pack_features([feature])
        decoded, _ = unpack_features(payload)
        return FeatureArtifact(
            key=key or f"features/{uuid4().hex}.npz",
            workspace=workspace,
            model_id=feature.model_id,
            dimension=len(decoded[0].embedding),
            dtype="float32",
            encoding="numeric-npz-v1",
            sha256=hashlib.sha256(payload).hexdigest(),
            payload=payload,
        )

    @staticmethod
    def decode(artifact: FeatureArtifact, expected_model: str) -> FeatureBundle:
        try:
            if (
                artifact.encoding != "numeric-npz-v1"
                or artifact.dtype != "float32"
                or artifact.model_id != expected_model
                or hashlib.sha256(artifact.payload).hexdigest() != artifact.sha256
            ):
                raise ValueError("Feature metadata or checksum mismatch")
            features, extra = unpack_features(artifact.payload)
            if (
                len(features) != 1
                or extra
                or features[0].model_id != artifact.model_id
                or len(features[0].embedding) != artifact.dimension
            ):
                raise ValueError("Feature shape or model mismatch")
            return features[0]
        except (ValueError, OSError, TypeError) as exc:
            raise ServiceError(
                "feature_corrupt", "Сохранённые признаки повреждены; поиск остановлен", 503
            ) from exc

    def read_many(
        self, workspace: str, observations: list[Observation], model_id: str
    ) -> list[FeatureBundle]:
        if any(item.workspace != workspace for item in observations):
            raise ServiceError("not_found", "Объект не найден", 404)
        keys = [item.feature_key for item in observations]
        records = {}
        # Bounded IN clauses also support SQLite installations with 999 SQL variables.
        with self.db.transaction() as session:
            for offset in range(0, len(keys), 500):
                rows = session.scalars(
                    select(FeatureArtifact).where(
                        FeatureArtifact.workspace == workspace,
                        FeatureArtifact.key.in_(keys[offset : offset + 500]),
                    )
                )
                records.update((item.key, item) for item in rows)
        for key in dict.fromkeys(keys):
            if key not in records:
                records[key] = self._backfill(workspace, key, model_id)
        # Never mask corrupted authoritative database bytes by falling back to a file.
        return [self.decode(records[key], model_id) for key in keys]

    def _backfill(self, workspace: str, key: str, model_id: str) -> FeatureArtifact:
        try:
            feature = self.legacy.read_feature(key)
            record = self.prepare(feature, workspace, key)
            self.decode(record, model_id)
        except (ValueError, OSError, KeyError, TypeError, ServiceError) as exc:
            raise ServiceError(
                "feature_unavailable", "Признаки старой версии недоступны для переноса в БД", 503
            ) from exc
        with self.db.transaction() as session:
            insert = postgres_insert if self.db.engine.dialect.name == "postgresql" else sqlite_insert
            session.execute(
                insert(FeatureArtifact)
                .values(
                    key=record.key,
                    workspace=record.workspace,
                    model_id=record.model_id,
                    dimension=record.dimension,
                    dtype=record.dtype,
                    encoding=record.encoding,
                    sha256=record.sha256,
                    payload=record.payload,
                )
                .on_conflict_do_nothing(index_elements=[FeatureArtifact.key])
            )
            winner = session.scalar(
                select(FeatureArtifact).where(
                    FeatureArtifact.key == key, FeatureArtifact.workspace == workspace
                )
            )
            if winner is None:
                raise ServiceError("feature_unavailable", "Признаки версии недоступны", 503)
            return winner
