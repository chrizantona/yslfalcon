"""Immutable FAISS HNSW indexes derived from versioned gallery features."""

import hashlib
import logging
import threading
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as postgres_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from .db import Database
from .errors import ServiceError
from .features import FeatureStore
from .inference.core import normalise
from .models import GalleryIndexArtifact, Observation
from .storage import ArtifactStore

INDEX_ALGORITHM = "faiss-hnsw-flat-ip-v1"
INDEX_METRIC = "inner_product"
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 128
HNSW_SHORTLIST_MIN = 100
HNSW_SHORTLIST_MULTIPLIER = 4
logger = logging.getLogger(__name__)


def _faiss():
    try:
        import faiss
    except ImportError as exc:  # pragma: no cover - startup dependency failure
        raise RuntimeError("faiss-cpu is required for gallery HNSW indexes") from exc
    return faiss


def _parameters() -> dict[str, int]:
    return {
        "m": HNSW_M,
        "ef_construction": HNSW_EF_CONSTRUCTION,
        "ef_search": HNSW_EF_SEARCH,
        "shortlist_min": HNSW_SHORTLIST_MIN,
        "shortlist_multiplier": HNSW_SHORTLIST_MULTIPLIER,
    }


def prepare_index_artifact(
    store: ArtifactStore,
    workspace: str,
    version_id: str,
    model_id: str,
    features,
) -> GalleryIndexArtifact:
    """Build one immutable HNSW graph; callers publish its metadata transactionally."""

    if not features or any(item.model_id != model_id for item in features):
        raise ValueError("Index features are empty or use a different model")
    matrix = np.stack([normalise(item.embedding) for item in features])
    faiss = _faiss()
    index = faiss.IndexHNSWFlat(matrix.shape[1], HNSW_M, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
    index.add(np.ascontiguousarray(matrix, dtype=np.float32))
    payload = faiss.serialize_index(index).tobytes()
    checksum = hashlib.sha256(payload).hexdigest()
    artifact_key = store.put(payload, "indexes", "faiss")
    return GalleryIndexArtifact(
        version_id=version_id,
        workspace=workspace,
        model_id=model_id,
        algorithm=INDEX_ALGORITHM,
        metric=INDEX_METRIC,
        dimension=matrix.shape[1],
        count=len(matrix),
        parameters=_parameters(),
        artifact_key=artifact_key,
        sha256=checksum,
        size_bytes=len(payload),
    )


@dataclass(frozen=True)
class LoadedGalleryIndex:
    version_id: str
    model_id: str
    count: int
    dimension: int
    index: object

    def retrieve(self, embedding, limit: int, *, exact_population: bool):
        query = normalise(embedding)[None]
        if query.shape[1] != self.dimension:
            raise ServiceError("index_model_mismatch", "Индекс галереи не соответствует модели", 503)
        count = min(
            self.count,
            max(HNSW_SHORTLIST_MIN, limit * HNSW_SHORTLIST_MULTIPLIER),
        )
        if exact_population:
            distances, labels = self.index.storage.search(query, self.count)
            # FAISS is free to order equal distances by label. Carry the complete cutoff tie into
            # the application-owned feature tie-break so CSV/insertion order cannot select rows.
            cutoff = distances[0, count - 1]
            tied = np.flatnonzero(distances[0] == cutoff)
            end = max(count, int(tied[-1]) + 1 if len(tied) else count)
            return labels[0, :end].astype(int).tolist(), distances[0].astype(np.float32)
        faiss = _faiss()
        parameters = faiss.SearchParametersHNSW()
        parameters.efSearch = max(HNSW_EF_SEARCH, count * 2)
        _, labels = self.index.search(query, count, params=parameters)
        positions = [int(item) for item in labels[0] if item >= 0]
        if not positions:
            raise ServiceError("index_empty_result", "Индекс галереи не вернул кандидатов", 503)
        return positions, None


class GalleryIndexStore:
    """Load immutable graphs once per process and rebuild missing legacy derivatives."""

    def __init__(self, db: Database, artifacts: ArtifactStore, cache_size: int = 2):
        self.db = db
        self.artifacts = artifacts
        self.cache_size = cache_size
        self._cache: OrderedDict[str, LoadedGalleryIndex] = OrderedDict()
        self._lock = threading.RLock()

    def get(self, workspace: str, version_id: str, model_id: str, count: int) -> LoadedGalleryIndex:
        with self._lock:
            cached = self._cache.get(version_id)
            if cached is not None:
                if cached.model_id != model_id or cached.count != count:
                    raise ServiceError("index_model_mismatch", "Индекс галереи не соответствует версии", 503)
                self._cache.move_to_end(version_id)
                return cached
            with self.db.transaction() as session:
                record = session.get(GalleryIndexArtifact, version_id)
            if record is None:
                record = self._backfill(workspace, version_id, model_id, count)
            try:
                loaded = self._load(record, workspace, model_id, count)
            except ServiceError:
                record = self._rebuild(record, workspace, version_id, model_id, count)
                loaded = self._load(record, workspace, model_id, count)
            self._cache[version_id] = loaded
            self._cache.move_to_end(version_id)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
            return loaded

    def _canonical_features(self, workspace, version_id, model_id, count):
        with self.db.transaction() as session:
            observations = list(
                session.scalars(
                    select(Observation)
                    .where(
                        Observation.workspace == workspace,
                        Observation.version_id == version_id,
                    )
                    .order_by(Observation.ordinal)
                )
            )
        if len(observations) != count or [item.ordinal for item in observations] != list(range(count)):
            raise ServiceError("gallery_incomplete", "Снимок галереи неполон; индекс не построен", 503)
        return FeatureStore(self.db, self.artifacts).read_many(workspace, observations, model_id)

    def _backfill(self, workspace, version_id, model_id, count):
        features = self._canonical_features(workspace, version_id, model_id, count)
        proposed = prepare_index_artifact(self.artifacts, workspace, version_id, model_id, features)
        values = {
            column.name: getattr(proposed, column.name)
            for column in GalleryIndexArtifact.__table__.columns
            if column.name != "created_at"
        }
        try:
            with self.db.transaction() as session:
                insert = postgres_insert if self.db.engine.dialect.name == "postgresql" else sqlite_insert
                session.execute(
                    insert(GalleryIndexArtifact)
                    .values(**values)
                    .on_conflict_do_nothing(index_elements=[GalleryIndexArtifact.version_id])
                )
                winner = session.get(GalleryIndexArtifact, version_id)
        except Exception:
            self._discard(proposed.artifact_key)
            raise
        if winner is None:
            self._discard(proposed.artifact_key)
            raise ServiceError("index_unavailable", "Индекс галереи недоступен", 503)
        if winner.artifact_key != proposed.artifact_key:
            self._discard(proposed.artifact_key)
        return winner

    def _rebuild(self, previous, workspace, version_id, model_id, count):
        features = self._canonical_features(workspace, version_id, model_id, count)
        proposed = prepare_index_artifact(self.artifacts, workspace, version_id, model_id, features)
        values = {
            column.name: getattr(proposed, column.name)
            for column in GalleryIndexArtifact.__table__.columns
            if column.name not in {"version_id", "created_at"}
        }
        try:
            with self.db.transaction() as session:
                changed = session.execute(
                    update(GalleryIndexArtifact)
                    .where(
                        GalleryIndexArtifact.version_id == version_id,
                        GalleryIndexArtifact.artifact_key == previous.artifact_key,
                        GalleryIndexArtifact.sha256 == previous.sha256,
                    )
                    .values(**values)
                )
                session.expire_all()
                winner = session.get(GalleryIndexArtifact, version_id)
        except Exception:
            self._discard(proposed.artifact_key)
            raise
        if winner is None:
            self._discard(proposed.artifact_key)
            raise ServiceError("index_unavailable", "Индекс галереи недоступен", 503)
        if changed.rowcount != 1:
            self._discard(proposed.artifact_key)
        else:
            self._discard_if_unreferenced(previous.artifact_key)
        return winner

    def _discard_if_unreferenced(self, artifact_key):
        with self.db.transaction() as session:
            referenced = session.scalar(
                select(GalleryIndexArtifact.version_id).where(
                    GalleryIndexArtifact.artifact_key == artifact_key
                )
            )
        if referenced is None:
            self._discard(artifact_key)

    def _discard(self, artifact_key):
        try:
            self.artifacts.path(artifact_key).unlink(missing_ok=True)
        except (OSError, ServiceError):
            logger.warning(
                "index_artifact_cleanup_failed",
                extra={"artifact_key": artifact_key},
                exc_info=True,
            )

    def _load(self, record, workspace, model_id, count):
        if (
            record.workspace != workspace
            or record.model_id != model_id
            or record.count != count
            or record.algorithm != INDEX_ALGORITHM
            or record.metric != INDEX_METRIC
            or record.parameters != _parameters()
        ):
            raise ServiceError("index_model_mismatch", "Индекс галереи не соответствует версии", 503)
        try:
            path = self.artifacts.path(record.artifact_key)
            size = path.stat().st_size
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    digest.update(chunk)
        except (OSError, ServiceError) as exc:
            raise ServiceError("index_unavailable", "Индекс галереи недоступен", 503) from exc
        if size != record.size_bytes or digest.hexdigest() != record.sha256:
            raise ServiceError("index_corrupt", "Индекс галереи повреждён", 503)
        try:
            faiss = _faiss()
            index = faiss.read_index(str(path))
        except Exception as exc:
            raise ServiceError("index_corrupt", "Индекс галереи повреждён", 503) from exc
        if (
            index.ntotal != count
            or index.d != record.dimension
            or index.metric_type != faiss.METRIC_INNER_PRODUCT
            or not hasattr(index, "hnsw")
            or not hasattr(index, "storage")
        ):
            raise ServiceError("index_corrupt", "Структура индекса галереи некорректна", 503)
        return LoadedGalleryIndex(record.version_id, model_id, count, record.dimension, index)
