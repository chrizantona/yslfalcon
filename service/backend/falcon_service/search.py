"""Version-pinned, independent search with durable idempotency."""

import hashlib
import time
from datetime import timedelta
from uuid import uuid4

from sqlalchemy import or_, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .catalog import fingerprint, scoped
from .config import Settings
from .db import Database, now, timestamp
from .errors import ServiceError
from .leases import heartbeat
from .models import Gallery, GalleryVersion, Observation, Review, Search
from .presenters import review_view
from .runtime import EncoderRuntime
from .storage import ArtifactStore


def search_view(session: Session, row: Search) -> dict:
    if row.state != "completed" or row.result is None:
        raise ServiceError("search_not_complete", "Поиск ещё не завершён", 409)
    result = dict(row.result)
    result["reviews"] = [
        review_view(item)
        for item in session.scalars(
            select(Review)
            .where(Review.search_id == row.id, Review.workspace == row.workspace)
            .order_by(Review.created_at, Review.id)
        )
    ]
    return result


def _existing(session: Session, workspace: str, key: str) -> Search | None:
    return session.scalar(select(Search).where(Search.workspace == workspace, Search.idempotency_key == key))


def reserve_search(
    db: Database,
    store: ArtifactStore,
    runtime: EncoderRuntime,
    workspace: str,
    gallery_id: str,
    image: bytes,
    bbox: list[int],
    width: int,
    height: int,
    top_n: int,
    key: str | None = None,
    pinned: Search | None = None,
    bbox_source: str = "explicit",
) -> str:
    if key is not None and (not key.strip() or len(key) > 128):
        raise ServiceError("invalid_idempotency_key", "Idempotency-Key должен содержать 1–128 символов")
    key = key or str(uuid4())
    request_hash = fingerprint(
        {
            "image": hashlib.sha256(image).hexdigest(),
            "bbox": bbox,
            "gallery_id": gallery_id,
            "top_n": top_n,
            "replay_of": pinned.id if pinned else None,
        }
    )
    with db.transaction() as session:
        existing = _existing(session, workspace, key)
        if existing:
            if existing.request_hash != request_hash:
                raise ServiceError(
                    "idempotency_conflict", "Этот ключ уже использован для другого запроса", 409
                )
            return existing.id
        encoder = runtime.require()
        gallery = scoped(session, Gallery, gallery_id, workspace)
        version_id = pinned.gallery_version_id if pinned else gallery.active_version_id
        if not version_id:
            raise ServiceError("gallery_not_ready", "Сначала активируйте готовую версию галереи", 409)
        version = scoped(session, GalleryVersion, version_id, workspace)
        if version.status != "ready" or version.count == 0:
            raise ServiceError("gallery_not_ready", "Галерея не готова или пуста", 409)
        if version.model_id != encoder.model_id:
            raise ServiceError("model_unavailable", "Модель сохранённой версии сейчас недоступна", 409)
        image_key = store.put(image, "queries", "img")
        query = {
            "image_key": image_key,
            "bbox": bbox,
            "width": width,
            "height": height,
            "bbox_source": bbox_source,
            "sha256": hashlib.sha256(image).hexdigest(),
            "calibration_policy": pinned.query.get("calibration_policy") if pinned else runtime.policy,
        }
        row = Search(
            id=str(uuid4()),
            workspace=workspace,
            gallery_id=gallery_id,
            gallery_version_id=version.id,
            model_id=encoder.model_id,
            idempotency_key=key,
            request_hash=request_hash,
            query=query,
            top_n=top_n,
            replay_of=pinned.id if pinned else None,
        )
        session.add(row)
        try:
            session.flush()
        except IntegrityError:
            # Another process reserved this workspace/key while decoding this upload.
            session.rollback()
            with db.transaction() as concurrent:
                winner = _existing(concurrent, workspace, key)
                if winner is None or winner.request_hash != request_hash:
                    raise ServiceError("idempotency_conflict", "Ключ уже занят другим запросом", 409)
                return winner.id
        return row.id


def execute_search(
    db: Database,
    store: ArtifactStore,
    runtime: EncoderRuntime,
    settings: Settings,
    workspace: str,
    search_id: str,
) -> dict:
    from .inference import rank_gallery

    token = str(uuid4())
    with db.transaction() as session:
        row = scoped(session, Search, search_id, workspace)
        if row.state == "completed":
            return search_view(session, row)
        changed = session.execute(
            update(Search)
            .where(
                Search.id == row.id,
                Search.state != "completed",
                or_(Search.lease_until.is_(None), Search.lease_until < now()),
            )
            .values(
                state="running",
                claim_token=token,
                lease_until=now() + timedelta(seconds=settings.job_lease_seconds),
            )
        )
        if changed.rowcount != 1:
            raise ServiceError(
                "request_in_progress", "Этот запрос уже обрабатывается; повторите его с тем же ключом", 409
            )
        session.refresh(row)
    started = time.perf_counter()
    try:
        with heartbeat(db, Search, search_id, token, settings.job_lease_seconds):
            encoder = runtime.require()
            if encoder.model_id != row.model_id:
                raise ServiceError("model_unavailable", "Модель сохранённого поиска недоступна", 409)
            with db.transaction() as session:
                version = scoped(session, GalleryVersion, row.gallery_version_id, workspace)
                observations = list(
                    session.scalars(
                        select(Observation)
                        .where(
                            Observation.version_id == version.id,
                            Observation.workspace == workspace,
                        )
                        .order_by(Observation.ordinal)
                    )
                )
            if not observations or len(observations) != version.count:
                raise ServiceError("gallery_incomplete", "Снимок галереи неполон; поиск остановлен", 503)
            image = store.read(row.query["image_key"])
            features = [store.read_feature(item.feature_key) for item in observations]
            loaded = time.perf_counter()
            with runtime.lock:
                query = encoder.encode(image, tuple(row.query["bbox"]))
                encoded = time.perf_counter()
                ranked = rank_gallery(
                    query, features, encoder, top_n=row.top_n, policy=row.query.get("calibration_policy")
                )
            finished = time.perf_counter()
            candidates = []
            for rank, item in enumerate(ranked["ranked"], start=1):
                observation = observations[item["index"]]
                candidates.append(
                    {
                        "observation_id": observation.id,
                        "rank": rank,
                        "cosine": float(item["cosine"]),
                        "score": float(item["score"]),
                        "accepted": bool(item["accepted"]),
                        "image_url": f"/api/v1/observations/{observation.id}/image",
                        "crop_url": f"/api/v1/observations/{observation.id}/crop",
                        "bbox": observation.bbox,
                        "external_id": observation.external_id,
                        "bbox_source": observation.bbox_source,
                    }
                )
            result = {
                "id": row.id,
                "decision": ranked["decision"],
                "query": {
                    key: row.query[key] for key in ("bbox", "width", "height", "bbox_source", "sha256")
                },
                "gallery_id": row.gallery_id,
                "gallery_version_id": row.gallery_version_id,
                "gallery_manifest_hash": version.manifest_hash,
                "model_id": row.model_id,
                "query_confidence": ranked["query_confidence"],
                "threshold": ranked["threshold"],
                "confidence_kind": ranked["confidence_kind"],
                "calibration_reason": ranked["calibration_reason"],
                "calibration_policy": row.query.get("calibration_policy"),
                "ranked_candidates": candidates,
                "accepted_candidates": [item for item in candidates if item["accepted"]],
                "timings": {
                    "load_ms": (loaded - started) * 1000,
                    "encode_ms": (encoded - loaded) * 1000,
                    "rank_ms": (finished - encoded) * 1000,
                    "total_ms": (finished - started) * 1000,
                },
                "created_at": timestamp(row.created_at),
                "replay_of": row.replay_of,
            }
            result["query"].update(
                {
                    "image_url": f"/api/v1/searches/{row.id}/image",
                    "crop_url": f"/api/v1/searches/{row.id}/crop",
                }
            )
            with db.transaction() as session:
                changed = session.execute(
                    update(Search)
                    .where(Search.id == row.id, Search.claim_token == token)
                    .values(result=result, state="completed", error=None, lease_until=None, claim_token=None)
                )
                if changed.rowcount != 1:
                    raise ServiceError(
                        "search_lease_lost", "Запрос восстановлен другим обработчиком; повторите чтение", 409
                    )
                return search_view(session, scoped(session, Search, row.id, workspace))
    except Exception:
        with db.transaction() as session:
            session.execute(
                update(Search)
                .where(Search.id == search_id, Search.claim_token == token)
                .values(state="failed", error="search_failed", lease_until=None, claim_token=None)
            )
        raise
