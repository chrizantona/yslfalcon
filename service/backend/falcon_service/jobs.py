"""Durable leased work queue with fencing, retries and idempotent publication."""

import hashlib
import logging
import tempfile
import time
from datetime import timedelta
from uuid import uuid4

from sqlalchemy import and_, or_, select, update
from sqlalchemy.orm import Session

from .catalog import fingerprint, scoped
from .config import Settings
from .db import Database, now
from .errors import ServiceError
from .exports import (
    CASE_SPOOL_BYTES,
    MAX_CASE_ARCHIVE_BYTES,
    make_passport,
    render_case,
    render_export,
)
from .features import FeatureStore
from .indexes import prepare_index_artifact
from .leases import heartbeat
from .models import Export, GalleryVersion, Job, Observation, ObservationDigest
from .runtime import EncoderRuntime
from .storage import ArtifactStore

logger = logging.getLogger(__name__)


def enqueue(db: Database, workspace: str, kind: str, payload: dict) -> Job:
    with db.transaction() as session:
        job = Job(id=str(uuid4()), workspace=workspace, kind=kind, payload=payload)
        session.add(job)
        session.flush()
        return job


def claim(db: Database, settings: Settings) -> Job | None:
    instant = now()
    eligible = or_(
        and_(Job.status == "pending", Job.available_at <= instant),
        and_(Job.status == "running", Job.lease_until < instant),
    )
    with db.transaction() as session:
        # Workers killed on their last attempt must eventually become terminal.
        exhausted = list(
            session.scalars(select(Job).where(eligible, Job.attempts >= settings.job_max_attempts))
        )
        for item in exhausted:
            changed = session.execute(
                update(Job)
                .where(Job.id == item.id, eligible)
                .values(
                    status="failed",
                    error="worker_lease_expired",
                    message="Обработка прервана; попытки исчерпаны",
                    completed_at=instant,
                    claim_token=None,
                    lease_until=None,
                )
            )
            if changed.rowcount and item.kind == "import":
                session.execute(
                    update(GalleryVersion)
                    .where(
                        GalleryVersion.id == item.payload["version_id"], GalleryVersion.status == "preparing"
                    )
                    .values(status="failed")
                )
        candidates = list(
            session.scalars(
                select(Job.id)
                .where(eligible, Job.attempts < settings.job_max_attempts)
                .order_by(Job.created_at)
                .limit(10)
            )
        )
        for identifier in candidates:
            token = str(uuid4())
            changed = session.execute(
                update(Job)
                .where(Job.id == identifier, eligible, Job.attempts < settings.job_max_attempts)
                .values(
                    status="running",
                    claim_token=token,
                    attempts=Job.attempts + 1,
                    error=None,
                    lease_until=instant + timedelta(seconds=settings.job_lease_seconds),
                    message="Обработка",
                )
            )
            if changed.rowcount == 1:
                session.expire_all()
                return session.get(Job, identifier)
    return None


def _owned_job(session: Session, job: Job) -> Job:
    # A conditional write is a fence AND a transaction-held write lock on both
    # PostgreSQL and SQLite (SQLite ignores SELECT ... FOR UPDATE).
    changed = session.execute(
        update(Job)
        .where(
            Job.id == job.id,
            Job.claim_token == job.claim_token,
            Job.status == "running",
        )
        .values(claim_token=job.claim_token)
    )
    if changed.rowcount != 1:
        raise ServiceError("job_lease_lost", "Задание передано другому обработчику", 409)
    return session.get(Job, job.id)


def _progress(db: Database, job: Job, progress: int, message: str) -> None:
    with db.transaction() as session:
        changed = session.execute(
            update(Job)
            .where(Job.id == job.id, Job.claim_token == job.claim_token)
            .values(progress=progress, message=message)
        )
        if changed.rowcount != 1:
            raise ServiceError("job_lease_lost", "Задание передано другому обработчику", 409)


def import_snapshot(db: Database, store: ArtifactStore, runtime: EncoderRuntime, job: Job) -> dict:
    encoder = runtime.require()
    if encoder.model_id != job.payload["model_id"]:
        raise ServiceError("model_mismatch", "Модель изменилась после начала импорта", 409)
    with db.transaction() as session:
        version = scoped(session, GalleryVersion, job.payload["version_id"], job.workspace)
        if version.status == "ready":
            return {
                "gallery_id": version.gallery_id,
                "version_id": version.id,
                "count": version.count,
                "status": "ready",
                "manifest_hash": version.manifest_hash,
            }
    observations, digests, artifacts, features, hashes = [], [], [], [], []
    files = job.payload["files"]
    for ordinal, item in enumerate(files):
        image = store.read(item["image_key"])
        with runtime.lock:
            feature = encoder.encode(image, tuple(item["bbox"]))
        artifact = FeatureStore.prepare(feature, job.workspace)
        if artifact.model_id != encoder.model_id or artifact.dimension != encoder.dimension:
            raise ServiceError("feature_model_mismatch", "Признаки не соответствуют версии модели", 503)
        artifacts.append(artifact)
        features.append(feature)
        observation_id = str(uuid4())
        image_sha256 = hashlib.sha256(image).hexdigest()
        observations.append(
            Observation(
                id=observation_id,
                workspace=job.workspace,
                version_id=version.id,
                ordinal=ordinal,
                external_id=item.get("external_id"),
                image_key=item["image_key"],
                feature_key=artifact.key,
                bbox=item["bbox"],
                width=item["width"],
                height=item["height"],
                bbox_source=item["bbox_source"],
            )
        )
        digests.append(
            ObservationDigest(
                observation_id=observation_id,
                workspace=job.workspace,
                sha256=image_sha256,
            )
        )
        hashes.append({"image_sha256": image_sha256, "bbox": item["bbox"]})
        _progress(
            db,
            job,
            min(95, int((ordinal + 1) / len(files) * 95)),
            f"Рассчитано {ordinal + 1} из {len(files)}",
        )
    manifest_hash = fingerprint({"model_id": encoder.model_id, "observations": hashes})
    index_artifact = prepare_index_artifact(store, job.workspace, version.id, encoder.model_id, features)
    try:
        with db.transaction() as session:
            _owned_job(session, job)
            version = scoped(session, GalleryVersion, version.id, job.workspace)
            if version.status != "preparing":
                raise ServiceError("invalid_version_state", "Состояние версии изменилось", 409)
            session.add_all(artifacts)
            session.add_all(observations)
            session.add_all(digests)
            session.add(index_artifact)
            version.status, version.count, version.manifest_hash = (
                "ready",
                len(observations),
                manifest_hash,
            )
    except Exception:
        try:
            store.path(index_artifact.artifact_key).unlink(missing_ok=True)
        except (OSError, ServiceError):
            logger.warning(
                "index_artifact_cleanup_failed",
                extra={"artifact_key": index_artifact.artifact_key},
                exc_info=True,
            )
        raise
    return {
        "gallery_id": version.gallery_id,
        "version_id": version.id,
        "count": version.count,
        "status": "ready",
        "manifest_hash": manifest_hash,
    }


def export_result(db: Database, store: ArtifactStore, job: Job) -> dict:
    export_id = job.payload["export_id"]
    with db.transaction() as session:
        existing = session.get(Export, export_id)
        if existing:
            return {
                "export_id": existing.id,
                "download_url": f"/api/v1/exports/{existing.id}/download",
                "sha256": existing.sha256,
            }
    passport = make_passport(db, store, job.workspace, job.payload["search_id"])
    if job.payload["format"] == "case":
        with tempfile.SpooledTemporaryFile(max_size=CASE_SPOOL_BYTES, mode="w+b") as archive:
            render_case(passport, db, store, job.workspace, archive)
            if archive.tell() > MAX_CASE_ARCHIVE_BYTES:
                raise ServiceError("case_too_large", "Размер offline-дела превышает 256 МиБ", 413)
            archive.seek(0)
            storage_key, checksum, _ = store.put_stream(
                archive,
                "exports",
                "zip",
                max_bytes=MAX_CASE_ARCHIVE_BYTES,
            )
    else:
        data = render_export(passport, job.payload["format"], db, store, job.workspace)
        storage_key = store.put(data, "exports", job.payload["format"])
        checksum = hashlib.sha256(data).hexdigest()
    with db.transaction() as session:
        _owned_job(session, job)
        session.add(
            Export(
                id=export_id,
                workspace=job.workspace,
                search_id=job.payload["search_id"],
                format=job.payload["format"],
                storage_key=storage_key,
                sha256=checksum,
            )
        )
    return {
        "export_id": export_id,
        "download_url": f"/api/v1/exports/{export_id}/download",
        "sha256": checksum,
    }


def run_once(db: Database, store: ArtifactStore, runtime: EncoderRuntime, settings: Settings) -> bool:
    job = claim(db, settings)
    if job is None:
        return False
    started = time.perf_counter()
    log_context = {"component": "worker", "job_id": job.id, "job_kind": job.kind, "attempt": job.attempts}
    logger.info("job_claimed", extra=log_context)
    try:
        with heartbeat(db, Job, job.id, job.claim_token, settings.job_lease_seconds):
            if job.kind == "import":
                result = import_snapshot(db, store, runtime, job)
            elif job.kind == "export":
                result = export_result(db, store, job)
            else:
                raise ServiceError("unknown_job_kind", "Неизвестный тип задания")
            with db.transaction() as session:
                current = _owned_job(session, job)
                current.status, current.progress, current.result = "completed", 100, result
                current.completed_at, current.lease_until, current.claim_token = now(), None, None
                current.message = "Готово"
            logger.info(
                "job_completed",
                extra={**log_context, "duration_ms": round((time.perf_counter() - started) * 1000, 3)},
            )
    except Exception as exc:
        logger.exception(
            "job_failed",
            extra={**log_context, "duration_ms": round((time.perf_counter() - started) * 1000, 3)},
        )
        with db.transaction() as session:
            changed = session.execute(
                update(Job)
                .where(
                    Job.id == job.id,
                    Job.claim_token == job.claim_token,
                    Job.status == "running",
                )
                .values(claim_token=job.claim_token)
            )
            if changed.rowcount == 1:
                current = session.get(Job, job.id)
                permanent = isinstance(exc, (ValueError, ServiceError)) and not (
                    isinstance(exc, ServiceError) and exc.status >= 500
                )
                terminal = permanent or current.attempts >= settings.job_max_attempts
                current.status = "failed" if terminal else "pending"
                current.error = exc.code if isinstance(exc, ServiceError) else "processing_failed"
                current.message = "Ошибка обработки" if terminal else "Повторная попытка запланирована"
                current.available_at = now() + timedelta(seconds=min(60, 2**current.attempts))
                current.completed_at = now() if terminal else None
                current.lease_until, current.claim_token = None, None
                if terminal and current.kind == "import":
                    session.execute(
                        update(GalleryVersion)
                        .where(
                            GalleryVersion.id == current.payload["version_id"],
                            GalleryVersion.status == "preparing",
                        )
                        .values(status="failed")
                    )
        if changed.rowcount == 1 and not terminal:
            logger.warning(
                "job_retry_scheduled",
                extra={**log_context, "retry_in_seconds": min(60, 2**current.attempts)},
            )
    return True
