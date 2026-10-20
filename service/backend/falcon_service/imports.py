"""Resumable CSV imports: validate chunks, then publish exactly one snapshot job."""

import csv
import hashlib
import io
from decimal import Decimal, InvalidOperation
from pathlib import PurePosixPath
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from .catalog import create_import_job, fingerprint, scoped
from .config import Settings
from .db import Database, timestamp
from .errors import ServiceError, image_validation_message
from .inference import validate_image
from .models import Gallery, ImportItem, ImportSession, Job
from .runtime import EncoderRuntime
from .storage import ArtifactStore

MAX_SESSION_FILES = 10_000
MAX_CHUNK_FILES = 100
SUPPORTED_SUFFIXES = (".jpg", ".jpeg", ".png")


class CreateImportSession(BaseModel):
    model_config = ConfigDict(extra="forbid")
    manifest_csv: str = Field(min_length=1, max_length=2 * 1024 * 1024)


def limits(settings: Settings) -> dict:
    return {
        "max_chunk_files": min(MAX_CHUNK_FILES, settings.max_import_files),
        "max_chunk_bytes": settings.max_request_bytes,
        "max_upload_bytes": settings.max_upload_bytes,
    }


def _filename(value: str) -> bool:
    return (
        bool(value)
        and len(value) <= 255
        and value not in {".", ".."}
        and not any(ord(char) < 32 or char in "/\\" for char in value)
    )


def _aliases(identity: str) -> list[str]:
    suffix = PurePosixPath(identity).suffix
    if suffix:
        if suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ServiceError("invalid_manifest", f"image_id {identity}: поддерживаются JPEG и PNG")
        return [identity[: -len(suffix)] + suffix.lower()]
    return [identity + extension for extension in SUPPORTED_SUFFIXES]


def parse_manifest(value: str) -> list[dict]:
    try:
        reader = csv.DictReader(io.StringIO(value.lstrip("\ufeff")), strict=True)
        columns = reader.fieldnames or []
        if len(set(columns)) != len(columns) or not {"image_id", "x", "y", "w", "h"} <= set(columns):
            raise ServiceError("invalid_manifest", "CSV должен содержать колонки image_id,x,y,w,h")
        records, aliases = [], set()
        for ordinal, row in enumerate(reader):
            if ordinal >= MAX_SESSION_FILES:
                raise ServiceError("import_limit", f"Одна сессия принимает до {MAX_SESSION_FILES} строк", 413)
            identity = (row.get("image_id") or "").strip()
            if not _filename(identity) or None in row or any(value is None for value in row.values()):
                raise ServiceError("invalid_manifest", f"Некорректная строка CSV: {ordinal + 2}")
            names = _aliases(identity)
            if aliases.intersection(names):
                raise ServiceError(
                    "invalid_manifest", f"Повторяющийся или неоднозначный image_id: {identity}"
                )
            aliases.update(names)
            try:
                box = [Decimal(row[key]) for key in ("x", "y", "w", "h")]
                if any(
                    not item.is_finite() or item.copy_abs() > 10_000_000 or item != int(item) for item in box
                ):
                    raise ValueError("Invalid integer coordinates")
                bbox = [int(item) for item in box]
                if bbox[2] <= 0 or bbox[3] <= 0:
                    raise ValueError("Invalid bounding-box dimensions")
            except (InvalidOperation, ValueError, TypeError, OverflowError) as exc:
                raise ServiceError("invalid_manifest", f"Некорректная рамка для image_id {identity}") from exc
            records.append({"image_id": identity, "bbox": bbox})
        if not records:
            raise ServiceError("invalid_manifest", "CSV не содержит изображений")
        return records
    except csv.Error as exc:
        raise ServiceError("invalid_manifest", "Не удалось разобрать CSV") from exc


def import_session(session: Session, identifier: str, gallery_id: str, workspace: str) -> ImportSession:
    row = scoped(session, ImportSession, identifier, workspace)
    if row.gallery_id != gallery_id:
        raise ServiceError("not_found", "Сессия не найдена в этой галерее", 404)
    return row


def _items(session: Session, identifier: str) -> list[ImportItem]:
    return list(
        session.scalars(
            select(ImportItem).where(ImportItem.session_id == identifier).order_by(ImportItem.ordinal)
        )
    )


def session_view(session: Session, row: ImportSession, settings: Settings) -> dict:
    items = _items(session, row.id)
    missing = [item.image_id for item in items if item.image_key is None]
    received = [item.image_id for item in items if item.image_key is not None]
    return {
        "id": row.id,
        "gallery_id": row.gallery_id,
        "status": row.status,
        "expected_count": len(items),
        "uploaded_count": len(received),
        "missing_image_ids": missing,
        "received_image_ids": received,
        "validation": {
            "ready": not missing,
            "issues": [
                {
                    "code": "missing_files",
                    "message": f"Не загружено {len(missing)} изображений",
                    "image_id": None,
                }
            ]
            if missing
            else [],
        },
        "job_id": row.job_id,
        "version_id": row.version_id,
        "created_at": timestamp(row.created_at),
        "limits": limits(settings),
    }


def create_session(
    db: Database, settings: Settings, gallery_id: str, workspace: str, key: str, manifest_csv: str
) -> dict:
    if not key.strip() or len(key) > 128:
        raise ServiceError("invalid_idempotency_key", "Idempotency-Key должен содержать 1–128 символов")
    records = parse_manifest(manifest_csv)
    request_hash = fingerprint({"gallery_id": gallery_id, "records": records})

    def existing(session: Session) -> ImportSession | None:
        return session.scalar(
            select(ImportSession).where(
                ImportSession.workspace == workspace, ImportSession.idempotency_key == key
            )
        )

    def match(row: ImportSession) -> None:
        if row.request_hash != request_hash:
            raise ServiceError("idempotency_conflict", "Ключ уже использован для другого импорта", 409)

    try:
        with db.transaction() as session:
            scoped(session, Gallery, gallery_id, workspace)
            row = existing(session)
            if row:
                match(row)
                return session_view(session, row, settings)
            row = ImportSession(
                id=str(uuid4()),
                workspace=workspace,
                gallery_id=gallery_id,
                idempotency_key=key,
                request_hash=request_hash,
            )
            session.add(row)
            session.flush()
            session.add_all(
                ImportItem(
                    id=str(uuid4()),
                    session_id=row.id,
                    ordinal=ordinal,
                    image_id=item["image_id"],
                    requested_bbox=item["bbox"],
                )
                for ordinal, item in enumerate(records)
            )
            session.flush()
            return session_view(session, row, settings)
    except IntegrityError:
        with db.transaction() as session:
            row = existing(session)
            if row is None:
                raise
            match(row)
            return session_view(session, row, settings)


def _lock(session: Session, identifier: str, gallery_id: str, workspace: str) -> ImportSession:
    # A write fence serializes chunks and finalization on PostgreSQL and SQLite.
    changed = session.execute(
        update(ImportSession)
        .where(
            ImportSession.id == identifier,
            ImportSession.gallery_id == gallery_id,
            ImportSession.workspace == workspace,
        )
        .values(revision=ImportSession.revision + 1)
    )
    if changed.rowcount != 1:
        raise ServiceError("not_found", "Сессия не найдена", 404)
    return import_session(session, identifier, gallery_id, workspace)


def upload_chunk(
    db: Database,
    store: ArtifactStore,
    settings: Settings,
    gallery_id: str,
    workspace: str,
    identifier: str,
    uploaded: list[tuple[str, bytes]],
) -> dict:
    with db.transaction() as session:
        row = import_session(session, identifier, gallery_id, workspace)
        items = _items(session, row.id)
    lookup = {alias: item for item in items for alias in _aliases(item.image_id)}
    validated, seen = [], set()
    for filename, data in uploaded:
        if not _filename(filename):
            raise ServiceError("invalid_filenames", "Нужны обычные имена файлов без пути")
        names = _aliases(filename)
        item = lookup.get(names[0]) if len(names) == 1 else None
        if item is None:
            raise ServiceError("unexpected_file", f"Изображение {filename} отсутствует в CSV")
        if item.id in seen:
            raise ServiceError("ambiguous_file", f"Несколько файлов соответствуют image_id {item.image_id}")
        seen.add(item.id)
        try:
            original, bbox, width, height = validate_image(data, item.requested_bbox)
        except ValueError as exc:
            raise ServiceError("invalid_image", f"Файл {filename}: {image_validation_message(exc)}") from exc
        validated.append(
            {
                "id": item.id,
                "filename": filename,
                "sha256": hashlib.sha256(original).hexdigest(),
                "data": original,
                "bbox": list(bbox),
                "width": width,
                "height": height,
            }
        )
    owned_keys, committed = [], False
    try:
        with db.transaction() as session:
            row = _lock(session, identifier, gallery_id, workspace)
            current = {item.id: item for item in _items(session, row.id)}
            for value in validated:
                item = current[value["id"]]
                if item.image_key is not None:
                    if item.image_sha256 != value["sha256"] or item.received_filename != value["filename"]:
                        raise ServiceError(
                            "upload_conflict", f"Для image_id {item.image_id} уже загружен другой файл", 409
                        )
                    continue
                if row.status != "uploading":
                    raise ServiceError(
                        "import_finalized", "Импорт уже завершён и недоступен для изменения", 409
                    )
                key = store.put(value["data"], "observations", "img")
                owned_keys.append(key)
                item.image_key, item.image_sha256 = key, value["sha256"]
                item.received_filename = value["filename"]
                item.bbox, item.width, item.height = value["bbox"], value["width"], value["height"]
            session.flush()
            result = session_view(session, row, settings)
        committed = True
        return result
    finally:
        if not committed:
            for key in owned_keys:
                store.path(key).unlink(missing_ok=True)


def finalize_session(
    db: Database, runtime: EncoderRuntime, gallery_id: str, workspace: str, identifier: str
) -> Job:
    with db.transaction() as session:
        row = _lock(session, identifier, gallery_id, workspace)
        if row.job_id is not None:
            return scoped(session, Job, row.job_id, workspace)
        items = _items(session, row.id)
        missing = sum(item.image_key is None for item in items)
        if missing:
            raise ServiceError(
                "import_incomplete", f"Не загружено {missing} изображений; продолжите загрузку", 409
            )
        model_id = runtime.require().model_id
        job = create_import_job(
            session,
            model_id,
            gallery_id,
            workspace,
            [
                {
                    "image_key": item.image_key,
                    "bbox": item.bbox,
                    "width": item.width,
                    "height": item.height,
                    "external_id": item.image_id,
                    "bbox_source": "explicit",
                }
                for item in items
            ],
        )
        row.status, row.job_id, row.version_id = "finalized", job.id, job.payload["version_id"]
        return job
