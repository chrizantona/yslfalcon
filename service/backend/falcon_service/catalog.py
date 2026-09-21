"""Catalog commands: prepare immutable snapshots and publish by atomic pointer."""

import hashlib
import json
from typing import TypeVar
from uuid import uuid4

from sqlalchemy import select, update
from sqlalchemy.orm import Session

from .db import Database
from .errors import ServiceError
from .models import Gallery, GalleryVersion, Job
from .runtime import EncoderRuntime

Record = TypeVar("Record")


def scoped(session: Session, model: type[Record], identifier: str, workspace: str) -> Record:
    row = session.scalar(select(model).where(model.id == identifier, model.workspace == workspace))
    if row is None:
        raise ServiceError("not_found", "Объект не найден", 404)
    return row


def prepare_import(
    db: Database, runtime: EncoderRuntime, gallery_id: str, workspace: str, files: list[dict]
) -> Job:
    encoder = runtime.require()
    with db.transaction() as session:
        gallery = scoped(session, Gallery, gallery_id, workspace)
        # UPDATE obtains the row lock on PostgreSQL and SQLite; increment and snapshot
        # creation belong to the same transaction, preventing duplicate numbers.
        number = (
            session.execute(
                update(Gallery)
                .where(Gallery.id == gallery.id)
                .values(next_version=Gallery.next_version + 1)
                .returning(Gallery.next_version)
            ).scalar_one()
            - 1
        )
        version = GalleryVersion(
            id=str(uuid4()),
            workspace=workspace,
            gallery_id=gallery.id,
            number=number,
            model_id=encoder.model_id,
            status="preparing",
            count=0,
        )
        session.add(version)
        session.flush()
        job = Job(
            id=str(uuid4()),
            workspace=workspace,
            kind="import",
            payload={
                "gallery_id": gallery.id,
                "version_id": version.id,
                "model_id": encoder.model_id,
                "files": files,
            },
        )
        session.add(job)
        session.flush()
        return job


def activate(
    db: Database,
    runtime: EncoderRuntime,
    workspace: str,
    gallery_id: str,
    version_id: str,
    expected: str | None = None,
    compare: bool = False,
) -> tuple[Gallery, int]:
    encoder = runtime.require()
    with db.transaction() as session:
        gallery = scoped(session, Gallery, gallery_id, workspace)
        version = scoped(session, GalleryVersion, version_id, workspace)
        if version.gallery_id != gallery.id:
            raise ServiceError("not_found", "Версия не принадлежит галерее", 404)
        if version.status != "ready" or version.count == 0:
            raise ServiceError("gallery_not_ready", "Версия ещё не готова к поиску", 409)
        if version.model_id != encoder.model_id:
            raise ServiceError("model_mismatch", "Версия галереи рассчитана другой моделью", 409)
        statement = update(Gallery).where(Gallery.id == gallery.id, Gallery.workspace == workspace)
        statement = statement.where(
            Gallery.active_version_id == (expected if compare else gallery.active_version_id)
        )
        result = session.execute(statement.values(active_version_id=version.id))
        if result.rowcount != 1:
            raise ServiceError("activation_conflict", "Активная версия уже изменилась; обновите список", 409)
        session.refresh(gallery)
        return gallery, version.count


def fingerprint(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
