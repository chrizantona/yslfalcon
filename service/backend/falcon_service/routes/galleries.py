"""Gallery management and validated snapshot imports."""

import json
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, Query, Request, UploadFile
from sqlalchemy import func, select
from starlette.concurrency import run_in_threadpool

from ..auth import Principal, principal, require_role
from ..catalog import activate, prepare_import, scoped
from ..errors import ServiceError, image_validation_message
from ..models import Gallery, GalleryVersion
from ..presenters import gallery_view, job_view, version_view
from ..schemas import ActivateVersion, CreateGallery

router = APIRouter(prefix="/galleries", tags=["Galleries"])


@router.get("")
def galleries(
    request: Request,
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    identity: Principal = Depends(principal),
):
    with request.app.state.db.transaction() as session:
        total = session.scalar(
            select(func.count()).select_from(Gallery).where(Gallery.workspace == identity.workspace)
        )
        rows = session.scalars(
            select(Gallery)
            .where(Gallery.workspace == identity.workspace)
            .order_by(Gallery.created_at.desc(), Gallery.id.desc())
            .offset(offset)
            .limit(limit)
        )
        items = []
        for row in rows:
            version = session.get(GalleryVersion, row.active_version_id) if row.active_version_id else None
            items.append(gallery_view(row, version.count if version else 0))
        next_offset = offset + len(items)
        return {"items": items, "total": total, "next_offset": next_offset if next_offset < total else None}


@router.post("", status_code=201)
def create_gallery(
    body: CreateGallery, request: Request, identity: Principal = Depends(require_role("admin"))
):
    with request.app.state.db.transaction() as session:
        row = Gallery(
            id=str(uuid4()), workspace=identity.workspace, name=body.name, description=body.description
        )
        session.add(row)
        session.flush()
        return gallery_view(row)


@router.get("/{gallery_id}")
def get_gallery(gallery_id: str, request: Request, identity: Principal = Depends(principal)):
    with request.app.state.db.transaction() as session:
        row = scoped(session, Gallery, gallery_id, identity.workspace)
        version = session.get(GalleryVersion, row.active_version_id) if row.active_version_id else None
        return gallery_view(row, version.count if version else 0)


@router.get("/{gallery_id}/versions")
def versions(gallery_id: str, request: Request, identity: Principal = Depends(principal)):
    with request.app.state.db.transaction() as session:
        gallery = scoped(session, Gallery, gallery_id, identity.workspace)
        rows = session.scalars(
            select(GalleryVersion)
            .where(GalleryVersion.gallery_id == gallery_id, GalleryVersion.workspace == identity.workspace)
            .order_by(GalleryVersion.number.desc())
        )
        return {"items": [version_view(row, gallery.active_version_id) for row in rows]}


def _decode_manifest(value: str | None, filenames: set[str]) -> dict:
    try:
        result = json.loads(value) if value else {}
    except (TypeError, ValueError) as exc:
        raise ServiceError("invalid_manifest", "Manifest должен быть JSON-объектом") from exc
    if not isinstance(result, dict) or set(result) - filenames:
        raise ServiceError("invalid_manifest", "Manifest содержит неизвестные имена файлов")
    for item in result.values():
        if not isinstance(item, dict) or set(item) - {"bbox", "external_id"}:
            raise ServiceError("invalid_manifest", "Допустимы только bbox и external_id")
        external_id = item.get("external_id")
        if external_id is not None and (not isinstance(external_id, str) or len(external_id) > 256):
            raise ServiceError("invalid_manifest", "external_id должен быть строкой до 256 символов")
    return result


def _prepare_images(state, uploaded: list[tuple[str, bytes]], manifest: dict) -> list[dict]:
    from ..inference import validate_image

    validated = []
    for filename, data in uploaded:
        item = manifest.get(filename, {})
        try:
            original, bbox, width, height = validate_image(data, item.get("bbox"))
        except ValueError as exc:
            raise ServiceError("invalid_image", f"Файл {filename}: {image_validation_message(exc)}") from exc
        validated.append(
            {
                "original": original,
                "bbox": list(bbox),
                "width": width,
                "height": height,
                "external_id": item.get("external_id"),
                "bbox_source": "explicit" if item.get("bbox") is not None else "full_frame",
            }
        )
    # No files are published to the catalog until every upload is validated.
    for item in validated:
        item["image_key"] = state.store.put(item.pop("original"), "observations", "img")
    return validated


@router.post("/{gallery_id}/imports", status_code=202)
async def import_images(
    gallery_id: str,
    request: Request,
    files: list[UploadFile] = File(...),
    manifest: str | None = Form(default=None, max_length=200000),
    identity: Principal = Depends(require_role("admin")),
):
    state = request.app.state
    state.runtime.require()
    with state.db.transaction() as session:
        scoped(session, Gallery, gallery_id, identity.workspace)
    if not files or len(files) > state.settings.max_import_files:
        raise ServiceError("import_limit", f"Допустимо от 1 до {state.settings.max_import_files} файлов", 413)
    filenames = [file.filename for file in files]
    if any(not name or len(name) > 255 for name in filenames) or len(set(filenames)) != len(filenames):
        raise ServiceError(
            "invalid_filenames", "Имена загружаемых файлов должны быть непустыми и уникальными"
        )
    parsed_manifest = _decode_manifest(manifest, set(filenames))
    uploaded, total = [], 0
    for file in files:
        data = await file.read(state.settings.max_upload_bytes + 1)
        await file.close()
        if len(data) > state.settings.max_upload_bytes:
            raise ServiceError("upload_too_large", "Изображение превышает допустимый размер", 413)
        total += len(data)
        if total > state.settings.max_request_bytes:
            raise ServiceError("upload_too_large", "Импорт превышает допустимый размер", 413)
        uploaded.append((file.filename, data))
    staged = await run_in_threadpool(_prepare_images, state, uploaded, parsed_manifest)
    job = await run_in_threadpool(
        prepare_import, state.db, state.runtime, gallery_id, identity.workspace, staged
    )
    return job_view(job)


@router.post("/{gallery_id}/versions/{version_id}/activate")
def activate_version(
    gallery_id: str,
    version_id: str,
    request: Request,
    body: ActivateVersion,
    identity: Principal = Depends(require_role("admin")),
):
    state = request.app.state
    row, count = activate(
        state.db,
        state.runtime,
        identity.workspace,
        gallery_id,
        version_id,
        expected=body.expected_active_version_id,
        compare=True,
    )
    return gallery_view(row, count)
