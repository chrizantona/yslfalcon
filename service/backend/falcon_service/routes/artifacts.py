"""Private artifact serving with the same authorization as the owning record."""

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import FileResponse

from ..auth import Principal, principal
from ..catalog import scoped
from ..models import Export, Observation, Search

router = APIRouter(tags=["Artifacts"])


def _image(request: Request, model, identifier: str, identity: Principal, crop: bool):
    from ..inference import crop_image

    state = request.app.state
    with state.db.transaction() as session:
        row = scoped(session, model, identifier, identity.workspace)
        key = row.query["image_key"] if model is Search else row.image_key
        bbox = row.query["bbox"] if model is Search else row.bbox
    data = state.store.read(key)
    if crop:
        data = crop_image(data, bbox)
    media_type = "image/png" if data.startswith(b"\x89PNG") else "image/jpeg"
    return Response(
        data,
        media_type=media_type,
        headers={"Cache-Control": "private, no-store", "X-Content-Type-Options": "nosniff"},
    )


@router.get("/observations/{observation_id}/image")
def observation_image(observation_id: str, request: Request, identity: Principal = Depends(principal)):
    return _image(request, Observation, observation_id, identity, False)


@router.get("/observations/{observation_id}/crop")
def observation_crop(observation_id: str, request: Request, identity: Principal = Depends(principal)):
    return _image(request, Observation, observation_id, identity, True)


@router.get("/searches/{search_id}/image")
def search_image(search_id: str, request: Request, identity: Principal = Depends(principal)):
    return _image(request, Search, search_id, identity, False)


@router.get("/searches/{search_id}/crop")
def search_crop(search_id: str, request: Request, identity: Principal = Depends(principal)):
    return _image(request, Search, search_id, identity, True)


@router.get("/exports/{export_id}/download")
def download_export(export_id: str, request: Request, identity: Principal = Depends(principal)):
    with request.app.state.db.transaction() as session:
        row = scoped(session, Export, export_id, identity.workspace)
    path = request.app.state.store.path(row.storage_key)
    if not path.is_file():
        from ..errors import ServiceError

        raise ServiceError("artifact_unavailable", "Файл отчёта недоступен", 410)
    media = {"json": "application/json", "csv": "text/csv", "pdf": "application/pdf"}[row.format]
    return FileResponse(
        path,
        media_type=media,
        filename=f"falcon-{row.search_id}.{row.format}",
        headers={"Cache-Control": "private, no-store", "ETag": f'"{row.sha256}"'},
    )
