"""Search, human review and asynchronous result artifacts."""

import json
from uuid import uuid4

from fastapi import APIRouter, Depends, File, Form, Header, Query, Request, UploadFile
from sqlalchemy import func, select
from starlette.concurrency import run_in_threadpool

from ..auth import Principal, principal, require_role
from ..catalog import scoped
from ..errors import ServiceError, image_validation_message
from ..jobs import enqueue
from ..models import Review, ReviewAnnotation, Search
from ..presenters import job_view, review_view
from ..response_schemas import JobResponse, ReviewResponse, SearchList, SearchResponse
from ..schemas import CreateExplanation, CreateExport, CreateReview
from ..search import execute_search, reserve_search, search_view

router = APIRouter(prefix="/searches", tags=["Searches"])


@router.get("", response_model=SearchList)
def searches(
    request: Request,
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    identity: Principal = Depends(principal),
):
    with request.app.state.db.transaction() as session:
        total = session.scalar(
            select(func.count())
            .select_from(Search)
            .where(Search.workspace == identity.workspace, Search.state == "completed")
        )
        rows = session.scalars(
            select(Search)
            .where(Search.workspace == identity.workspace, Search.state == "completed")
            .order_by(Search.created_at.desc(), Search.id.desc())
            .offset(offset)
            .limit(limit)
        )
        items = [search_view(session, row) for row in rows]
        next_offset = offset + len(items)
        return {"items": items, "total": total, "next_offset": next_offset if next_offset < total else None}


def _search(state, workspace, gallery_id, data, bbox, top_n, key):
    from ..inference import validate_image

    try:
        original, validated_bbox, width, height = validate_image(data, bbox)
    except ValueError as exc:
        raise ServiceError("invalid_image", image_validation_message(exc)) from exc
    search_id = reserve_search(
        state.db,
        state.store,
        state.runtime,
        workspace,
        gallery_id,
        original,
        list(validated_bbox),
        width,
        height,
        top_n,
        key,
        bbox_source="explicit" if bbox is not None else "full_frame",
    )
    return execute_search(state.db, state.store, state.runtime, state.settings, workspace, search_id)


@router.post("", response_model=SearchResponse)
async def create_search(
    request: Request,
    image: UploadFile = File(...),
    gallery_id: str = Form(...),
    bbox: str | None = Form(default=None, max_length=256),
    top_n: int = Form(default=10, ge=1, le=50),
    idempotency_key: str | None = Header(default=None),
    identity: Principal = Depends(require_role("operator")),
):
    data = await image.read(request.app.state.settings.max_upload_bytes + 1)
    await image.close()
    if len(data) > request.app.state.settings.max_upload_bytes:
        raise ServiceError("upload_too_large", "Изображение превышает допустимый размер", 413)
    try:
        parsed_bbox = json.loads(bbox) if bbox else None
    except ValueError as exc:
        raise ServiceError("invalid_bbox", "BBox должен быть JSON-массивом [x,y,w,h]") from exc
    return await run_in_threadpool(
        _search, request.app.state, identity.workspace, gallery_id, data, parsed_bbox, top_n, idempotency_key
    )


@router.get("/{search_id}", response_model=SearchResponse)
def get_search(search_id: str, request: Request, identity: Principal = Depends(principal)):
    with request.app.state.db.transaction() as session:
        return search_view(session, scoped(session, Search, search_id, identity.workspace))


@router.post("/{search_id}/replays", response_model=SearchResponse)
def replay_search(
    search_id: str,
    request: Request,
    idempotency_key: str | None = Header(default=None),
    identity: Principal = Depends(require_role("operator")),
):
    state = request.app.state
    with state.db.transaction() as session:
        original = scoped(session, Search, search_id, identity.workspace)
        search_view(session, original)
    data = state.store.read(original.query["image_key"])
    replay_id = reserve_search(
        state.db,
        state.store,
        state.runtime,
        identity.workspace,
        original.gallery_id,
        data,
        original.query["bbox"],
        original.query["width"],
        original.query["height"],
        original.top_n,
        idempotency_key,
        pinned=original,
        bbox_source=original.query["bbox_source"],
    )
    return execute_search(state.db, state.store, state.runtime, state.settings, identity.workspace, replay_id)


def _completed_search(session, search_id, workspace, observation_id=None):
    row = scoped(session, Search, search_id, workspace)
    result = search_view(session, row)
    if observation_id is not None and observation_id not in {
        item["observation_id"] for item in result["ranked_candidates"]
    }:
        raise ServiceError("invalid_candidate", "Наблюдение не входит в сохранённую выдачу", 400)
    return row


@router.post("/{search_id}/reviews", status_code=201, response_model=ReviewResponse)
def create_review(
    search_id: str,
    body: CreateReview,
    request: Request,
    identity: Principal = Depends(require_role("operator")),
):
    with request.app.state.db.transaction() as session:
        search = _completed_search(session, search_id, identity.workspace, body.observation_id)
        valid_refs = {"query"} | {item["observation_id"] for item in search.result["ranked_candidates"]}
        if any(item.image_ref not in valid_refs for item in body.annotations):
            raise ServiceError(
                "invalid_annotation", "Пометка относится к изображению вне сохранённой выдачи", 400
            )
        review = Review(
            id=str(uuid4()),
            workspace=identity.workspace,
            search_id=search_id,
            observation_id=body.observation_id,
            verdict=body.verdict,
            note=body.note,
            actor=identity.actor,
        )
        session.add(review)
        session.flush()
        annotations = [
            ReviewAnnotation(
                id=str(uuid4()),
                workspace=identity.workspace,
                review_id=review.id,
                image_ref=item.image_ref,
                surface=item.surface,
                kind=item.kind,
                x=item.x,
                y=item.y,
            )
            for item in body.annotations
        ]
        session.add_all(annotations)
        session.flush()
        return review_view(review, annotations)


@router.post("/{search_id}/explanations", status_code=202, response_model=JobResponse)
def create_explanation(
    search_id: str,
    body: CreateExplanation,
    request: Request,
    identity: Principal = Depends(require_role("operator")),
):
    with request.app.state.db.transaction() as session:
        _completed_search(session, search_id, identity.workspace, body.observation_id)
    return job_view(
        enqueue(
            request.app.state.db,
            identity.workspace,
            "explanation",
            {
                "search_id": search_id,
                "observation_id": body.observation_id,
            },
        )
    )


@router.post("/{search_id}/exports", status_code=202, response_model=JobResponse)
def create_export(
    search_id: str,
    body: CreateExport,
    request: Request,
    identity: Principal = Depends(require_role("operator")),
):
    with request.app.state.db.transaction() as session:
        _completed_search(session, search_id, identity.workspace)
    return job_view(
        enqueue(
            request.app.state.db,
            identity.workspace,
            "export",
            {
                "search_id": search_id,
                "format": body.format,
                "export_id": str(uuid4()),
            },
        )
    )
