"""Real service state, durable jobs and configured model information."""

import os

import numpy as np
from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy import func, select, text

from .. import __version__
from ..auth import Principal, principal
from ..catalog import scoped
from ..errors import ServiceError
from ..models import Job, Search
from ..presenters import job_view
from ..response_schemas import (
    CalibrationList,
    CapabilitiesResponse,
    JobList,
    JobResponse,
    LiveResponse,
    ModelList,
    ReadyResponse,
    SystemResponse,
)

router = APIRouter(tags=["Operations"])


@router.get("/health/live", response_model=LiveResponse)
def live():
    return {"status": "alive", "version": __version__}


@router.get("/health/ready", response_model=ReadyResponse)
def ready(request: Request):
    state = request.app.state
    try:
        with state.db.transaction() as session:
            session.execute(text("SELECT 1"))
    except Exception as exc:
        raise ServiceError("not_ready", "База данных недоступна", 503) from exc
    if not os.access(state.store.root, os.W_OK):
        raise ServiceError("not_ready", "Хранилище недоступно для записи", 503)
    state.runtime.require()
    return {"status": "ready", "mode": state.runtime.capabilities()["mode"]}


@router.get("/capabilities", response_model=CapabilitiesResponse)
def capabilities(request: Request, _: Principal = Depends(principal)):
    state = request.app.state
    return {
        "service_version": __version__,
        "engine": state.runtime.capabilities(),
        "profiles": [{"id": "default", "name": "Визуальный поиск", "top_n_max": 50}],
        "limits": {
            "max_upload_bytes": state.settings.max_upload_bytes,
            "max_import_files": state.settings.max_import_files,
            "max_request_bytes": state.settings.max_request_bytes,
        },
        "features": {
            "immutable_snapshots": True,
            "replay": True,
            "occlusion_explanations": True,
            "durable_jobs": True,
            "exports": ["json", "csv", "pdf"],
            "trained_model_connected": False,
        },
        "notice": "Модель ReID не подключена. Preview проверяет работу сервиса и не подтверждает совпадения автомобилей.",
    }


@router.get("/models", response_model=ModelList)
def models(request: Request, _: Principal = Depends(principal)):
    engine = request.app.state.runtime.capabilities()
    return {
        "items": [
            {
                "id": engine["model_id"],
                "mode": engine["mode"],
                "dimension": engine["dimension"],
                "status": "ready" if engine["ready"] else "unavailable",
                "trained_model_connected": False,
                "description": "Preview: модель ReID не подключена",
            }
        ]
    }


@router.get("/calibrations", response_model=CalibrationList)
def calibrations(request: Request, _: Principal = Depends(principal)):
    policy = request.app.state.runtime.policy
    return {
        "items": [policy] if policy else [],
        "notice": "Без проверенной калибровки совпадения не подтверждаются",
    }


@router.get("/system", response_model=SystemResponse)
def system(request: Request, identity: Principal = Depends(principal)):
    state = request.app.state
    engine = state.runtime.capabilities()
    with state.db.transaction() as session:
        counts = dict(
            session.execute(
                select(Job.status, func.count())
                .where(Job.workspace == identity.workspace)
                .group_by(Job.status)
            ).all()
        )
        samples = [
            row.result["timings"]
            for row in session.scalars(
                select(Search)
                .where(Search.workspace == identity.workspace, Search.state == "completed")
                .order_by(Search.created_at.desc())
                .limit(500)
            )
        ]
    timings = {"sample_count": len(samples)}
    for key in ["total_ms", "load_ms", "encode_ms", "rank_ms"]:
        values = [sample[key] for sample in samples]
        timings[key] = (
            {"p50": float(np.percentile(values, 50)), "p95": float(np.percentile(values, 95))}
            if values
            else None
        )
    writable = os.access(state.store.root, os.W_OK)
    return {
        "status": "ready" if engine["ready"] and writable else "degraded",
        "dependencies": {
            "database": {"status": "ready"},
            "storage": {"status": "ready" if writable else "unavailable"},
            "model": {**engine, "status": "ready" if engine["ready"] else "unavailable"},
        },
        "jobs": {status: counts.get(status, 0) for status in ["pending", "running", "completed", "failed"]},
        "timings": timings,
        "timing_scope": "Last 500 completed searches in this workspace; includes preview processing, not a ReID benchmark",
    }


@router.get("/jobs", response_model=JobList)
def jobs(
    request: Request,
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    identity: Principal = Depends(principal),
):
    with request.app.state.db.transaction() as session:
        total = session.scalar(
            select(func.count()).select_from(Job).where(Job.workspace == identity.workspace)
        )
        rows = session.scalars(
            select(Job)
            .where(Job.workspace == identity.workspace)
            .order_by(Job.created_at.desc(), Job.id.desc())
            .offset(offset)
            .limit(limit)
        )
        items = [job_view(row) for row in rows]
        next_offset = offset + len(items)
        return {"items": items, "total": total, "next_offset": next_offset if next_offset < total else None}


@router.get("/jobs/{job_id}", response_model=JobResponse)
def get_job(job_id: str, request: Request, identity: Principal = Depends(principal)):
    with request.app.state.db.transaction() as session:
        return job_view(scoped(session, Job, job_id, identity.workspace))
