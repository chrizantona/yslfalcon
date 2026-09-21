"""Explicit public representations keep storage keys and job payloads private."""

from .db import timestamp


def gallery_view(row, count: int = 0):
    return {
        "id": row.id,
        "name": row.name,
        "description": row.description,
        "active_version_id": row.active_version_id,
        "observation_count": count,
        "created_at": timestamp(row.created_at),
    }


def version_view(row, active_id: str | None = None):
    return {
        "id": row.id,
        "number": row.number,
        "status": row.status,
        "count": row.count,
        "model_id": row.model_id,
        "active": row.id == active_id,
        "manifest_hash": row.manifest_hash,
        "created_at": timestamp(row.created_at),
    }


def job_view(row):
    return {
        "id": row.id,
        "kind": row.kind,
        "status": row.status,
        "progress": row.progress,
        "message": row.message,
        "result": row.result,
        "error": row.error,
        "attempts": row.attempts,
        "created_at": timestamp(row.created_at),
        "completed_at": timestamp(row.completed_at) if row.completed_at else None,
    }


def review_view(row):
    return {
        "id": row.id,
        "search_id": row.search_id,
        "observation_id": row.observation_id,
        "verdict": row.verdict,
        "note": row.note,
        "actor": row.actor,
        "created_at": timestamp(row.created_at),
    }
