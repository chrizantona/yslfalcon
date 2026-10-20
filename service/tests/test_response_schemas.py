"""Public response imports retain the wire and durable-job compatibility contract."""

import hashlib
import json
import pickle
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import BaseModel, TypeAdapter, ValidationError

from falcon_service import response_schemas as contracts


def public_models():
    return {
        name: value
        for name, value in vars(contracts).items()
        if not name.startswith("_")
        and isinstance(value, type)
        and issubclass(value, BaseModel)
        and value is not BaseModel
    }


def test_all_response_model_schemas_retain_the_pre_split_wire_contract():
    # Includes every original class, collection alias and generated generic name,
    # in both Pydantic schema modes. An intentional wire change updates this digest:
    # the last one added the optional model_ms to SearchTimings and TimingSummary.
    schemas = {
        name: {mode: model.model_json_schema(mode=mode) for mode in ("validation", "serialization")}
        for name, model in public_models().items()
    }
    serialized = json.dumps(schemas, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    assert hashlib.sha256(serialized.encode()).hexdigest() == (
        "2d2f0621ced473df4e10e69a1e98caf542d1656a50b79ce3fc1b138763609e0d"
    )


def test_existing_response_class_paths_and_pickled_instances_remain_resolvable():
    for name, model in public_models().items():
        assert model.__module__ == "falcon_service.response_schemas", name
        assert getattr(contracts, model.__qualname__) is model
        assert pickle.loads(pickle.dumps(model)) is model
    response = contracts.ErrorResponse(
        error=contracts.ErrorDetail(code="dependency_unavailable", message="Storage unavailable"),
        request_id="request-42",
    )
    restored = pickle.loads(pickle.dumps(response))
    assert type(restored) is contracts.ErrorResponse
    assert type(restored.error) is contracts.ErrorDetail
    assert restored.model_dump() == response.model_dump()


@pytest.mark.parametrize(
    ("alias", "envelope", "item"),
    [
        ("GalleryList", "Page", "GalleryResponse"),
        ("GalleryVersionList", "Items", "GalleryVersionResponse"),
        ("ImportSessionList", "Items", "ImportSessionResponse"),
        ("SearchList", "Page", "SearchResponse"),
        ("JobList", "Page", "JobResponse"),
        ("ModelList", "Items", "ModelResponse"),
    ],
)
def test_collection_aliases_keep_generic_type_identity(alias, envelope, item):
    assert getattr(contracts, alias) is getattr(contracts, envelope)[getattr(contracts, item)]


def test_domain_contracts_import_before_the_legacy_facade_without_cycles():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from falcon_service.response_contracts.jobs import JobResponse; "
            "from falcon_service.response_contracts.galleries import GalleryList; "
            "from falcon_service import response_schemas; "
            "assert JobResponse is response_schemas.JobResponse; "
            "assert GalleryList is response_schemas.GalleryList",
        ],
        cwd=Path(__file__).resolve().parents[1] / "backend",
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_collection_validation_preserves_nested_types_constraints_and_strict_fields():
    gallery = {
        "id": "gallery-1",
        "name": "Imported gallery",
        "description": "",
        "active_version_id": None,
        "observation_count": 0,
        "created_at": "2026-10-17T10:00:00Z",
    }
    payload = {"items": [gallery], "total": 1, "next_offset": None}
    page = contracts.GalleryList.model_validate(payload)
    assert type(page.items[0]) is contracts.GalleryResponse
    assert page.model_dump(mode="json") == payload
    for invalid, location in [
        ({**payload, "total": -1}, ("total",)),
        ({**payload, "items": [{**gallery, "observation_count": -1}]}, ("items", 0, "observation_count")),
        ({**payload, "items": [{**gallery, "artifact_key": "private"}]}, ("items", 0, "artifact_key")),
    ]:
        with pytest.raises(ValidationError) as error:
            contracts.GalleryList.model_validate(invalid)
        assert location in [detail["loc"] for detail in error.value.errors()]


@pytest.mark.parametrize(
    ("result", "expected_type"),
    [
        (
            {
                "gallery_id": "gallery-1",
                "version_id": "version-1",
                "count": 1,
                "status": "ready",
                "manifest_hash": "abc",
            },
            contracts.ImportResult,
        ),
        (
            {"export_id": "export-1", "download_url": "/api/v1/exports/export-1/download", "sha256": "abc"},
            contracts.ExportResult,
        ),
        (
            {
                "method": "occlusion_sensitivity",
                "score_kind": "analysis_crop_embedding_cosine",
                "analysis_max_edge": 512,
                "analysis_sizes": {"query": [32, 32], "candidate": [32, 32]},
                "grid": 1,
                "baseline_score": 0.5,
                "query_map": [[0.1]],
                "candidate_map": [[0.2]],
                "fill": "crop_mean_rgb",
                "model_id": "legacy-model",
                "mode": "preview",
                "semantics": "legacy",
                "preview_warning": None,
                "search_id": "search-1",
                "observation_id": "observation-1",
                "gallery_version_id": "version-1",
            },
            contracts.LegacyExplanationResult,
        ),
        (None, type(None)),
    ],
)
def test_durable_jobs_keep_all_result_variants_readable(result, expected_type):
    payload = {
        "id": "job-1",
        "kind": "explanation" if expected_type is contracts.LegacyExplanationResult else "import",
        "status": "completed",
        "progress": 100,
        "message": "",
        "result": result,
        "error": None,
        "attempts": 1,
        "created_at": "2026-10-17T10:00:00Z",
        "completed_at": "2026-10-17T10:00:01Z",
    }
    job = contracts.JobResponse.model_validate(payload)
    assert type(job.result) is expected_type
    assert job.model_dump(mode="json") == payload
    with pytest.raises(ValidationError):
        contracts.JobResponse.model_validate({**payload, "progress": 101})


def test_shared_type_aliases_and_error_statuses_keep_the_public_contract():
    assert TypeAdapter(contracts.BBox).validate_python([0, 1, 32, 48]) == (0, 1, 32, 48)
    with pytest.raises(ValidationError):
        TypeAdapter(contracts.BBox).validate_python([0, 1, 32])
    roles = TypeAdapter(contracts.Role)
    for role in ("viewer", "operator", "admin"):
        assert roles.validate_python(role) == role
    with pytest.raises(ValidationError):
        roles.validate_python("owner")
    assert TypeAdapter(contracts.Timestamp).json_schema() == {"type": "string", "format": "date-time"}
    assert set(contracts.ERROR_RESPONSES) == {400, 401, 403, 404, 405, 409, 410, 413, 422, 500, 503}
    assert all(
        response["model"] is contracts.ErrorResponse for response in contracts.ERROR_RESPONSES.values()
    )
