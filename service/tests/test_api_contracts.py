"""Consumers can discover and rely on the actual public response/error contract."""

import io
import json
import logging

import numpy as np
import pytest
from fastapi.testclient import TestClient

from falcon_service.app import create_app
from falcon_service.config import Settings
from falcon_service.inference import FeatureBundle, PreviewEncoder
from falcon_service.inference.wire import pack_features, unpack_features
from falcon_service.logging_setup import JsonFormatter
from falcon_service.model_runtime import create_app as create_runtime
from falcon_service.models import Job
from falcon_service.response_schemas import ErrorResponse
from falcon_service.runtime_schemas import RuntimeErrorResponse

TOKEN = "contract-test-credential-" + "x" * 32
HEADERS = {"Authorization": f"Bearer {TOKEN}"}


@pytest.fixture
def application(tmp_path):
    return create_app(
        Settings(
            database_url=f"sqlite:///{tmp_path}/contract.db",
            storage_root=tmp_path / "artifacts",
            engine_mode="preview",
            allowed_origins=["http://testserver"],
            auth_tokens={TOKEN: {"role": "admin", "workspace": "contracts"}},
            max_request_bytes=4096,
        )
    )


def test_all_operations_have_explicit_success_and_actual_error_contracts(application):
    schema = application.openapi()
    assert "/api/v1/searches/{search_id}/explanations" not in schema["paths"]
    for path, operations in schema["paths"].items():
        for method, operation in operations.items():
            for status, response in operation["responses"].items():
                if status == "204":
                    assert "content" not in response
                elif status.startswith("2"):
                    assert response.get("content"), (path, method, status)
                    for content in response["content"].values():
                        assert content.get("schema"), (path, method, status)
                else:
                    assert response["content"]["application/json"]["schema"] == {
                        "$ref": "#/components/schemas/ErrorResponse"
                    }, (path, method, status)
    assert "HTTPValidationError" not in schema["components"]["schemas"]
    search = schema["paths"]["/api/v1/searches"]["post"]
    assert search["security"] == [{"BearerAuth": []}, {"SessionCookie": []}]
    assert search["responses"]["200"]["content"]["application/json"]["schema"] == {
        "$ref": "#/components/schemas/SearchResponse"
    }


def test_binary_routes_declare_the_real_content_types(application):
    paths = application.openapi()["paths"]
    image = paths["/api/v1/searches/{search_id}/image"]["get"]["responses"]["200"]
    assert set(image["content"]) == {"image/jpeg", "image/png"}
    exported = paths["/api/v1/exports/{export_id}/download"]["get"]["responses"]["200"]
    assert set(exported["content"]) == {
        "application/json",
        "text/csv",
        "application/pdf",
        "application/zip",
    }
    assert exported["content"]["application/json"]["schema"] == {"$ref": "#/components/schemas/SearchReport"}


@pytest.mark.parametrize(
    ("path", "headers", "body", "status", "code"),
    [
        ("/api/v1/galleries", {}, {"name": "Private"}, 401, "unauthorized"),
        ("/api/v1/galleries", HEADERS, {"name": ""}, 422, "invalid_request"),
        ("/api/v1/auth/session", {}, {"token": TOKEN}, 403, "origin_forbidden"),
        ("/api/v1/galleries", HEADERS, {"name": "x" * 5000}, 413, "request_too_large"),
    ],
)
def test_error_payload_and_header_share_the_incoming_request_id(
    application, path, headers, body, status, code
):
    with TestClient(application) as client:
        response = client.post(path, headers={**headers, "X-Request-ID": "partner.batch-42"}, json=body)
    assert response.status_code == status
    parsed = ErrorResponse.model_validate(response.json())
    assert parsed.error.code == code
    assert parsed.request_id == response.headers["X-Request-ID"] == "partner.batch-42"
    assert TOKEN not in response.text


def test_unsafe_request_id_is_replaced_and_docs_are_complete_offline(application):
    with TestClient(application) as client:
        response = client.get("/api/v1/health/live", headers={"X-Request-ID": "unsafe supplied id"})
        assert response.status_code == 200
        assert response.headers["X-Request-ID"] != "unsafe supplied id"
        docs = client.get("/docs")
        assert docs.status_code == 200
        assert "SearchResponse" in docs.text and "ErrorResponse" in docs.text
        assert "Request body" in docs.text and "Idempotency-Key" in docs.text
        assert "<script" not in docs.text
        assert "https://" not in docs.text


def test_unexpected_error_keeps_correlation_and_logs_no_exception_values(application):
    @application.get("/contract-crash", include_in_schema=False)
    def crash():
        raise ValueError("private diagnostic " + TOKEN)

    output = io.StringIO()
    handler = logging.StreamHandler(output)
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger("falcon_service.app")
    logger.addHandler(handler)
    try:
        with TestClient(application, raise_server_exceptions=False) as client:
            response = client.get("/contract-crash", headers={"X-Request-ID": "partner.crash-1"})
    finally:
        logger.removeHandler(handler)
    assert response.status_code == 500
    assert response.json()["request_id"] == response.headers["X-Request-ID"] == "partner.crash-1"
    assert TOKEN not in response.text and TOKEN not in output.getvalue()
    events = [json.loads(line) for line in output.getvalue().splitlines()]
    failure = next(event for event in events if event["message"] == "http_unexpected_error")
    assert failure["request_id"] == "partner.crash-1"
    assert failure["error_type"] == "ValueError"


def test_internal_runtime_documents_binary_transport_and_actual_auth_errors():
    runtime = create_runtime(PreviewEncoder(), TOKEN)
    schema = runtime.openapi()
    for path in ("/internal/v1/encode", "/internal/v1/rerank"):
        operation = schema["paths"][path]["post"]
        assert operation["security"] == [{"RuntimeBearer": []}]
        assert set(operation["responses"]["200"]["content"]) == {"application/octet-stream"}
        assert "allow_pickle=False" in operation["responses"]["200"]["description"]
        assert operation["responses"]["422"]["content"]["application/json"]["schema"] == {
            "$ref": "#/components/schemas/RuntimeErrorResponse"
        }
    request_body = schema["paths"]["/internal/v1/rerank"]["post"]["requestBody"]
    assert set(request_body["content"]) == {"application/octet-stream"}
    assert "similarities" in request_body["description"]
    with TestClient(runtime) as client:
        unauthenticated = client.get("/internal/v1/capabilities")
        assert unauthenticated.status_code == 401
        assert isinstance(RuntimeErrorResponse.model_validate(unauthenticated.json()).detail, str)
        invalid = client.post("/internal/v1/encode", headers=HEADERS)
        assert invalid.status_code == 422
        assert isinstance(RuntimeErrorResponse.model_validate(invalid.json()).detail, list)


def test_public_capabilities_report_an_injected_production_runtime(tmp_path):
    class ProductionRuntime(PreviewEncoder):
        mode = "production"
        model_id = "falcon-serving-release-fixture"
        real_model_connected = True
        device = "cuda:0"
        precision = "float16"
        weights_bytes = 123_456_789

    settings = Settings(
        database_url=f"sqlite:///{tmp_path}/production-contract.db",
        storage_root=tmp_path / "artifacts",
        engine_mode="remote",
        expected_model_id=ProductionRuntime.model_id,
        expected_weights_bytes=ProductionRuntime.weights_bytes,
        auth_tokens={TOKEN: {"role": "admin", "workspace": "contracts"}},
    )
    with TestClient(create_app(settings, ProductionRuntime())) as client:
        capabilities = client.get("/api/v1/capabilities", headers=HEADERS).json()
        models = client.get("/api/v1/models", headers=HEADERS).json()
    assert capabilities["engine"] == {
        "mode": "production",
        "model_id": ProductionRuntime.model_id,
        "dimension": ProductionRuntime.dimension,
        "ready": True,
        "real_model_connected": True,
        "device": "cuda:0",
        "precision": "float16",
        "weights_bytes": 123_456_789,
    }
    assert capabilities["features"]["trained_model_connected"] is True
    assert capabilities["features"]["hnsw_index"] is True
    assert models["items"][0]["trained_model_connected"] is True
    assert models["items"][0]["weights_bytes"] == 123_456_789


def test_legacy_explanation_jobs_remain_readable_after_creation_route_is_removed(application):
    legacy_result = {
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
        "search_id": "legacy-search",
        "observation_id": "legacy-observation",
        "gallery_version_id": "legacy-version",
    }
    with TestClient(application) as client:
        with client.app.state.db.transaction() as session:
            session.add(
                Job(
                    id="legacy-explanation-job",
                    workspace="contracts",
                    kind="explanation",
                    payload={},
                    status="completed",
                    progress=100,
                    result=legacy_result,
                )
            )
        item = client.get("/api/v1/jobs/legacy-explanation-job", headers=HEADERS)
        listing = client.get("/api/v1/jobs", headers=HEADERS)
    assert item.status_code == 200
    assert item.json()["result"] == legacy_result
    assert listing.status_code == 200
    assert listing.json()["items"][0]["kind"] == "explanation"


def test_documented_runtime_rerank_wire_round_trip():
    encoder = PreviewEncoder()
    feature = FeatureBundle(np.eye(1, encoder.dimension, dtype=np.float32)[0], None, encoder.model_id)
    payload = pack_features([feature, feature], similarities=np.array([1.0], dtype=np.float32))
    with TestClient(create_runtime(encoder, TOKEN)) as client:
        response = client.post(
            "/internal/v1/rerank",
            content=payload,
            headers={**HEADERS, "Content-Type": "application/octet-stream"},
        )
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/octet-stream"
    features, extras = unpack_features(response.content)
    assert len(features) == 1 and features[0].model_id == encoder.model_id
    np.testing.assert_array_equal(extras["scores"], np.array([1.0], dtype=np.float32))
