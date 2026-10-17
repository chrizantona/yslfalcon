"""Readiness reflects the live runtime, not only its cached startup handshake."""

from io import BytesIO

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from falcon_service.app import create_app
from falcon_service.config import Settings
from falcon_service.inference import PreviewEncoder, RemoteEncoder
from falcon_service.inference.wire import pack_features

TOKEN = "readiness-private-token-32-characters"


@pytest.fixture
def runtime(monkeypatch):
    preview = PreviewEncoder()
    capabilities = {
        "mode": preview.mode,
        "model_id": preview.model_id,
        "dimension": preview.dimension,
        "rerank_window": 0,
        "calibration_policy": None,
        "real_model_connected": False,
        "device": "cpu",
        "precision": "float32",
        "weights_bytes": 0,
    }
    state = {"failure": None, "requests": []}

    def route(request):
        state["requests"].append(request)
        assert request.headers["Authorization"] == f"Bearer {TOKEN}"
        if request.url.path == "/internal/v1/encode":
            return httpx.Response(200, content=pack_features([preview.encode(state["image"])]))
        assert request.url.path == "/internal/v1/capabilities"
        failure = state["failure"]
        if failure == "timeout":
            raise httpx.ReadTimeout(TOKEN, request=request)
        if failure == "disconnected":
            raise httpx.ConnectError(TOKEN, request=request)
        if isinstance(failure, int):
            return httpx.Response(failure, text=TOKEN)
        if failure == "invalid_json":
            return httpx.Response(200, text=TOKEN)
        payload = dict(capabilities)
        if failure == "invalid_schema":
            payload.pop("model_id")
        elif failure == "model_changed":
            payload["model_id"] = "other-model"
        elif failure == "calibration_changed":
            payload["calibration_policy"] = {"threshold": 123}
        return httpx.Response(200, json=payload)

    client_class = httpx.Client
    with monkeypatch.context() as patch:
        patch.setattr(
            "falcon_service.inference.remote.httpx.Client",
            lambda **kwargs: client_class(transport=httpx.MockTransport(route), **kwargs),
        )
        remote = RemoteEncoder("http://runtime.local", TOKEN)
    try:
        yield remote, state
    finally:
        remote.close()


@pytest.mark.parametrize(
    "failure",
    [
        "timeout",
        "disconnected",
        401,
        503,
        "invalid_json",
        "invalid_schema",
        "model_changed",
        "calibration_changed",
    ],
)
def test_api_readiness_detects_runtime_failure_and_recovers(tmp_path, runtime, failure):
    remote, state = runtime
    settings = Settings(
        database_url=f"sqlite:///{tmp_path}/readiness.db",
        storage_root=tmp_path / "artifacts",
        engine_mode="remote",
        runtime_token=TOKEN,
    )
    with TestClient(create_app(settings, remote)) as client:
        assert client.get("/api/v1/health/ready").status_code == 200
        assert len(state["requests"]) == 2
        state["failure"] = failure
        response = client.get("/api/v1/health/ready", headers={"X-Request-ID": "readiness-123"})
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "model_unavailable"
        assert TOKEN not in response.text
        assert response.headers["X-Request-ID"] == "readiness-123"
        assert state["requests"][-1].headers["X-Request-ID"] == "readiness-123"
        assert state["requests"][-1].extensions["timeout"] == dict.fromkeys(
            ["connect", "read", "write", "pool"], 3
        )
        assert client.get("/api/v1/health/live").status_code == 200
        assert client.app.state.runtime.require() is remote
        state["failure"] = None
        assert client.get("/api/v1/health/ready").status_code == 200
        assert len(state["requests"]) == 4


def test_readiness_retries_a_failed_startup_with_a_short_network_timeout(tmp_path, monkeypatch, runtime):
    remote, state = runtime
    settings = Settings(
        database_url=f"sqlite:///{tmp_path}/cold-start.db",
        storage_root=tmp_path / "artifacts",
        engine_mode="remote",
        runtime_token=TOKEN,
        runtime_url="http://runtime.local",
    )
    client_class = httpx.Client
    transport = remote._client._transport
    app = create_app(settings)
    client = TestClient(app)
    with monkeypatch.context() as patch:
        patch.setattr(
            "falcon_service.inference.remote.httpx.Client",
            lambda **kwargs: client_class(transport=transport, **kwargs),
        )
        state["failure"] = "disconnected"
        with client:
            assert app.state.runtime.encoder is None
            assert state["requests"][-1].extensions["timeout"]["read"] == 60
            assert client.get("/api/v1/health/ready").status_code == 503
            assert state["requests"][-1].extensions["timeout"]["read"] == 3
            state["failure"] = None
            assert client.get("/api/v1/health/ready").status_code == 200
            assert len(state["requests"]) == 5
            assert all(request.extensions["timeout"]["read"] == 3 for request in state["requests"][-3:])
            recovered = app.state.runtime.encoder
            assert recovered._client.timeout == httpx.Timeout(60)
            image = BytesIO()
            Image.new("RGB", (24, 24), "red").save(image, "PNG")
            state["image"] = image.getvalue()
            assert recovered.encode(state["image"]).model_id == remote.model_id
            assert state["requests"][-1].extensions["timeout"]["read"] == 60
        recovered.close()
