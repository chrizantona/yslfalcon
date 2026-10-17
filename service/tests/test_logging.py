"""Operational logs follow real HTTP/worker outcomes without exposing request data."""

import io
import json
import logging
import os
import socket
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import select

from falcon_service.app import create_app
from falcon_service.config import Settings
from falcon_service.inference import PreviewEncoder, RemoteEncoder
from falcon_service.inference.wire import pack_features
from falcon_service.jobs import enqueue, run_once
from falcon_service.logging_setup import JsonFormatter, request_id_context
from falcon_service.model_runtime import create_app as create_runtime
from falcon_service.models import Job
from falcon_service.runtime import EncoderRuntime

SECRET = "private-token-never-log-this"


@pytest.fixture
def events():
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    logger = logging.getLogger("falcon_service")
    previous_level = logger.level
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    try:
        yield lambda: [json.loads(line) for line in stream.getvalue().splitlines()]
    finally:
        logger.removeHandler(handler)
        logger.setLevel(previous_level)


@pytest.fixture
def app(tmp_path):
    return create_app(
        Settings(database_url=f"sqlite:///{tmp_path}/logs.db", storage_root=tmp_path / "artifacts")
    )


def http_events(events):
    return [event for event in events() if event["message"] == "http_request"]


def test_api_logs_success_rejection_and_template_without_request_data(app, events):
    with TestClient(app) as client:
        response = client.get("/api/v1/health/live", headers={"X-Request-ID": "trace-123"})
        assert response.headers["X-Request-ID"] == "trace-123"
        denied = client.get(
            f"/api/v1/galleries/{SECRET}?token={SECRET}",
            headers={"Authorization": f"Bearer {SECRET}"},
        )
        assert denied.status_code == 401
        assert (
            client.post(
                "/api/v1/auth/session", json={"token": SECRET}, headers={"Origin": "http://localhost:8080"}
            ).status_code
            == 401
        )
    successful, rejected, _ = http_events(events)
    assert (
        successful.items()
        >= {
            "service": "falcon_service",
            "component": "api",
            "request_id": "trace-123",
            "method": "GET",
            "route": "/health/live",
            "status": 200,
            "level": "info",
        }.items()
    )
    assert rejected["route"] == "/galleries/{gallery_id}"
    assert rejected["level"] == "warning"
    assert all(event["duration_ms"] >= 0 for event in http_events(events))
    assert SECRET not in json.dumps(events())


def test_api_logs_unhandled_error_and_returns_matching_request_id(app, events):
    @app.get("/fail")
    def fail():
        raise RuntimeError(SECRET)

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/fail", headers={"X-Request-ID": "failure-123"})
    assert response.status_code == 500
    assert response.json()["request_id"] == response.headers["X-Request-ID"] == "failure-123"
    [event] = http_events(events)
    assert event["status"] == 500
    assert event["level"] == "error"
    assert event["exception_type"] == "RuntimeError"
    assert any(frame["function"] == "fail" for frame in event["traceback"])
    assert SECRET not in json.dumps(events())


def test_invalid_request_ids_are_replaced_and_parallel_contexts_are_isolated(app, events):
    with TestClient(app) as client:
        response = client.get("/api/v1/health/live", headers={"X-Request-ID": "invalid id"})
        assert response.headers["X-Request-ID"] != "invalid id"
        assert len(response.headers["X-Request-ID"]) == 36

        @app.get("/context")
        def context():
            logging.getLogger("falcon_service.test").info("inside_request")
            return {"request_id": request_id_context.get()}

        def get_context(identifier):
            result = client.get("/context", headers={"X-Request-ID": identifier})
            assert result.json()["request_id"] == identifier
            assert result.headers["X-Request-ID"] == identifier

        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(get_context, [f"concurrent-{index}" for index in range(8)]))
    nested = [event for event in events() if event["message"] == "inside_request"]
    assert {event["request_id"] for event in nested} == {f"concurrent-{index}" for index in range(8)}
    assert request_id_context.get() is None


def test_runtime_uses_same_logging_and_request_context(events):
    with TestClient(create_runtime(PreviewEncoder(), SECRET)) as client:
        response = client.get("/health/live", headers={"X-Request-ID": "runtime-123"})
        assert response.headers["X-Request-ID"] == "runtime-123"
        assert client.get("/internal/v1/capabilities").status_code == 401
    success, rejected = http_events(events)
    assert success["component"] == rejected["component"] == "runtime"
    assert success["status"] == 200
    assert rejected["status"] == 401
    assert {event["message"] for event in events()} >= {"runtime_started", "runtime_stopped"}
    assert SECRET not in json.dumps(events())


def test_runtime_unexpected_error_returns_request_id_and_safe_error(events):
    app = create_runtime(PreviewEncoder(), SECRET)

    @app.get("/fail")
    def fail():
        raise RuntimeError(SECRET)

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/fail", headers={"X-Request-ID": "runtime-failure-123"})
    assert response.status_code == 500
    assert response.headers["X-Request-ID"] == "runtime-failure-123"
    assert response.headers["Cache-Control"] == "no-store"
    assert SECRET not in response.text
    [event] = http_events(events)
    assert event["request_id"] == "runtime-failure-123"
    assert event["status"] == 500
    assert event["exception_type"] == "RuntimeError"
    assert SECRET not in json.dumps(events())


def test_real_uvicorn_does_not_repeat_raw_exception_values(tmp_path):
    source = """
import os
import sys
from pathlib import Path
import uvicorn
from falcon_service.app import create_app
from falcon_service.config import Settings

root = Path(os.environ['FALCON_TEST_LOG_ROOT'])
app = create_app(Settings(database_url=f'sqlite:///{root}/test.db', storage_root=root / 'artifacts'))
@app.get('/fail')
def fail():
    raise ValueError('private-token-never-log-this')
uvicorn.run(app, fd=int(sys.argv[1]), access_log=False)
"""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        url = f"http://127.0.0.1:{listener.getsockname()[1]}"
        process = subprocess.Popen(
            [sys.executable, "-c", source, str(listener.fileno())],
            pass_fds=(listener.fileno(),),
            env={**os.environ, "PYTHONPATH": "backend", "FALCON_TEST_LOG_ROOT": str(tmp_path)},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            with httpx.Client(base_url=url, timeout=0.5) as client:
                deadline = time.monotonic() + 10
                while True:
                    try:
                        if client.get("/api/v1/health/live").status_code == 200:
                            break
                    except httpx.TransportError:
                        pass
                    assert process.poll() is None, "Uvicorn exited before startup"
                    assert time.monotonic() < deadline, "Uvicorn startup timed out"
                    time.sleep(0.05)
                response = client.get("/fail", headers={"X-Request-ID": "uvicorn-failure-123"})
                assert response.status_code == 500
                assert response.headers["X-Request-ID"] == "uvicorn-failure-123"
        finally:
            process.terminate()
            try:
                output, _ = process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                output, _ = process.communicate(timeout=5)
    assert SECRET not in output
    events = [json.loads(line) for line in output.splitlines() if line.startswith("{")]
    server_error = next(event for event in events if event["message"] == "server_error")
    assert server_error["logger"] == "uvicorn.error"
    assert server_error["exception_type"] == "ValueError"
    assert server_error["traceback"]


@pytest.mark.parametrize("component", ["api", "runtime"])
def test_real_uvicorn_configuration_failure_is_safe_json(component):
    marker = "private-configuration-marker"
    module = "falcon_service.app" if component == "api" else "falcon_service.model_runtime"
    invalid_config = (
        {"FALCON_AUTH_TOKENS": json.dumps({marker: {"role": "invalid", "workspace": "local"}})}
        if component == "api"
        else {"FALCON_RUNTIME_MAX_CONCURRENCY": marker}
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "uvicorn",
            f"{module}:app",
            "--host",
            "127.0.0.1",
            "--port",
            "0",
            "--no-access-log",
        ],
        env={
            **{key: value for key, value in os.environ.items() if not key.startswith("FALCON_")},
            "PYTHONPATH": "backend",
            **invalid_config,
        },
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 1
    output = result.stdout + result.stderr
    assert "Traceback (most recent call last)" not in output
    assert marker[:12] not in output
    [failure] = [json.loads(line) for line in output.splitlines()]
    assert failure["message"] == f"{component}_startup_failed"
    assert failure["component"] == component
    assert failure["exception_type"] == ("ValidationError" if component == "api" else "RuntimeError")
    assert failure["traceback"]


@pytest.mark.parametrize("component", ["api", "runtime"])
def test_real_uvicorn_logs_safe_startup_failure_metadata(component):
    source = """
import sys
import uvicorn
from falcon_service.app import create_app
from falcon_service.config import Settings
from falcon_service.db import Database
from falcon_service import model_runtime

def fail(*args, **kwargs):
    raise ValueError('private-token-never-log-this')

if sys.argv[1] == 'api':
    Database.initialize = fail
    app = create_app(Settings(database_url='sqlite:///:memory:'))
else:
    model_runtime.PreviewEncoder = fail
    app = model_runtime.create_app(token='private-token-never-log-this')
uvicorn.run(app, host='127.0.0.1', port=0, access_log=False)
"""
    result = subprocess.run(
        [sys.executable, "-c", source, component],
        env={
            **{key: value for key, value in os.environ.items() if not key.startswith("FALCON_")},
            "PYTHONPATH": "backend",
        },
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 3
    assert SECRET not in result.stdout + result.stderr
    events = [json.loads(line) for line in result.stderr.splitlines() if line.startswith("{")]
    failure = next(event for event in events if event["message"] == f"{component}_startup_failed")
    assert failure["component"] == component
    assert failure["level"] == "error"
    assert failure["exception_type"] == "ValueError"
    assert any(frame["function"] == "fail" for frame in failure["traceback"])
    assert not any(event["message"] == f"{component}_started" for event in events)


def test_api_startup_failure_closes_database_and_propagates(app, events, monkeypatch):
    closed = []

    def fail(_database):
        raise RuntimeError(SECRET)

    monkeypatch.setattr("falcon_service.app.Database.initialize", fail)
    monkeypatch.setattr("falcon_service.app.Database.close", lambda database: closed.append(database))
    with pytest.raises(RuntimeError, match=SECRET), TestClient(app):
        pass
    assert closed == [app.state.db]
    assert any(event["message"] == "api_startup_failed" for event in events())
    assert SECRET not in json.dumps(events())


def test_worker_startup_failure_exits_with_safe_json_and_closes_database():
    source = """
import logging
from falcon_service import worker

def fail(_database):
    raise RuntimeError('private-token-never-log-this')

def close(_database):
    logging.getLogger('falcon_service.worker').info('test_database_closed')

worker.Database.initialize = fail
worker.Database.close = close
worker.main()
"""
    result = subprocess.run(
        [sys.executable, "-c", source],
        env={
            **{key: value for key, value in os.environ.items() if not key.startswith("FALCON_")},
            "PYTHONPATH": "backend",
        },
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 1
    assert SECRET not in result.stdout + result.stderr
    assert "Traceback (most recent call last)" not in result.stderr
    events = [json.loads(line) for line in result.stderr.splitlines()]
    failure = next(event for event in events if event["message"] == "worker_startup_failed")
    assert failure["component"] == "worker"
    assert failure["level"] == "error"
    assert failure["exception_type"] == "RuntimeError"
    assert any(frame["function"] == "fail" for frame in failure["traceback"])
    assert any(event["message"] == "test_database_closed" for event in events)
    assert not any(event["message"] == "worker_started" for event in events)


def test_worker_healthcheck_failure_uses_safe_json_logging():
    source = """
import sys
import time
from types import SimpleNamespace
from falcon_service import inference, worker
from falcon_service.config import Settings

def fail():
    raise RuntimeError('private-token-never-log-this')

worker.HEARTBEAT_FILE = SimpleNamespace(stat=lambda: SimpleNamespace(st_mtime=time.time()))
worker.Settings = lambda: Settings(database_url='sqlite:///:memory:')
inference.PreviewEncoder = fail
sys.argv.append('--healthcheck')
worker.main()
"""
    result = subprocess.run(
        [sys.executable, "-c", source],
        env={
            **{key: value for key, value in os.environ.items() if not key.startswith("FALCON_")},
            "PYTHONPATH": "backend",
        },
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 1
    assert SECRET not in result.stdout + result.stderr
    assert "Traceback (most recent call last)" not in result.stderr
    [failure] = [json.loads(line) for line in result.stderr.splitlines()]
    assert failure["message"] == "encoder_initialization_failed"
    assert failure["exception_type"] == "RuntimeError"
    assert any(frame["function"] == "fail" for frame in failure["traceback"])


def test_encoder_initialization_failure_logs_safe_metadata_and_preserves_unavailable_state(
    events, monkeypatch
):
    def fail():
        raise RuntimeError(SECRET)

    monkeypatch.setattr("falcon_service.inference.PreviewEncoder", fail)
    runtime = EncoderRuntime(Settings())
    runtime.initialize()
    assert runtime.encoder is None
    assert runtime.error == SECRET
    [failure] = events()
    assert failure["message"] == "encoder_initialization_failed"
    assert failure["level"] == "error"
    assert failure["exception_type"] == "RuntimeError"
    assert any(frame["function"] == "fail" for frame in failure["traceback"])
    assert SECRET not in json.dumps(events())


def test_remote_adapter_forwards_request_id_without_leaking_between_calls(monkeypatch):
    encoder = PreviewEncoder()
    image = io.BytesIO()
    Image.new("RGB", (24, 24), "red").save(image, "PNG")
    image_bytes = image.getvalue()
    feature = encoder.encode(image_bytes)
    requests = []
    client_class = httpx.Client

    def route(request):
        requests.append(request)
        if request.method == "GET":
            return httpx.Response(
                200,
                json={
                    "model_id": encoder.model_id,
                    "mode": encoder.mode,
                    "dimension": encoder.dimension,
                    "rerank_window": 0,
                    "calibration_policy": None,
                    "real_model_connected": False,
                    "device": "cpu",
                    "precision": "float32",
                    "weights_bytes": 0,
                },
            )
        return httpx.Response(200, content=pack_features([feature]))

    monkeypatch.setattr(
        "falcon_service.inference.remote.httpx.Client",
        lambda **kwargs: client_class(transport=httpx.MockTransport(route), **kwargs),
    )
    remote = RemoteEncoder("http://runtime.local", SECRET)
    try:
        token = request_id_context.set("forwarded-123")
        try:
            assert remote.encode(image_bytes).model_id == encoder.model_id
        finally:
            request_id_context.reset(token)
        assert remote.encode(image_bytes).model_id == encoder.model_id
    finally:
        remote.close()
    assert requests[1].headers["X-Request-ID"] == "forwarded-123"
    assert "X-Request-ID" not in requests[2].headers
    assert all(request.headers["Authorization"] == f"Bearer {SECRET}" for request in requests)


def test_worker_logs_failure_and_retry_without_payload_or_exception_message(app, events):
    with TestClient(app):
        state = app.state
        job = enqueue(state.db, "private-workspace", "export", {"search_id": SECRET})
        # A missing export_id triggers a genuine retryable infrastructure/programming failure.
        assert run_once(state.db, state.store, state.runtime, state.settings)
        with state.db.transaction() as session:
            saved = session.scalar(select(Job).where(Job.id == job.id))
            assert saved.status == "pending"
            assert saved.attempts == 1
    claimed, failed, retry = [event for event in events() if event.get("job_id") == job.id]
    assert claimed["message"] == "job_claimed"
    assert failed["message"] == "job_failed"
    assert failed["exception_type"] == "KeyError"
    assert failed["duration_ms"] >= 0
    assert retry["message"] == "job_retry_scheduled"
    assert retry["retry_in_seconds"] == 2
    assert all(event["attempt"] == 1 for event in [claimed, failed, retry])
    assert SECRET not in json.dumps(events())
