"""Validate deployment probes after Compose merges the supported image overlays."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

SERVICE_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "overlays",
    [
        (),
        ("docker-compose.registry.yml",),
        ("docker-compose.registry.yml", "docker-compose.model.yml"),
        ("docker-compose.gpu.yml",),
    ],
)
def test_compose_defines_probes_without_relying_on_image_metadata(overlays, tmp_path):
    if shutil.which("docker") is None:
        pytest.skip("Docker Compose CLI is unavailable")
    version = subprocess.run(["docker", "compose", "version"], capture_output=True, text=True)
    if version.returncode:
        pytest.skip("Docker Compose CLI is unavailable")

    # Never load a developer's .env or expose its credentials in assertion output.
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith(("FALCON_", "COMPOSE_"))
    }
    environment.update(
        FALCON_DB_PASSWORD="compose-validation-password",
        FALCON_AUTH_TOKENS="compose-validation-token",
        FALCON_RUNTIME_TOKEN="compose-validation-runtime-token",
        FALCON_API_IMAGE="example.invalid/falcon/api@sha256:" + "a" * 64,
        FALCON_WEB_IMAGE="example.invalid/falcon/web@sha256:" + "b" * 64,
        FALCON_RUNTIME_IMAGE="example.invalid/falcon/runtime@sha256:" + "c" * 64,
        FALCON_MODEL_BUNDLE_DIR=str(tmp_path),
        FALCON_EXPECTED_WEIGHTS_BYTES="1717977561",
    )
    command = ["docker", "compose", "--env-file", os.devnull, "-f", "docker-compose.yml"]
    for overlay in overlays:
        command.extend(["-f", overlay])
    rendered = subprocess.run(
        [*command, "config", "--format", "json"],
        cwd=SERVICE_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
    )
    services = json.loads(rendered.stdout)["services"]
    for service in services.values():
        probe = service["healthcheck"]
        assert not probe.get("disable")
        assert probe["test"][0] in {"CMD", "CMD-SHELL"}
        assert probe["timeout"]
        assert probe["retries"] > 0
        for dependency in service.get("depends_on", {}).values():
            assert dependency["condition"] == "service_healthy"

    for name, endpoint in {
        "api": "http://127.0.0.1:8000/api/v1/health/ready",
        "runtime": "http://127.0.0.1:8001/health/live",
    }.items():
        probe = services[name]["healthcheck"]["test"]
        assert probe[:3] == ["CMD", "python", "-c"]
        assert endpoint in probe[3]
        compile(probe[3], f"{name} healthcheck", "exec")
    assert services["worker"]["healthcheck"]["test"] == [
        "CMD",
        "python",
        "-m",
        "falcon_service.worker",
        "--healthcheck",
    ]
    assert services["web"]["healthcheck"]["test"][-1] == "http://127.0.0.1:8080/health"
    assert services["web"]["healthcheck"]["test"][:2] == ["CMD", "wget"]
    if {"docker-compose.model.yml", "docker-compose.gpu.yml"}.intersection(overlays):
        assert services["runtime"]["healthcheck"]["start_period"] == "3m0s"
