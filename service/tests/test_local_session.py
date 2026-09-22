"""The local UI opens without a secret form while preserving session and data boundaries."""

from datetime import timedelta

import pytest
from fastapi.testclient import TestClient

from falcon_service.app import create_app
from falcon_service.auth import digest
from falcon_service.config import Settings
from falcon_service.db import now
from falcon_service.models import BrowserSession

LOCAL = "local-test-credential-" + "a" * 32
OTHER = "other-test-credential-" + "b" * 32
ORIGIN = {"Origin": "http://localhost:8090"}


def application(tmp_path, **overrides):
    values = dict(
        database_url=f"sqlite:///{tmp_path}/local.db",
        storage_root=tmp_path / "artifacts",
        engine_mode="preview",
        local_auto_login=True,
        cookie_secure=False,
        allowed_origins=["http://localhost:8090"],
        auth_tokens={
            OTHER: {"role": "admin", "workspace": "another-organization"},
            LOCAL: {"role": "admin", "workspace": "local"},
        },
    )
    values.update(overrides)
    return create_app(Settings(**values))


def test_local_entry_creates_cookie_session_without_exposing_credential(tmp_path):
    with TestClient(application(tmp_path), base_url="http://localhost:8090") as client:
        assert client.get("/api/v1/auth/me").status_code == 401
        response = client.post("/api/v1/auth/local-session", headers=ORIGIN)
        assert response.status_code == 200, response.text
        assert response.json()["role"] == "admin"
        assert response.json()["workspace"] == "local"
        assert LOCAL not in response.text and OTHER not in response.text
        assert "HttpOnly" in response.headers["set-cookie"]
        assert client.get("/api/v1/galleries").status_code == 200
        # Automatic entry does not disable the existing mutation checks.
        assert client.post("/api/v1/galleries", json={"name": "No CSRF"}, headers=ORIGIN).status_code == 403
        saved = client.post(
            "/api/v1/galleries",
            json={"name": "Existing local archive"},
            headers={**ORIGIN, "X-CSRF-Token": response.json()["csrf_token"]},
        )
        assert saved.status_code == 201
        assert client.get("/api/v1/galleries").json()["items"][0]["id"] == saved.json()["id"]
        other = client.get("/api/v1/galleries", headers={"Authorization": f"Bearer {OTHER}"})
        assert other.json()["items"] == []


def test_expired_session_can_reopen_without_losing_existing_gallery(tmp_path):
    with TestClient(application(tmp_path), base_url="http://localhost:8090") as client:
        opened = client.post("/api/v1/auth/local-session", headers=ORIGIN).json()
        saved = client.post(
            "/api/v1/galleries",
            json={"name": "Preserved"},
            headers={**ORIGIN, "X-CSRF-Token": opened["csrf_token"]},
        ).json()
        old_cookie = client.cookies.get("falcon_session")
        with client.app.state.db.transaction() as session:
            row = session.get(BrowserSession, digest(old_cookie))
            row.expires_at = now() - timedelta(seconds=1)
        assert client.get("/api/v1/auth/me").status_code == 401
        reopened = client.post("/api/v1/auth/local-session", headers=ORIGIN)
        assert reopened.status_code == 200
        assert client.cookies.get("falcon_session") != old_cookie
        assert client.get("/api/v1/galleries").json()["items"][0]["id"] == saved["id"]


def test_local_entry_is_disabled_by_default_for_standalone_api(tmp_path):
    assert Settings().local_auto_login is False
    with TestClient(
        application(tmp_path, local_auto_login=False), base_url="http://localhost:8090"
    ) as client:
        response = client.post("/api/v1/auth/local-session", headers=ORIGIN)
        assert response.status_code == 404
        assert not client.cookies
        assert client.get("/api/v1/galleries").status_code == 401


@pytest.mark.parametrize("origin", [None, "http://attacker.example", "null"])
def test_local_entry_rejects_foreign_or_missing_origin(tmp_path, origin):
    with TestClient(application(tmp_path), base_url="http://localhost:8090") as client:
        headers = {"Origin": origin} if origin is not None else {}
        assert client.post("/api/v1/auth/local-session", headers=headers).status_code == 403
        assert not client.cookies


def test_an_allowed_network_origin_or_host_does_not_enable_local_access(tmp_path):
    app = application(tmp_path, allowed_origins=["http://localhost:8090", "http://service.example"])
    with TestClient(app, base_url="http://service.example") as client:
        assert client.post("/api/v1/auth/local-session", headers=ORIGIN).status_code == 403
    with TestClient(app, base_url="http://localhost:8090") as client:
        assert (
            client.post(
                "/api/v1/auth/local-session", headers={"Origin": "http://service.example"}
            ).status_code
            == 403
        )


def test_local_entry_never_falls_back_to_another_organizations_admin(tmp_path):
    app = application(tmp_path, auth_tokens={OTHER: {"role": "admin", "workspace": "another-organization"}})
    with TestClient(app, base_url="http://localhost:8090") as client:
        assert client.post("/api/v1/auth/local-session", headers=ORIGIN).status_code == 503
        assert not client.cookies
