"""Behavioral integration checks against the real API, queue and file store."""

import io
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import event, func, select, update
from sqlalchemy.orm import Session

from falcon_service.app import create_app
from falcon_service.config import Settings
from falcon_service.db import now
from falcon_service.jobs import claim, run_once
from falcon_service.models import GalleryVersion, Job, Observation, Search

ADMIN = "admin-test-token-" + "a" * 24
OTHER = "other-test-token-" + "b" * 24
VIEWER = "viewer-test-token-" + "c" * 24
HEADERS = {"Authorization": f"Bearer {ADMIN}"}
OTHER_HEADERS = {"Authorization": f"Bearer {OTHER}"}


def picture(color="red"):
    buffer = io.BytesIO()
    Image.new("RGB", (48, 32), color).save(buffer, "PNG")
    return buffer.getvalue()


@pytest.fixture
def settings(tmp_path):
    return Settings(
        database_url=f"sqlite:///{tmp_path}/test.db",
        storage_root=tmp_path / "artifacts",
        engine_mode="preview",
        cookie_secure=False,
        allowed_origins=["http://testserver"],
        job_lease_seconds=5,
        auth_tokens={
            ADMIN: {"role": "admin", "workspace": "one"},
            OTHER: {"role": "admin", "workspace": "two"},
            VIEWER: {"role": "viewer", "workspace": "one"},
        },
    )


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings)) as client:
        yield client


def work(client):
    state = client.app.state
    assert run_once(state.db, state.store, state.runtime, state.settings)


def make_gallery(client):
    response = client.post("/api/v1/galleries", json={"name": "Test gallery"}, headers=HEADERS)
    assert response.status_code == 201, response.text
    return response.json()["id"]


def import_gallery(client, gallery_id, colors=("red", "blue")):
    response = client.post(
        f"/api/v1/galleries/{gallery_id}/imports",
        headers=HEADERS,
        files=[("files", (f"{color}.png", picture(color), "image/png")) for color in colors],
    )
    assert response.status_code == 202, response.text
    job_id = response.json()["id"]
    work(client)
    job = client.get(f"/api/v1/jobs/{job_id}", headers=HEADERS).json()
    assert job["status"] == "completed", job
    return job["result"]["version_id"], job_id


def activate(client, gallery_id, version_id, expected=None):
    response = client.post(
        f"/api/v1/galleries/{gallery_id}/versions/{version_id}/activate",
        json={"expected_active_version_id": expected},
        headers=HEADERS,
    )
    assert response.status_code == 200, response.text


def search(client, gallery_id, key=None, image=None):
    headers = dict(HEADERS)
    if key:
        headers["Idempotency-Key"] = key
    return client.post(
        "/api/v1/searches",
        headers=headers,
        data={"gallery_id": gallery_id, "bbox": "[0,0,48,32]", "top_n": "10"},
        files={"image": ("query.png", image or picture(), "image/png")},
    )


def ready_gallery(client):
    gallery_id = make_gallery(client)
    version_id, job_id = import_gallery(client, gallery_id)
    activate(client, gallery_id, version_id)
    return gallery_id, version_id, job_id


def test_search_review_explanation_and_all_exports(client):
    gallery, version, _ = ready_gallery(client)
    response = search(client, gallery)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["gallery_version_id"] == version
    assert result["decision"] == "calibration_unavailable"
    assert result["accepted_candidates"] == []
    assert all(not item["accepted"] for item in result["ranked_candidates"])
    assert result["query"]["bbox_source"] == "explicit"
    assert all(value >= 0 for value in result["timings"].values())
    candidate = result["ranked_candidates"][0]
    assert client.get(candidate["image_url"], headers=HEADERS).status_code == 200
    assert client.get(result["query"]["crop_url"], headers=HEADERS).headers["content-type"] == "image/png"
    review = client.post(
        f"/api/v1/searches/{result['id']}/reviews",
        headers=HEADERS,
        json={
            "observation_id": candidate["observation_id"],
            "verdict": "uncertain",
            "note": "Проверить ракурс",
        },
    )
    assert review.status_code == 201
    reread = client.get(f"/api/v1/searches/{result['id']}", headers=HEADERS).json()
    assert reread["ranked_candidates"] == result["ranked_candidates"]
    assert reread["reviews"][0]["note"] == "Проверить ракурс"
    explained = client.post(
        f"/api/v1/searches/{result['id']}/explanations",
        headers=HEADERS,
        json={"observation_id": candidate["observation_id"]},
    ).json()
    work(client)
    explanation = client.get(f"/api/v1/jobs/{explained['id']}", headers=HEADERS).json()
    assert explanation["status"] == "completed", explanation
    assert explanation["result"]["method"] == "occlusion_sensitivity"
    assert len(explanation["result"]["query_map"]) == 6
    for format in ("json", "csv", "pdf"):
        job = client.post(
            f"/api/v1/searches/{result['id']}/exports", headers=HEADERS, json={"format": format}
        ).json()
        work(client)
        completed = client.get(f"/api/v1/jobs/{job['id']}", headers=HEADERS).json()
        assert completed["status"] == "completed", completed
        download = client.get(completed["result"]["download_url"], headers=HEADERS)
        assert download.status_code == 200
        if format == "pdf":
            assert download.content.startswith(b"%PDF")
        elif format == "json":
            assert "PREVIEW" in download.json()["notice"]
            assert download.json()["search"]["reviews"][0]["verdict"] == "uncertain"
        else:
            assert "observation_id" in download.text


def test_idempotency_pins_snapshot_across_activation_and_restart(client, settings):
    gallery, first_version, _ = ready_gallery(client)
    original = search(client, gallery, key="repeat-me").json()
    second_version, _ = import_gallery(client, gallery, ("green",))
    activate(client, gallery, second_version, expected=first_version)
    assert search(client, gallery, key="repeat-me").json()["id"] == original["id"]
    assert search(client, gallery, key="repeat-me", image=picture("yellow")).status_code == 409
    with TestClient(create_app(settings)) as restarted:
        repeated = search(restarted, gallery, key="repeat-me").json()
        assert repeated["id"] == original["id"]
        assert repeated["gallery_version_id"] == first_version
        fresh = search(restarted, gallery).json()
        assert fresh["gallery_version_id"] == second_version
        replay = restarted.post(f"/api/v1/searches/{original['id']}/replays", headers=HEADERS).json()
        assert replay["gallery_version_id"] == first_version
        assert replay["replay_of"] == original["id"]
        assert replay["ranked_candidates"] == original["ranked_candidates"]


def test_workspace_isolation_covers_files_jobs_exports_and_replays(client):
    gallery, _, job_id = ready_gallery(client)
    result = search(client, gallery).json()
    export = client.post(
        f"/api/v1/searches/{result['id']}/exports", headers=HEADERS, json={"format": "json"}
    ).json()
    work(client)
    exported = client.get(f"/api/v1/jobs/{export['id']}", headers=HEADERS).json()
    paths = [
        f"/api/v1/galleries/{gallery}",
        f"/api/v1/galleries/{gallery}/versions",
        f"/api/v1/jobs/{job_id}",
        f"/api/v1/searches/{result['id']}",
        result["query"]["image_url"],
        result["query"]["crop_url"],
        result["ranked_candidates"][0]["image_url"],
        result["ranked_candidates"][0]["crop_url"],
        exported["result"]["download_url"],
    ]
    for path in paths:
        assert client.get(path, headers=OTHER_HEADERS).status_code == 404, path
    assert client.get("/api/v1/searches", headers=OTHER_HEADERS).json()["items"] == []
    assert client.post(f"/api/v1/searches/{result['id']}/replays", headers=OTHER_HEADERS).status_code == 404
    assert (
        client.post(
            f"/api/v1/searches/{result['id']}/reviews", headers=OTHER_HEADERS, json={"verdict": "match"}
        ).status_code
        == 404
    )


def test_browser_session_csrf_origin_revocation_and_viewer(client, settings):
    login = client.post(
        "/api/v1/auth/session", json={"token": ADMIN}, headers={"Origin": "http://testserver"}
    )
    assert login.status_code == 200
    assert "HttpOnly" in login.headers.get_list("set-cookie")[0]
    csrf = login.json()["csrf_token"]
    assert client.get("/api/v1/auth/me").json()["workspace"] == "one"
    assert client.post("/api/v1/galleries", json={"name": "Denied"}).status_code == 403
    assert (
        client.post(
            "/api/v1/galleries",
            json={"name": "Denied"},
            headers={"Origin": "https://evil.example", "X-CSRF-Token": csrf},
        ).status_code
        == 403
    )
    good = {"Origin": "http://testserver", "X-CSRF-Token": csrf}
    assert client.post("/api/v1/galleries", json={"name": "Allowed"}, headers=good).status_code == 201
    assert (
        client.post(
            "/api/v1/galleries", json={"name": "Denied"}, headers={"Authorization": f"Bearer {VIEWER}"}
        ).status_code
        == 403
    )
    with TestClient(create_app(settings.model_copy(update={"auth_tokens": {}}))) as revoked:
        revoked.cookies.update(client.cookies)
        assert revoked.get("/api/v1/auth/me").status_code == 401
    assert client.delete("/api/v1/auth/session", headers=good).status_code == 204
    assert client.get("/api/v1/auth/me").status_code == 401


def test_auth_errors_do_not_echo_secrets(client):
    assert client.get("/api/v1/galleries").status_code == 401
    malformed = client.post(
        "/api/v1/auth/session", json={"token": "тест"}, headers={"Origin": "http://testserver"}
    )
    assert malformed.status_code == 401
    secret = "secret-that-must-not-be-echoed"
    invalid = client.post(
        "/api/v1/auth/session", json={"token": {"password": secret}}, headers={"Origin": "http://testserver"}
    )
    assert invalid.status_code == 422
    assert secret not in invalid.text
    assert client.get("/api/v1/health/live").status_code == 200


def test_import_validates_every_file_before_snapshot_and_activation_is_explicit(client):
    gallery = make_gallery(client)
    invalid = client.post(
        f"/api/v1/galleries/{gallery}/imports",
        headers=HEADERS,
        files=[
            ("files", ("good.png", picture(), "image/png")),
            ("files", ("bad.png", b"invalid", "image/png")),
        ],
    )
    assert invalid.status_code == 400
    assert client.get(f"/api/v1/galleries/{gallery}/versions", headers=HEADERS).json()["items"] == []
    version, _ = import_gallery(client, gallery)
    assert search(client, gallery).json()["error"]["code"] == "gallery_not_ready"
    activate(client, gallery, version)
    second, _ = import_gallery(client, gallery, ("green",))
    stale = client.post(
        f"/api/v1/galleries/{gallery}/versions/{second}/activate",
        headers=HEADERS,
        json={"expected_active_version_id": None},
    )
    assert stale.status_code == 409
    assert client.get(f"/api/v1/galleries/{gallery}", headers=HEADERS).json()["active_version_id"] == version


def test_manifest_rejects_unknown_keys_and_marks_full_frame(client):
    gallery = make_gallery(client)
    url = f"/api/v1/galleries/{gallery}/imports"
    unknown = client.post(
        url,
        headers=HEADERS,
        files={"files": ("a.png", picture(), "image/png")},
        data={"manifest": json.dumps({"a.png": {"camera_id": 1}})},
    )
    assert unknown.status_code == 400
    explicit = client.post(
        url,
        headers=HEADERS,
        files={"files": ("a.png", picture(), "image/png")},
        data={"manifest": json.dumps({"a.png": {"bbox": [4, 3, 20, 15], "external_id": "source-1"}})},
    )
    assert explicit.status_code == 202
    work(client)
    completed = client.get(f"/api/v1/jobs/{explicit.json()['id']}", headers=HEADERS).json()
    activate(client, gallery, completed["result"]["version_id"])
    result = search(client, gallery).json()
    assert result["ranked_candidates"][0]["bbox"] == [4, 3, 20, 15]
    assert result["ranked_candidates"][0]["external_id"] == "source-1"


def test_durable_worker_recovers_expired_lease_and_does_not_duplicate_snapshot(client, settings):
    gallery = make_gallery(client)
    queued = client.post(
        f"/api/v1/galleries/{gallery}/imports",
        headers=HEADERS,
        files={"files": ("a.png", picture(), "image/png")},
    ).json()
    state = client.app.state
    claimed = claim(state.db, settings)
    assert claimed.id == queued["id"]
    with state.db.transaction() as session:
        session.execute(
            update(Job).where(Job.id == claimed.id).values(lease_until=now() - timedelta(seconds=1))
        )
    with TestClient(create_app(settings)) as restarted:
        work(restarted)
        done = restarted.get(f"/api/v1/jobs/{claimed.id}", headers=HEADERS).json()
        assert done["status"] == "completed"
        assert done["attempts"] == 2
        # Simulate interruption after snapshot publication but before queue ACK.
        with state.db.transaction() as session:
            session.execute(
                update(Job).where(Job.id == claimed.id).values(status="pending", result=None, attempts=0)
            )
        work(restarted)
    with state.db.transaction() as session:
        assert session.scalar(select(func.count()).select_from(Observation)) == 1
        assert (
            session.scalar(
                select(func.count()).select_from(GalleryVersion).where(GalleryVersion.status == "ready")
            )
            == 1
        )


def test_last_attempt_crash_is_marked_failed(client, settings):
    gallery = make_gallery(client)
    queued = client.post(
        f"/api/v1/galleries/{gallery}/imports",
        headers=HEADERS,
        files={"files": ("a.png", picture(), "image/png")},
    ).json()
    with client.app.state.db.transaction() as session:
        session.execute(
            update(Job)
            .where(Job.id == queued["id"])
            .values(
                status="running", attempts=settings.job_max_attempts, lease_until=now() - timedelta(seconds=1)
            )
        )
    state = client.app.state
    assert claim(state.db, settings) is None
    assert client.get(f"/api/v1/jobs/{queued['id']}", headers=HEADERS).json()["status"] == "failed"
    assert (
        client.get(f"/api/v1/galleries/{gallery}/versions", headers=HEADERS).json()["items"][0]["status"]
        == "failed"
    )


def test_empty_gallery_invalid_bbox_and_oversized_body_are_errors(client, settings):
    gallery = make_gallery(client)
    assert search(client, gallery).status_code == 409
    response = client.post(
        "/api/v1/searches",
        headers=HEADERS,
        data={"gallery_id": gallery, "bbox": "[1,2,-2,5]"},
        files={"image": ("query.png", picture(), "image/png")},
    )
    assert response.status_code == 400
    response = client.post(
        "/api/v1/galleries",
        headers={**HEADERS, "Content-Length": str(settings.max_request_bytes + 1)},
        json={"name": "x"},
    )
    assert response.status_code == 413


def test_incomplete_gallery_never_becomes_a_refusal(client):
    gallery, version, _ = ready_gallery(client)
    with client.app.state.db.transaction() as session:
        row = session.scalar(select(Observation).where(Observation.version_id == version))
        session.delete(row)
    response = search(client, gallery)
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "gallery_incomplete"
    with client.app.state.db.transaction() as session:
        assert (
            session.scalar(select(func.count()).select_from(Search).where(Search.state == "completed")) == 0
        )


def test_offline_docs_have_no_cdn_and_storage_paths_cannot_escape(client):
    docs = client.get("/docs")
    assert docs.status_code == 200
    assert "https://" not in docs.text and "<script" not in docs.text
    from falcon_service.errors import ServiceError

    with pytest.raises(ServiceError):
        client.app.state.store.path("../../etc/passwd")


def test_system_measurements_start_empty_and_are_workspace_scoped(client):
    status = client.get("/api/v1/system", headers=HEADERS).json()
    assert status["timings"]["sample_count"] == 0
    assert status["timings"]["total_ms"] is None
    gallery, _, _ = ready_gallery(client)
    search(client, gallery)
    assert client.get("/api/v1/system", headers=HEADERS).json()["timings"]["sample_count"] == 1
    assert client.get("/api/v1/system", headers=OTHER_HEADERS).json()["timings"]["sample_count"] == 0


@pytest.mark.parametrize("failure", [True, False])
def test_stale_worker_cannot_publish_or_clear_new_workers_claim(client, monkeypatch, failure):
    """Lease takeover between compute and publish must fence both error and success."""
    from falcon_service import jobs

    gallery = make_gallery(client)
    queued = client.post(
        f"/api/v1/galleries/{gallery}/imports",
        headers=HEADERS,
        files={"files": ("a.png", picture(), "image/png")},
    ).json()
    state = client.app.state
    original_encode = state.runtime.encoder.encode
    replacement = []

    def takeover(image, bbox):
        with state.db.transaction() as session:
            session.execute(
                update(Job).where(Job.id == queued["id"]).values(lease_until=now() - timedelta(seconds=1))
            )
        replacement.append(jobs.claim(state.db, state.settings))
        if failure:
            raise ValueError("Old worker failed after lease takeover")
        return original_encode(image, bbox)

    monkeypatch.setattr(state.runtime.encoder, "encode", takeover)
    work(client)
    with state.db.transaction() as session:
        row = session.get(Job, queued["id"])
        assert row.status == "running"
        assert row.claim_token == replacement[0].claim_token
        assert row.attempts == 2
        assert session.get(GalleryVersion, row.payload["version_id"]).status == "preparing"
        assert session.scalar(select(func.count()).select_from(Observation)) == 0


def test_search_reservation_survives_interruption_before_compute(client, settings, monkeypatch):
    from falcon_service.routes import searches

    gallery, first_version, _ = ready_gallery(client)
    execute = searches.execute_search

    def interrupted(*_args, **_kwargs):
        raise RuntimeError("Simulated API process interruption")

    monkeypatch.setattr(searches, "execute_search", interrupted)
    with pytest.raises(RuntimeError, match="Simulated"):
        search(client, gallery, key="reserved-before-crash")
    monkeypatch.setattr(searches, "execute_search", execute)
    second_version, _ = import_gallery(client, gallery, ("green",))
    activate(client, gallery, second_version, expected=first_version)
    with TestClient(create_app(settings)) as restarted:
        response = search(restarted, gallery, key="reserved-before-crash")
        assert response.status_code == 200, response.text
        assert response.json()["gallery_version_id"] == first_version


def test_activation_requires_explicit_compare_and_swap(client):
    gallery, version, _ = ready_gallery(client)
    for body in (None, {}):
        response = client.post(
            f"/api/v1/galleries/{gallery}/versions/{version}/activate", headers=HEADERS, json=body
        )
        assert response.status_code == 422


def test_streamed_oversized_multipart_returns_typed_413(settings):
    settings = settings.model_copy(update={"max_request_bytes": 1024})
    with TestClient(create_app(settings)) as client:
        boundary = "test-boundary"
        chunks = [
            f'--{boundary}\r\nContent-Disposition: form-data; name="image"; filename="x.png"\r\nContent-Type: image/png\r\n\r\n'.encode(),
            b"x" * 2048,
            f"\r\n--{boundary}--\r\n".encode(),
        ]
        response = client.post(
            "/api/v1/searches",
            headers={**HEADERS, "Content-Type": f"multipart/form-data; boundary={boundary}"},
            content=iter(chunks),
        )
        assert response.status_code == 413
        assert response.json()["error"]["code"] == "request_too_large"


def test_failure_finalization_holds_write_fence_until_commit(client, monkeypatch):
    """A contender cannot replace ownership between the failure read and flush."""
    from falcon_service import jobs

    gallery = make_gallery(client)
    queued = client.post(
        f"/api/v1/galleries/{gallery}/imports",
        headers=HEADERS,
        files={"files": ("a.png", picture(), "image/png")},
    ).json()
    state = client.app.state
    failed, loaded, release, contender_started = (threading.Event() for _ in range(4))

    def fail_import(*_args):
        failed.set()
        raise ValueError("Processing failed")

    def pause_after_read(session, instance):
        if (
            session.bind is state.db.engine
            and isinstance(instance, Job)
            and instance.id == queued["id"]
            and failed.is_set()
            and not loaded.is_set()
        ):
            loaded.set()
            assert release.wait(5)

    def contender():
        contender_started.set()
        with state.db.transaction() as session:
            session.execute(
                update(Job)
                .where(Job.id == queued["id"], Job.status == "running")
                .values(lease_until=now() - timedelta(seconds=1))
            )
        return claim(state.db, state.settings)

    monkeypatch.setattr(jobs, "import_snapshot", fail_import)
    event.listen(Session, "loaded_as_persistent", pause_after_read)
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            failing_worker = pool.submit(run_once, state.db, state.store, state.runtime, state.settings)
            assert loaded.wait(5)
            other_worker = pool.submit(contender)
            assert contender_started.wait(5)
            try:
                # The failure handler has read ownership; its transaction must
                # retain the write fence so this takeover cannot complete.
                with pytest.raises(TimeoutError):
                    other_worker.result(timeout=0.1)
            finally:
                release.set()
            assert failing_worker.result(timeout=5)
            assert other_worker.result(timeout=5) is None
    finally:
        release.set()
        event.remove(Session, "loaded_as_persistent", pause_after_read)


def test_pagination_is_complete_and_workspace_scoped(client):
    for _ in range(3):
        make_gallery(client)
    first = client.get("/api/v1/galleries?limit=2", headers=HEADERS).json()
    assert first["total"] == 3
    assert first["next_offset"] == 2
    second = client.get("/api/v1/galleries?limit=2&offset=2", headers=HEADERS).json()
    assert len(second["items"]) == 1
    assert second["next_offset"] is None
    assert {item["id"] for item in first["items"]}.isdisjoint({item["id"] for item in second["items"]})
    assert client.get("/api/v1/galleries?limit=2", headers=OTHER_HEADERS).json()["total"] == 0


def test_concurrent_idempotency_reserves_one_search(client, monkeypatch):
    gallery, _, _ = ready_gallery(client)
    barrier = threading.Barrier(2)
    store = client.app.state.store
    original_put = store.put

    def simultaneous_upload(data, category, extension):
        if category == "queries":
            barrier.wait(timeout=5)
        return original_put(data, category, extension)

    monkeypatch.setattr(store, "put", simultaneous_upload)
    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(lambda _: search(client, gallery, key="concurrent-key"), range(2)))
    assert all(response.status_code in {200, 409} for response in responses)
    monkeypatch.setattr(store, "put", original_put)
    completed = search(client, gallery, key="concurrent-key")
    assert completed.status_code == 200, completed.text
    identifiers = {response.json()["id"] for response in responses if response.status_code == 200}
    assert identifiers <= {completed.json()["id"]}
    with client.app.state.db.transaction() as session:
        assert session.scalar(select(func.count()).select_from(Search)) == 1


def test_openapi_declares_bearer_or_cookie_for_protected_operations(client):
    schema = client.get("/api/v1/openapi.json").json()
    requirements = [{"BearerAuth": []}, {"SessionCookie": []}]
    for method in ("get", "post"):
        assert schema["paths"]["/api/v1/galleries"][method].get("security") == requirements
    assert schema["components"]["securitySchemes"]["BearerAuth"]["scheme"] == "bearer"
    cookie = schema["components"]["securitySchemes"]["SessionCookie"]
    assert cookie["type"] == "apiKey"
    assert cookie["in"] == "cookie"
    assert cookie["name"] == "falcon_session"
    for path, method in (
        ("/api/v1/auth/session", "post"),
        ("/api/v1/health/live", "get"),
        ("/api/v1/health/ready", "get"),
    ):
        assert not schema["paths"][path][method].get("security")
