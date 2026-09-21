"""Whole-dataset import through public, restartable and bounded upload requests."""

from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from test_backend import (
    HEADERS,
    OTHER_HEADERS,
    activate,
    make_gallery,
    picture,
    search,
    work,
)
from test_backend import client as client
from test_backend import settings as settings

from falcon_service.app import create_app
from falcon_service.models import FeatureArtifact, GalleryVersion, Job, Observation


def manifest(count=2):
    return "image_id,x,y,w,h\n" + "".join(f"car-{number},0,0,48,32\n" for number in range(count))


def start(client, gallery, csv=None, key="gallery-upload"):
    response = client.post(
        f"/api/v1/galleries/{gallery}/import-sessions",
        headers={**HEADERS, "Idempotency-Key": key},
        json={"manifest_csv": csv or manifest()},
    )
    assert response.status_code == 201, response.text
    return response.json()


def base(gallery, identifier):
    return f"/api/v1/galleries/{gallery}/import-sessions/{identifier}"


def files(numbers):
    return [("files", (f"car-{number}.png", picture(), "image/png")) for number in numbers]


def test_full_750_gallery_resumes_after_restart_and_finalizes_once(client, settings):
    gallery = make_gallery(client)
    opened = start(client, gallery, manifest(750))
    url = base(gallery, opened["id"])
    assert opened["expected_count"] == 750
    assert opened["limits"]["max_chunk_files"] == 100
    assert opened["validation"]["ready"] is False
    assert client.post(url + "/files", headers=HEADERS, files=files(range(100))).status_code == 200
    assert client.post(url + "/finalize", headers=HEADERS).status_code == 409
    assert client.get(f"/api/v1/galleries/{gallery}/versions", headers=HEADERS).json()["items"] == []
    with TestClient(create_app(settings)) as restarted:
        listing = restarted.get(f"/api/v1/galleries/{gallery}/import-sessions", headers=HEADERS).json()
        assert listing["items"][0]["id"] == opened["id"]
        state = restarted.get(url, headers=HEADERS).json()
        assert state["received_image_ids"] == [f"car-{number}" for number in range(100)]
        for offset in range(100, 750, 100):
            chunk = restarted.post(
                url + "/files", headers=HEADERS, files=files(range(offset, min(offset + 100, 750)))
            )
            assert chunk.status_code == 200, chunk.text
        # Repeating a lost chunk acknowledgement must not duplicate any observation.
        repeated = restarted.post(url + "/files", headers=HEADERS, files=files(range(100))).json()
        assert repeated["uploaded_count"] == 750
        assert repeated["missing_image_ids"] == []
        assert repeated["validation"] == {"ready": True, "issues": []}
        first = restarted.post(url + "/finalize", headers=HEADERS)
        second = restarted.post(url + "/finalize", headers=HEADERS)
        assert first.status_code == second.status_code == 202
        assert first.json()["id"] == second.json()["id"]
        work(restarted)
        job = restarted.get(f"/api/v1/jobs/{first.json()['id']}", headers=HEADERS).json()
        assert job["status"] == "completed", job
        assert job["result"]["count"] == 750
        assert (
            restarted.get(f"/api/v1/galleries/{gallery}", headers=HEADERS).json()["active_version_id"] is None
        )
        activate(restarted, gallery, job["result"]["version_id"])
        result = search(restarted, gallery)
        assert result.status_code == 200, result.text
        assert len(result.json()["ranked_candidates"]) == 10
        assert result.json()["accepted_candidates"] == []
        with restarted.app.state.db.transaction() as session:
            assert session.scalar(select(func.count()).select_from(GalleryVersion)) == 1
            assert session.scalar(select(func.count()).select_from(Observation)) == 750
            assert session.scalar(select(func.count()).select_from(FeatureArtifact)) == 750
            assert session.scalar(select(func.count()).select_from(Job)) == 1


def test_chunk_failure_is_atomic_and_duplicate_conflicts_do_not_replace_files(client):
    gallery = make_gallery(client)
    opened = start(client, gallery)
    url = base(gallery, opened["id"])
    bad = files([0]) + [("files", ("car-1.png", b"broken", "image/png"))]
    response = client.post(url + "/files", headers=HEADERS, files=bad)
    assert response.status_code == 400
    assert client.get(url, headers=HEADERS).json()["uploaded_count"] == 0
    assert client.post(url + "/files", headers=HEADERS, files=files([0])).status_code == 200
    conflict = [("files", ("car-0.png", picture("blue"), "image/png")), *files([1])]
    assert client.post(url + "/files", headers=HEADERS, files=conflict).status_code == 409
    state = client.get(url, headers=HEADERS).json()
    assert state["received_image_ids"] == ["car-0"]
    assert state["missing_image_ids"] == ["car-1"]
    duplicate = [("files", ("car-0.jpg", picture(), "image/png"))]
    assert client.post(url + "/files", headers=HEADERS, files=duplicate).status_code == 409
    assert client.post(url + "/files", headers=HEADERS, files=files([999])).status_code == 400
    assert client.get(url, headers=HEADERS).json()["uploaded_count"] == 1


def test_chunk_limits_idempotent_creation_and_workspace_isolation(client):
    gallery = make_gallery(client)
    opened = start(client, gallery, manifest(101))
    assert start(client, gallery, manifest(101))["id"] == opened["id"]
    response = client.post(
        f"/api/v1/galleries/{gallery}/import-sessions",
        headers={**HEADERS, "Idempotency-Key": "gallery-upload"},
        json={"manifest_csv": manifest(102)},
    )
    assert response.status_code == 409
    url = base(gallery, opened["id"])
    assert client.post(url + "/files", headers=HEADERS, files=files(range(101))).status_code == 413
    assert client.get(url, headers=HEADERS).json()["uploaded_count"] == 0
    assert client.get(url, headers=OTHER_HEADERS).status_code == 404
    assert client.post(url + "/files", headers=OTHER_HEADERS, files=files([0])).status_code == 404
    assert client.post(url + "/finalize", headers=OTHER_HEADERS).status_code == 404
    another_gallery = make_gallery(client)
    assert client.get(base(another_gallery, opened["id"]), headers=HEADERS).status_code == 404


@pytest.mark.parametrize(
    "csv",
    [
        "image_id,x,y,w,h\n",
        "image_id,x,y,w,h\na,0,0,1,1\na,0,0,1,1\n",
        "image_id,x,y,w,h\na,0,0,1,1\na.JPG,0,0,1,1\n",
        "image_id,x,y,w,h\n../a,0,0,1,1\n",
        "image_id,x,y,w,h\na,NaN,0,1,1\n",
        "image_id,x,y,w,h\na,1e999999999,0,1,1\n",
        "image_id,x,y,w,h\na,-1e999999999,0,1,1\n",
        "image_id,x,y,w,h\na,0,0,-1,1\n",
        "image_id,x,y,w,h\na,0.1,0,1,1\n",
        "image_id,x,y,w,h\na,0,0,1\n",
    ],
)
def test_invalid_manifests_never_create_snapshot(client, csv):
    gallery = make_gallery(client)
    response = client.post(
        f"/api/v1/galleries/{gallery}/import-sessions",
        headers={**HEADERS, "Idempotency-Key": "bad"},
        json={"manifest_csv": csv},
    )
    assert response.status_code == 400, response.text
    assert client.get(f"/api/v1/galleries/{gallery}/import-sessions", headers=HEADERS).json()["items"] == []
    assert client.get(f"/api/v1/galleries/{gallery}/versions", headers=HEADERS).json()["items"] == []


def test_concurrent_finalize_publishes_only_one_snapshot(client):
    gallery = make_gallery(client)
    opened = start(client, gallery)
    url = base(gallery, opened["id"])
    assert client.post(url + "/files", headers=HEADERS, files=files([0, 1])).status_code == 200
    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(lambda _: client.post(url + "/finalize", headers=HEADERS), range(2)))
    assert all(response.status_code == 202 for response in responses)
    assert len({response.json()["id"] for response in responses}) == 1
    with client.app.state.db.transaction() as session:
        assert session.scalar(select(func.count()).select_from(GalleryVersion)) == 1
        assert session.scalar(select(func.count()).select_from(Job)) == 1


def test_case_insensitive_extensions_and_clipped_bbox_are_explicit(client):
    gallery = make_gallery(client)
    opened = start(client, gallery, "\ufeffimage_id,x,y,w,h\nCar, -4, -2, 52, 34\n")
    url = base(gallery, opened["id"])
    response = client.post(
        url + "/files", headers=HEADERS, files=[("files", ("Car.PNG", picture(), "image/png"))]
    )
    assert response.status_code == 200, response.text
    assert response.json()["received_image_ids"] == ["Car"]
    job = client.post(url + "/finalize", headers=HEADERS).json()
    work(client)
    completed = client.get(f"/api/v1/jobs/{job['id']}", headers=HEADERS).json()
    activate(client, gallery, completed["result"]["version_id"])
    candidate = search(client, gallery).json()["ranked_candidates"][0]
    assert candidate["bbox"] == [0, 0, 48, 32]
    assert candidate["bbox_source"] == "explicit"
    assert candidate["external_id"] == "Car"
