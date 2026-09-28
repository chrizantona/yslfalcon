"""Relational vectors survive cache loss and migrate legacy snapshots without ML."""

import io
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from fastapi.testclient import TestClient
from sqlalchemy import delete, func, select, update
from test_backend import HEADERS, activate, import_gallery, make_gallery, picture, ready_gallery, search, work
from test_backend import client as client
from test_backend import settings as settings

from falcon_service.app import create_app
from falcon_service.db import now
from falcon_service.features import FeatureStore
from falcon_service.indexes import GalleryIndexStore
from falcon_service.models import (
    FeatureArtifact,
    Gallery,
    GalleryIndexArtifact,
    GalleryVersion,
    Job,
    Observation,
)


def test_database_is_authoritative_after_feature_cache_removal_and_restart(client, settings):
    gallery, version, _ = ready_gallery(client)
    first = search(client, gallery).json()
    state = client.app.state
    with state.db.transaction() as session:
        artifacts = list(session.scalars(select(FeatureArtifact)))
        indexes = list(session.scalars(select(GalleryIndexArtifact)))
        assert len(artifacts) == 2
        assert len(indexes) == 1
        assert indexes[0].algorithm == "faiss-hnsw-flat-ip-v1"
        assert all(item.model_id == first["model_id"] and item.dtype == "float32" for item in artifacts)
        assert all(item.dimension == 192 and item.payload for item in artifacts)
    for path in (state.store.root / "features").glob("*.npz"):
        path.unlink()
    with TestClient(create_app(settings)) as restarted:
        fresh = search(restarted, gallery)
        assert fresh.status_code == 200, fresh.text
        assert fresh.json()["ranked_candidates"] == first["ranked_candidates"]
        replay = restarted.post(f"/api/v1/searches/{first['id']}/replays", headers=HEADERS)
        assert replay.status_code == 200, replay.text
        assert replay.json()["gallery_version_id"] == version
        assert replay.json()["ranked_candidates"] == first["ranked_candidates"]
        exported = restarted.post(
            f"/api/v1/searches/{first['id']}/exports", headers=HEADERS, json={"format": "json"}
        ).json()
        work(restarted)
        job = restarted.get(f"/api/v1/jobs/{exported['id']}", headers=HEADERS).json()
        assert job["status"] == "completed", job
        assert restarted.get(job["result"]["download_url"], headers=HEADERS).status_code == 200


def test_additive_schema_and_legacy_feature_files_backfill_on_first_search(client):
    gallery, _, _ = ready_gallery(client)
    state = client.app.state
    with state.db.transaction() as session:
        observations = list(session.scalars(select(Observation)))
    features = FeatureStore(state.db, state.store).read_many(
        "one", observations, state.runtime.require().model_id
    )
    with state.db.transaction() as session:
        for observation, feature in zip(observations, features, strict=True):
            legacy_key = state.store.put_feature(feature)
            session.execute(
                update(Observation).where(Observation.id == observation.id).values(feature_key=legacy_key)
            )
        session.execute(delete(FeatureArtifact))
    # A pre-upgrade database has no new table at all. Startup creates it additively.
    GalleryIndexArtifact.__table__.drop(state.db.engine)
    FeatureArtifact.__table__.drop(state.db.engine)
    state.db.initialize()
    with state.db.transaction() as session:
        assert session.scalar(select(func.count()).select_from(FeatureArtifact)) == 0
        assert session.scalar(select(func.count()).select_from(GalleryIndexArtifact)) == 0
    migrated = search(client, gallery)
    assert migrated.status_code == 200, migrated.text
    with state.db.transaction() as session:
        assert session.scalar(select(func.count()).select_from(FeatureArtifact)) == 2
        assert session.scalar(select(func.count()).select_from(GalleryIndexArtifact)) == 1
    for path in (state.store.root / "features").glob("*.npz"):
        path.unlink()
    assert search(client, gallery).json()["ranked_candidates"] == migrated.json()["ranked_candidates"]


def test_corrupt_hnsw_artifact_is_rebuilt_from_authoritative_features(client):
    gallery, _, _ = ready_gallery(client)
    state = client.app.state
    with state.db.transaction() as session:
        original = session.scalar(select(GalleryIndexArtifact))
        original_key = original.artifact_key
    state.store.path(original_key).write_bytes(b"not a faiss index")
    response = search(client, gallery)
    assert response.status_code == 200, response.text
    with state.db.transaction() as session:
        rebuilt = session.scalar(select(GalleryIndexArtifact))
        assert rebuilt.artifact_key != original_key
        assert state.store.path(rebuilt.artifact_key).stat().st_size == rebuilt.size_bytes


def test_concurrent_hnsw_rebuilds_publish_one_winner_and_clean_losers(client):
    _, version_id, _ = ready_gallery(client)
    state = client.app.state
    with state.db.transaction() as session:
        previous = session.get(GalleryIndexArtifact, version_id)
        model_id, count = previous.model_id, previous.count
    stores = [GalleryIndexStore(state.db, state.store) for _ in range(2)]
    with ThreadPoolExecutor(max_workers=2) as executor:
        records = list(
            executor.map(
                lambda indexes: indexes._rebuild(previous, "one", version_id, model_id, count),
                stores,
            )
        )
    assert records[0].artifact_key == records[1].artifact_key
    with state.db.transaction() as session:
        winner = session.get(GalleryIndexArtifact, version_id)
        assert winner.artifact_key == records[0].artifact_key
    assert [path.name for path in (state.store.root / "indexes").glob("*.faiss")] == [
        state.store.path(winner.artifact_key).name
    ]


def test_corrupt_database_feature_never_falls_back_to_valid_legacy_cache(client):
    gallery, _, _ = ready_gallery(client)
    state = client.app.state
    with state.db.transaction() as session:
        observation = session.scalar(select(Observation))
        artifact = session.get(FeatureArtifact, observation.feature_key)
        feature = FeatureStore.decode(artifact, artifact.model_id)
        stream = io.BytesIO()
        np.savez(stream, embedding=feature.embedding, model_id=np.array(feature.model_id))
        path = state.store.path(observation.feature_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(stream.getvalue())
        artifact.payload = b"broken database artifact"
    response = search(client, gallery)
    assert response.status_code == 503, response.text
    assert response.json()["error"]["code"] == "feature_corrupt"


def test_failed_import_does_not_publish_partial_vectors_or_change_active_version(client, monkeypatch):
    gallery, version, _ = ready_gallery(client)
    state = client.app.state
    response = client.post(
        f"/api/v1/galleries/{gallery}/imports",
        headers=HEADERS,
        files=[("files", (f"{color}.png", picture(color), "image/png")) for color in ("yellow", "blue")],
    )
    assert response.status_code == 202
    encoder = state.runtime.require()
    original, calls = encoder.encode, []

    def fail_second(*args, **kwargs):
        calls.append(True)
        if len(calls) == 2:
            raise RuntimeError("temporary encoding failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(encoder, "encode", fail_second)
    work(client)
    with state.db.transaction() as session:
        assert session.scalar(select(func.count()).select_from(FeatureArtifact)) == 2
        assert session.scalar(select(func.count()).select_from(Observation)) == 2
        assert session.get(Gallery, gallery).active_version_id == version
        job = session.get(Job, response.json()["id"])
        assert job.status == "pending"
        assert session.get(GalleryVersion, job.payload["version_id"]).status == "preparing"
        job.available_at = now()
    monkeypatch.setattr(encoder, "encode", original)
    work(client)
    with state.db.transaction() as session:
        assert session.scalar(select(func.count()).select_from(FeatureArtifact)) == 4
        assert session.scalar(select(func.count()).select_from(Observation)) == 4
        assert session.get(Gallery, gallery).active_version_id == version


def test_feature_workspace_metadata_cannot_be_used_by_another_workspace(client):
    gallery = make_gallery(client)
    version, _ = import_gallery(client, gallery)
    activate(client, gallery, version)
    with client.app.state.db.transaction() as session:
        session.execute(update(FeatureArtifact).values(workspace="another"))
    response = search(client, gallery)
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "feature_unavailable"
