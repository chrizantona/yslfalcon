"""The trained-model adapter: the viewpoint rerank contract and how the runtime selects the encoder.

The model itself needs PyTorch and a GPU, which only the runtime image has; everything checked here runs in
the ordinary service environment."""

import hashlib
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from falcon_service.inference import FeatureBundle, RepositoryEncoder, rank_gallery
from falcon_service.inference.encoders import view_rerank
from falcon_service.model_runtime import create_app

TOKEN = "fixture-runtime-token-32-characters"


def test_view_rerank_returns_a_permutation_of_the_incoming_scores():
    scores = np.array([0.91, 0.90, 0.72, 0.40], dtype=np.float32)
    result = view_rerank(scores, 0, np.array([0, 1, 0, 2]), 0.07)
    assert np.allclose(np.sort(result), np.sort(scores))


def test_a_same_view_candidate_drops_below_a_close_one_from_another_side():
    scores = np.array([0.91, 0.88], dtype=np.float32)  # gap 0.03 < delta
    result = view_rerank(scores, 1, np.array([1, 0]), 0.07)
    assert result[1] > result[0]
    far = np.array([0.95, 0.80], dtype=np.float32)  # gap 0.15 > delta: the order survives
    assert np.array_equal(view_rerank(far, 1, np.array([1, 0]), 0.07), far)


def test_a_candidate_without_a_view_is_never_penalised():
    scores = np.array([0.91, 0.88], dtype=np.float32)
    assert np.array_equal(view_rerank(scores, 1, np.array([-1, 0]), 0.07), scores)
    assert np.array_equal(view_rerank(scores, -1, np.array([1, 1]), 0.07), scores)


class ViewEncoder:
    """The rerank half of the trained adapter, over hand-made features."""

    mode, model_id, dimension, rerank_window = "production", "fixture-model", 2, 50
    calibration_policy, real_model_connected = None, True
    device, precision, weights_bytes = "cpu", "float32", 1

    def rerank(self, query, candidates, similarities):
        views = np.array([int(item.tokens.argmax()) for item in candidates])
        return view_rerank(similarities, int(query.tokens.argmax()), views, 0.07)


def feature(vector, view):
    tokens = np.zeros((1, 3), np.float32)
    tokens[0, view] = 1
    return FeatureBundle(np.asarray(vector, np.float32), tokens, "fixture-model")


def test_rank_gallery_accepts_the_viewpoint_rerank_and_reorders():
    query = feature([1.0, 0.0], view=1)
    gallery = [feature([0.99, 0.14], view=1), feature([0.97, 0.24], view=0), feature([0.0, 1.0], view=2)]
    result = rank_gallery(query, gallery, ViewEncoder(), top_n=3)
    assert [item["index"] for item in result["ranked"]] == [1, 0, 2]  # the same-side twin is no longer first
    assert result["decision"] == "calibration_unavailable"  # no policy on the fixture encoder


@pytest.mark.parametrize(
    ("environment", "message"),
    [({"FALCON_ENCODER": "production"}, "FALCON_MODEL_MANIFEST"), ({"FALCON_ENCODER": "turbo"}, "Unknown")],
)
def test_the_runtime_refuses_to_start_without_a_valid_model_choice(monkeypatch, environment, message):
    monkeypatch.delenv("FALCON_MODEL_MANIFEST", raising=False)
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    with pytest.raises(RuntimeError, match=message), TestClient(create_app(token=TOKEN)):
        pass


def test_the_preview_stays_the_default(monkeypatch):
    monkeypatch.delenv("FALCON_ENCODER", raising=False)
    with TestClient(create_app(token=TOKEN)) as client:
        capabilities = client.get("/internal/v1/capabilities", headers={"Authorization": f"Bearer {TOKEN}"})
        assert capabilities.json()["mode"] == "preview"


def bundle(tmp_path, content=b"weights", **override):
    (tmp_path / "encoder.pt").write_bytes(content)
    item = {
        "path": "encoder.pt",
        "role": "weight",
        "bytes": 7,
        "sha256": hashlib.sha256(b"weights").hexdigest(),
    }
    manifest = {
        "schema_version": 1,
        "model_id": "fixture",
        "runtime_image": "baked:/models",
        "total_weights_bytes": 7,
        "files": [{**item, **override}],
    }
    (tmp_path / "runtime-bundle.json").write_text(json.dumps(manifest))
    return tmp_path / "runtime-bundle.json"


@pytest.mark.parametrize(
    ("content", "override", "message"),
    [
        (b"weights", {}, None),
        (b"WEIGHTS", {}, "sha256"),
        (b"weight", {}, "wrong size"),
        (b"weights", {"path": "../encoder.pt"}, "plain file names"),
    ],
)
def test_the_bundle_is_verified_before_any_model_code_loads(tmp_path, content, override, message):
    path = bundle(tmp_path, content, **override)
    if message is None:
        manifest, root, weights = RepositoryEncoder.verified_bundle(path)
        assert (manifest["model_id"], root, weights) == ("fixture", tmp_path, 7)
    else:
        with pytest.raises(RuntimeError, match=message):
            RepositoryEncoder.verified_bundle(path)
