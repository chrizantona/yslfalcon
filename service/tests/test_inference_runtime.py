import csv
import json
from io import BytesIO

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from falcon_service.cli import preview_export
from falcon_service.config import Settings
from falcon_service.inference import PreviewEncoder, RemoteEncoder
from falcon_service.inference.wire import pack_features, unpack_features
from falcon_service.model_runtime import create_app
from falcon_service.runtime import EncoderRuntime
from falcon_service.runtime_schemas import RuntimeCapabilities

TOKEN = "fixture-runtime-token-32-characters"


def image_bytes(color=(255, 0, 0)):
    output = BytesIO()
    Image.new("RGB", (24, 24), color).save(output, "PNG")
    return output.getvalue()


def test_runtime_auth_binary_output_and_bad_image():
    with TestClient(create_app(PreviewEncoder(), TOKEN)) as client:
        assert client.get("/internal/v1/capabilities").status_code == 401
        assert (
            client.get(
                "/internal/v1/capabilities", headers={b"Authorization": b"Bearer \xc3\xa9"}
            ).status_code
            == 401
        )
        headers = {"Authorization": f"Bearer {TOKEN}"}
        capabilities = client.get("/internal/v1/capabilities", headers=headers).json()
        assert capabilities["mode"] == "preview"
        assert capabilities["real_model_connected"] is False
        assert capabilities["device"] == "cpu"
        assert capabilities["precision"] == "float32"
        assert capabilities["weights_bytes"] == 0
        response = client.post(
            "/internal/v1/encode", files={"image": ("image.png", image_bytes())}, headers=headers
        )
        assert response.status_code == 200
        features, _ = unpack_features(response.content)
        assert features[0].model_id == capabilities["model_id"]
        response = client.post("/internal/v1/encode", files={"image": ("bad.png", b"bad")}, headers=headers)
        assert response.status_code == 422


def test_runtime_refuses_missing_secret(monkeypatch):
    monkeypatch.delenv("FALCON_RUNTIME_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="24 characters"):
        with TestClient(create_app()):
            pass


def test_runtime_rejects_a_weight_inventory_mismatch():
    runtime = EncoderRuntime(Settings(expected_weights_bytes=1), PreviewEncoder())
    runtime.initialize()

    assert runtime.encoder is None
    assert "FALCON_EXPECTED_WEIGHTS_BYTES" in runtime.error


@pytest.mark.parametrize(
    ("mode", "connected", "weights"),
    [("preview", True, 1), ("production", False, 0), ("production", True, 0)],
)
def test_runtime_capabilities_reject_inconsistent_model_claims(mode, connected, weights):
    with pytest.raises(ValueError):
        RuntimeCapabilities(
            mode=mode,
            model_id="fixture",
            dimension=192,
            rerank_window=0,
            calibration_policy=None,
            real_model_connected=connected,
            device="cuda:0" if connected else "cpu",
            precision="float16" if connected else "float32",
            weights_bytes=weights,
        )


def test_remote_adapter_preserves_binary_features_and_auth(monkeypatch):
    encoder = PreviewEncoder()
    feature = encoder.encode(image_bytes())
    client_class = httpx.Client

    def route(request):
        assert request.headers["Authorization"] == f"Bearer {TOKEN}"
        if request.url.path.endswith("capabilities"):
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
        assert request.url.path.endswith("encode")
        assert b"image/png" not in request.content or b"upload" in request.content
        return httpx.Response(200, content=pack_features([feature]))

    def factory(**kwargs):
        return client_class(transport=httpx.MockTransport(route), **kwargs)

    monkeypatch.setattr("falcon_service.inference.remote.httpx.Client", factory)
    remote = RemoteEncoder("http://runtime.local", TOKEN)
    try:
        received = remote.encode(image_bytes())
        np.testing.assert_array_equal(received.embedding, feature.embedding)
        assert received.model_id == feature.model_id
        assert remote.mode == "preview"
        assert remote.real_model_connected is False
    finally:
        remote.close()


def test_offline_export_has_correct_order_headers_and_explicit_preview_marker(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    encoder = PreviewEncoder()
    for index in range(12):
        (images / f"image{index}.png").write_bytes(image_bytes((index * 20, 0, 255 - index * 20)))

    def write_csv(name, indices):
        target = tmp_path / name
        with target.open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["image_id", "x", "y", "w", "h"])
            writer.writerows([[f"image{i}", 0, 0, 24, 24] for i in indices])
        return target

    query_csv, gallery_csv = write_csv("queries.csv", [11, 10]), write_csv("gallery.csv", range(10))
    output = tmp_path / "out"
    report = preview_export(query_csv, gallery_csv, images, output)
    assert report["submission_ready"] is False
    with (output / "submission.csv").open() as stream:
        rows = list(csv.reader(stream))
    assert [row[0] for row in rows] == ["image11", "image10"]
    assert all(len(row) == 11 and len(set(row[1:])) == 10 for row in rows)
    assert (output / "candidates.csv").read_text().strip() == "query_id,gallery_id,confidence"
    features = np.load(output / "embeddings.npy", allow_pickle=False)
    assert features.dtype == np.float32
    assert features.shape == (12, 192)
    np.testing.assert_allclose(features[0], encoder.encode((images / "image11.png").read_bytes()).embedding)
    np.testing.assert_allclose(features[2], encoder.encode((images / "image0.png").read_bytes()).embedding)
    assert json.loads((output / "preview.json").read_text())["mode"] == "preview"
