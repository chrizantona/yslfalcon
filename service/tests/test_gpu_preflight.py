"""The GPU deployment gate validates handoff artifacts without importing model code."""

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "gpu_preflight", Path(__file__).parents[1] / "scripts/gpu_preflight.py"
)
gpu_preflight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gpu_preflight)

RUNTIME_IMAGE = "registry.example/falcon-runtime@sha256:" + "a" * 64


def bundle(tmp_path, *, image=RUNTIME_IMAGE, model_id="falcon-model-v1"):
    root = tmp_path / "bundle"
    root.mkdir()
    weights = b"fixture weights"
    calibration = b'{"threshold": 1.25}'
    (root / "encoder.safetensors").write_bytes(weights)
    (root / "calibration.json").write_bytes(calibration)
    manifest = {
        "schema_version": 1,
        "model_id": model_id,
        "runtime_image": image,
        "total_weights_bytes": len(weights),
        "files": [
            {
                "path": "encoder.safetensors",
                "role": "weight",
                "bytes": len(weights),
                "sha256": hashlib.sha256(weights).hexdigest(),
            },
            {
                "path": "calibration.json",
                "role": "calibration",
                "bytes": len(calibration),
                "sha256": hashlib.sha256(calibration).hexdigest(),
            },
        ],
    }
    (root / "runtime-bundle.json").write_text(json.dumps(manifest))
    return root, manifest


def test_gpu_bundle_preflight_verifies_inventory_hashes_identity_and_weight_total(tmp_path):
    root, manifest = bundle(tmp_path)
    report = gpu_preflight.validate_bundle(root, manifest["runtime_image"], manifest["model_id"])
    assert report == {
        "model_id": manifest["model_id"],
        "runtime_image": manifest["runtime_image"],
        "file_count": 2,
        "weights_bytes": len(b"fixture weights"),
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda root, manifest: manifest.update(runtime_image="falcon-runtime:release-123"),
            "@sha256",
        ),
        (lambda root, manifest: manifest.update(model_id="other-model"), "EXPECTED_MODEL_ID"),
        (
            lambda root, manifest: (root / "encoder.safetensors").write_bytes(b"tampered"),
            "Size mismatch",
        ),
        (lambda root, manifest: (root / "unlisted.bin").write_bytes(b"extra"), "inventory"),
    ],
)
def test_gpu_bundle_preflight_fails_closed(tmp_path, mutation, message):
    root, manifest = bundle(tmp_path)
    mutation(root, manifest)
    (root / "runtime-bundle.json").write_text(json.dumps(manifest))
    with pytest.raises(gpu_preflight.PreflightError, match=message):
        gpu_preflight.validate_bundle(root, RUNTIME_IMAGE, "falcon-model-v1")


def test_gpu_bundle_preflight_rejects_the_weight_limit(tmp_path, monkeypatch):
    root, manifest = bundle(tmp_path)
    monkeypatch.setattr(gpu_preflight, "MAX_INFERENCE_WEIGHTS_BYTES", 1)
    with pytest.raises(gpu_preflight.PreflightError, match="2 GB"):
        gpu_preflight.validate_bundle(root, manifest["runtime_image"], manifest["model_id"])


def test_gpu_bundle_preflight_rejects_a_symlinked_manifest(tmp_path):
    root, manifest = bundle(tmp_path)
    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps(manifest))
    (root / "runtime-bundle.json").unlink()
    (root / "runtime-bundle.json").symlink_to(outside)
    with pytest.raises(gpu_preflight.PreflightError, match="must not be a symlink"):
        gpu_preflight.validate_bundle(root, manifest["runtime_image"], manifest["model_id"])
