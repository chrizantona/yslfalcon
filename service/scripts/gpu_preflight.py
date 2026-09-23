"""Validate an immutable GPU runtime handoff without loading the model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path, PurePosixPath

MAX_INFERENCE_WEIGHTS_BYTES = 2_000_000_000
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
IMAGE_DIGEST = re.compile(r"[^\s]+@sha256:[0-9a-f]{64}\Z")
ROLES = {"weight", "calibration", "metadata"}


class PreflightError(ValueError):
    pass


def _pinned_image(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PreflightError("runtime_image must be a non-empty string")
    image = value.strip()
    if not IMAGE_DIGEST.fullmatch(image):
        raise PreflightError("runtime_image must use an immutable @sha256 digest")
    return image


def _integer(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise PreflightError(f"{name} must be a non-negative integer")
    return value


def _bundle_file(root: Path, relative: object) -> tuple[str, Path]:
    if not isinstance(relative, str) or not relative:
        raise PreflightError("Each file path must be a non-empty POSIX path")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or ".." in pure.parts or "\\" in relative:
        raise PreflightError(f"Unsafe bundle path: {relative}")
    candidate = root.joinpath(*pure.parts)
    if candidate.is_symlink():
        raise PreflightError(f"Bundle files must not be symlinks: {relative}")
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise PreflightError(f"Bundle file is missing: {relative}") from exc
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise PreflightError(f"Bundle path is not a regular file: {relative}")
    return pure.as_posix(), resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_bundle(root: Path, runtime_image: str, expected_model_id: str) -> dict[str, object]:
    try:
        root = root.resolve(strict=True)
    except FileNotFoundError as exc:
        raise PreflightError(f"Model bundle directory does not exist: {root}") from exc
    if not root.is_dir():
        raise PreflightError(f"Model bundle is not a directory: {root}")
    manifest_path = root / "runtime-bundle.json"
    if manifest_path.is_symlink():
        raise PreflightError("runtime-bundle.json must not be a symlink")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise PreflightError("runtime-bundle.json is missing") from exc
    except json.JSONDecodeError as exc:
        raise PreflightError("runtime-bundle.json is not valid JSON") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise PreflightError("runtime-bundle.json must use schema_version 1")
    manifest_image = _pinned_image(manifest.get("runtime_image"))
    supplied_image = _pinned_image(runtime_image)
    if supplied_image != manifest_image:
        raise PreflightError("FALCON_RUNTIME_IMAGE does not match runtime-bundle.json")
    model_id = manifest.get("model_id")
    if not isinstance(model_id, str) or not 1 <= len(model_id) <= 256:
        raise PreflightError("model_id must contain 1..256 characters")
    if expected_model_id != model_id:
        raise PreflightError("FALCON_EXPECTED_MODEL_ID does not match runtime-bundle.json")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise PreflightError("files must be a non-empty list")

    listed: set[str] = set()
    weights_bytes = 0
    weight_count = 0
    for item in files:
        if not isinstance(item, dict):
            raise PreflightError("Each files entry must be an object")
        relative, path = _bundle_file(root, item.get("path"))
        if relative in listed:
            raise PreflightError(f"Duplicate bundle path: {relative}")
        listed.add(relative)
        role = item.get("role")
        if role not in ROLES:
            raise PreflightError(f"Unknown role for {relative}: {role}")
        expected_size = _integer(item.get("bytes"), f"bytes for {relative}")
        if path.stat().st_size != expected_size:
            raise PreflightError(f"Size mismatch for {relative}")
        expected_hash = item.get("sha256")
        if not isinstance(expected_hash, str) or not SHA256.fullmatch(expected_hash):
            raise PreflightError(f"Invalid sha256 for {relative}")
        if _sha256(path) != expected_hash:
            raise PreflightError(f"SHA-256 mismatch for {relative}")
        if role == "weight":
            weight_count += 1
            weights_bytes += expected_size

    actual = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file() or path.is_symlink()
    } - {"runtime-bundle.json"}
    if actual != listed:
        missing = sorted(actual - listed)
        extra = sorted(listed - actual)
        raise PreflightError(f"Bundle inventory mismatch; unlisted={missing}, missing={extra}")
    if not weight_count or not weights_bytes:
        raise PreflightError("At least one non-empty weight file must be declared")
    declared = _integer(manifest.get("total_weights_bytes"), "total_weights_bytes")
    if declared != weights_bytes:
        raise PreflightError("total_weights_bytes does not match weight files")
    if weights_bytes > MAX_INFERENCE_WEIGHTS_BYTES:
        raise PreflightError("Inference weights exceed the 2 GB service gate")
    return {
        "model_id": model_id,
        "runtime_image": manifest_image,
        "file_count": len(files),
        "weights_bytes": weights_bytes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bundle",
        type=Path,
        default=os.environ.get("FALCON_MODEL_BUNDLE_DIR"),
        required=not os.environ.get("FALCON_MODEL_BUNDLE_DIR"),
    )
    parser.add_argument(
        "--runtime-image",
        default=os.environ.get("FALCON_RUNTIME_IMAGE"),
        required=not os.environ.get("FALCON_RUNTIME_IMAGE"),
    )
    parser.add_argument(
        "--expected-model-id",
        default=os.environ.get("FALCON_EXPECTED_MODEL_ID"),
        required=not os.environ.get("FALCON_EXPECTED_MODEL_ID"),
    )
    args = parser.parse_args()
    try:
        report = validate_bundle(args.bundle, args.runtime_image, args.expected_model_id)
    except PreflightError as exc:
        parser.error(str(exc))
    print(json.dumps({"status": "ready", **report}, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
