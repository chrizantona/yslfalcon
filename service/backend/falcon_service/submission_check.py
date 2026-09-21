"""Check delivery formats without evaluating identity, quality, or acceptance thresholds."""

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

ARTIFACTS = ("submission.csv", "candidates.csv", "embeddings.npy")


def file_digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def image_ids(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if "image_id" not in (reader.fieldnames or []):
            raise ValueError(f"{path.name}: image_id column is required")
        ids = [row["image_id"] for row in reader]
    if not ids or any(not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError(f"{path.name}: image IDs must be nonempty and unique")
    return ids


def write_manifest(destination: Path, query_csv: Path, gallery_csv: Path, model_id: str) -> None:
    """Record row provenance; hashes are integrity checks, not a signed attestation."""
    manifest = {
        "schema_version": 1,
        "model_id": model_id,
        "mode": "preview",
        "submission_ready": False,
        "input_sha256": {"query": file_digest(query_csv), "gallery": file_digest(gallery_csv)},
        "embedding_order": {"query": image_ids(query_csv), "gallery": image_ids(gallery_csv)},
        "files": {name: file_digest(destination / name) for name in ARTIFACTS},
    }
    (destination / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def validate_artifacts(query_csv, gallery_csv, output) -> dict:
    query_csv, gallery_csv, destination = Path(query_csv), Path(gallery_csv), Path(output)
    query_ids, gallery_ids = image_ids(query_csv), image_ids(gallery_csv)
    gallery_set, query_set = set(gallery_ids), set(query_ids)
    if len(gallery_set) < 10:
        raise ValueError("At least 10 gallery observations are required")
    with (destination / "submission.csv").open(newline="", encoding="utf-8") as stream:
        ranking = list(csv.reader(stream))
    if len(ranking) != len(query_ids) or [row[0] if row else None for row in ranking] != query_ids:
        raise ValueError(
            "submission.csv must contain every query exactly once in CSV order, without a header"
        )
    for row in ranking:
        if len(row) != 11 or len(set(row[1:])) != 10 or not set(row[1:]) <= gallery_set:
            raise ValueError(f"submission.csv requires 10 distinct known candidates for query {row[0]}")
    with (destination / "candidates.csv").open(newline="", encoding="utf-8") as stream:
        reader = csv.reader(stream)
        if next(reader, None) != ["query_id", "gallery_id", "confidence"]:
            raise ValueError("candidates.csv requires the query_id,gallery_id,confidence header")
        candidates = list(reader)
    seen = set()
    for row in candidates:
        if len(row) != 3 or row[0] not in query_set or row[1] not in gallery_set:
            raise ValueError("candidates.csv contains invalid or unknown IDs")
        pair = (row[0], row[1])
        if pair in seen or not math.isfinite(float(row[2])):
            raise ValueError("candidates.csv contains duplicate pairs or nonfinite confidence")
        seen.add(pair)
    try:
        embeddings = np.load(destination / "embeddings.npy", allow_pickle=False, mmap_mode="r")
    except (EOFError, OSError, ValueError) as exc:
        raise ValueError("embeddings.npy is missing, truncated, or not a safe numeric array") from exc
    if not isinstance(embeddings, np.ndarray):
        embeddings.close()
        raise ValueError("embeddings.npy must be a single numeric NPY array")
    if (
        embeddings.dtype != np.float32
        or embeddings.ndim != 2
        or embeddings.shape[0] != len(query_ids) + len(gallery_ids)
        or embeddings.shape[1] < 1
    ):
        raise ValueError(
            "embeddings.npy must be float32 with query+gallery rows and a fixed positive dimension"
        )
    # Bound temporary memory for a full gallery; never materialize a second full matrix.
    for start in range(0, len(embeddings), 8192):
        if not np.isfinite(embeddings[start : start + 8192]).all():
            raise ValueError("embeddings.npy contains nonfinite values")
    manifest_path = destination / "artifact_manifest.json"
    provenance = "not_supplied"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if not isinstance(manifest, dict):
            raise ValueError("Artifact manifest must be a JSON object")
        if manifest.get("embedding_order") != {"query": query_ids, "gallery": gallery_ids}:
            raise ValueError("Artifact manifest row order differs from the input CSVs")
        if manifest.get("input_sha256") != {
            "query": file_digest(query_csv),
            "gallery": file_digest(gallery_csv),
        }:
            raise ValueError("Artifact manifest refers to different input CSV contents")
        if manifest.get("files") != {name: file_digest(destination / name) for name in ARTIFACTS}:
            raise ValueError("Artifact integrity check failed")
        provenance = "manifest_consistent"
        if manifest.get("mode") == "preview" and candidates:
            raise ValueError("Preview artifacts must not contain accepted candidates")
    return {
        "format_valid": True,
        "queries": len(query_ids),
        "gallery": len(gallery_ids),
        "dimension": int(embeddings.shape[1]),
        "accepted_rows": len(candidates),
        "row_provenance": provenance,
        "model_quality_verified": False,
        "submission_ready": False,
        "warning": "Format checks do not verify model quality or prove the semantic order of embedding values.",
    }
