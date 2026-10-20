"""Offline contract rehearsal with an explicitly uncalibrated preview descriptor."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from .inference import PreviewEncoder, rank_gallery
from .submission_check import validate_artifacts, write_manifest


def _records(path, image_root):
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not {"image_id", "x", "y", "w", "h"} <= set(reader.fieldnames or []):
            raise ValueError("Input CSV requires image_id,x,y,w,h columns")
        rows = list(reader)
    if not rows or len({row["image_id"] for row in rows}) != len(rows):
        raise ValueError("Input CSV must contain unique, nonempty image IDs")
    root = Path(image_root).resolve()
    result = []
    for row in rows:
        identity = row["image_id"]
        if not identity or Path(identity).name != identity or identity in {".", ".."}:
            raise ValueError("Image IDs must be plain filenames or filename stems")
        candidates = (
            [root / identity]
            if Path(identity).suffix
            else [root / f"{identity}{ext}" for ext in (".jpg", ".jpeg", ".png")]
        )
        present = [
            candidate
            for candidate in candidates
            if candidate.is_file() and candidate.resolve().parent == root
        ]
        if len(present) != 1:
            raise ValueError(f"Expected one local JPEG/PNG image for {identity}")
        box = tuple(float(row[key]) for key in ("x", "y", "w", "h"))
        result.append((identity, present[0], box))
    return result


def preview_export(query_csv, gallery_csv, images_dir, output):
    queries, gallery = _records(query_csv, images_dir), _records(gallery_csv, images_dir)
    if len(gallery) < 10:
        raise ValueError("The submission format requires at least 10 distinct gallery observations")
    encoder = PreviewEncoder()
    gallery_features = [encoder.encode(path.read_bytes(), box) for _, path, box in gallery]
    query_features, ranking = [], []
    for identity, path, box in queries:
        feature = encoder.encode(path.read_bytes(), box)
        query_features.append(feature)
        result = rank_gallery(feature, gallery_features, encoder, top_n=10)
        ranking.append([identity, *[gallery[item["index"]][0] for item in result["ranked"]]])
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    if any(
        (destination / name).exists()
        for name in (
            "submission.csv",
            "candidates.csv",
            "embeddings.npy",
            "preview.json",
            "artifact_manifest.json",
        )
    ):
        raise ValueError("Output already contains artifacts; use a new directory")
    with (destination / "submission.csv").open("w", newline="") as stream:
        csv.writer(stream).writerows(ranking)  # Official submission.csv has NO header.
    with (destination / "candidates.csv").open("w", newline="") as stream:
        csv.writer(stream).writerow(["query_id", "gallery_id", "confidence"])
        # The preview descriptor cannot accept an identity: the candidates body stays empty.
    np.save(
        destination / "embeddings.npy",
        np.stack([item.embedding for item in [*query_features, *gallery_features]]).astype(np.float32),
        allow_pickle=False,
    )
    report = {
        "mode": "preview",
        "model_id": encoder.model_id,
        "queries": len(queries),
        "gallery": len(gallery),
        "decision": "calibration_unavailable",
        "submission_ready": False,
        "warning": "Contract rehearsal only. No trained vehicle model or calibration is connected.",
    }
    (destination / "preview.json").write_text(json.dumps(report, indent=2) + "\n")
    write_manifest(destination, Path(query_csv), Path(gallery_csv), encoder.model_id)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    export = subparsers.add_parser(
        "preview-export", help="Rehearse artifact formats; never a competition-ready submission"
    )
    export.add_argument("--query-csv", required=True)
    export.add_argument("--gallery-csv", required=True)
    export.add_argument("--images-dir", required=True)
    export.add_argument("--output", required=True)
    validate = subparsers.add_parser(
        "validate-artifacts", help="Check formats and integrity, never ReID quality"
    )
    validate.add_argument("--query-csv", required=True)
    validate.add_argument("--gallery-csv", required=True)
    validate.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        if args.command == "preview-export":
            report = preview_export(args.query_csv, args.gallery_csv, args.images_dir, args.output)
        else:
            report = validate_artifacts(args.query_csv, args.gallery_csv, args.output)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
