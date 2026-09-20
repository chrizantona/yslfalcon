"""Audit VehicleID against every LCT crop before external-data training.

The audit has two independent layers: exact encoded-file SHA256 matches and perceptual-hash near duplicates. It never
uses an id, a row order or a hash as a model feature; hashes are only an offline contamination check. Candidate near
duplicates must be inspected in the emitted montage before training starts.

Usage: PYTHONPATH=src python -m reid.vehicleid_audit --data-config configs/data.yaml \
           --source vehicleid_train --out artifacts/vehicleid_audit
"""
import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from reid.data import CSV_FILES, load_csv, load_yaml
from reid.external import load_table

BIT_COUNT = np.asarray([int(i).bit_count() for i in range(256)], dtype=np.uint8)


def perceptual_hash(path):
    """A deterministic 63-bit pHash (8x8 low-frequency DCT, DC omitted)."""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    small = cv2.resize(image, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    low = cv2.dct(small)[:8, :8].reshape(-1)[1:]
    bits = low > np.median(low)
    value = np.uint64(0)
    for bit, enabled in enumerate(bits):
        if enabled:
            value |= np.uint64(1) << np.uint64(bit)
    return value


def file_fingerprint(path):
    path = Path(path)
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return str(path), digest, perceptual_hash(path)


def fingerprints(paths, workers):
    cv2.setNumThreads(1)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(file_fingerprint, paths))


def hamming_matrix(left, right):
    """Pairwise Hamming distances for uint64 vectors without requiring a particular NumPy bit-count API."""
    xor = np.bitwise_xor(np.asarray(left, np.uint64)[:, None], np.asarray(right, np.uint64)[None, :])
    byte_view = xor.view(np.uint8).reshape(*xor.shape, 8)
    return BIT_COUNT[byte_view].sum(axis=2)


def near_duplicate_pairs(left, right, threshold=8, chunk_size=32):
    """Yield (left index, right index, Hamming distance) below the inclusive pHash threshold."""
    for start in range(0, len(left), chunk_size):
        distances = hamming_matrix(left[start:start + chunk_size], right)
        rows, columns = np.where(distances <= threshold)
        for row, column in zip(rows, columns):
            yield start + int(row), int(column), int(distances[row, column])


def _tile(path, title, size=(256, 192)):
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(path)
    canvas = np.full((size[1] + 28, size[0], 3), 245, np.uint8)
    scale = min(size[0] / image.shape[1], size[1] / image.shape[0])
    resized = cv2.resize(image, (max(1, round(image.shape[1] * scale)), max(1, round(image.shape[0] * scale))))
    y, x = (size[1] - resized.shape[0]) // 2, (size[0] - resized.shape[1]) // 2
    canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    cv2.putText(canvas, title[:38], (5, size[1] + 19), cv2.FONT_HERSHEY_SIMPLEX, .42, (20, 20, 20), 1,
                cv2.LINE_AA)
    return canvas


def save_montage(matches, out, limit=64):
    rows = []
    for match in matches.head(limit).itertuples():
        rows.append(np.concatenate([_tile(match.external_path, f"VehicleID {match.external_image_id}"),
                                    _tile(match.lct_path, f"LCT {match.lct_split}/{match.lct_image_id} d={match.hamming}")],
                                   axis=1))
    if rows:
        cv2.imwrite(str(out), np.concatenate(rows, axis=0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--source", default="vehicleid_train")
    parser.add_argument("--out", required=True)
    parser.add_argument("--phash-threshold", type=int, default=8)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    data, out = load_yaml(args.data_config), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    spec = data["external"][args.source]
    external = load_table(spec["dir"], spec.get("table", "train.csv"))
    external["path"] = external.image_id.map(lambda image_id: str(Path(spec["dir"]) / "images" / f"{image_id}.jpg"))
    lct = pd.concat([load_csv(data["dataset_dir"], split).assign(split=split) for split in CSV_FILES],
                    ignore_index=True).drop_duplicates("image_id")
    lct["path"] = lct.image_id.map(lambda image_id: str(Path(data["crops_dir"]) / f"{image_id}.jpg"))

    ext_fp = fingerprints(external.path, args.workers)
    lct_fp = fingerprints(lct.path, args.workers)
    ext_sha, ext_hash = [row[1] for row in ext_fp], np.asarray([row[2] for row in ext_fp], np.uint64)
    lct_sha, lct_hash = [row[1] for row in lct_fp], np.asarray([row[2] for row in lct_fp], np.uint64)

    lct_by_sha = {}
    for index, digest in enumerate(lct_sha):
        lct_by_sha.setdefault(digest, []).append(index)
    exact = [(i, j) for i, digest in enumerate(ext_sha) for j in lct_by_sha.get(digest, ())]
    near = list(near_duplicate_pairs(ext_hash, lct_hash, args.phash_threshold))
    records = [{"external_image_id": external.iloc[i].image_id, "external_vehicle_id": external.iloc[i].vehicle_id,
                "external_path": external.iloc[i].path, "lct_image_id": lct.iloc[j].image_id,
                "lct_split": lct.iloc[j]["split"], "lct_path": lct.iloc[j].path, "hamming": distance,
                "exact_sha256": ext_sha[i] == lct_sha[j]} for i, j, distance in near]
    matches = pd.DataFrame(records, columns=["external_image_id", "external_vehicle_id", "external_path",
                                             "lct_image_id", "lct_split", "lct_path", "hamming", "exact_sha256"])
    matches.to_csv(out / "near_duplicates.csv", index=False)
    save_montage(matches, out / "near_duplicates.jpg")
    report = {"source": args.source, "external_images": len(external), "external_vehicles": external.vehicle_id.nunique(),
              "lct_crops": len(lct), "exact_sha256_pairs": len(exact), "phash_threshold": args.phash_threshold,
              "phash_candidate_pairs": len(matches), "requires_manual_review": bool(len(matches))}
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
