"""Night diagnostic from saved official APs, using the fixed EDA brightness rule.

This only annotates development queries and averages their existing APs. It does
not run an encoder, change rankings, calibrate refusal, or use the holdout fold.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path

import cv2
import pandas as pd

from reid.compare import load_ap
from reid.research_compare import describe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-csv", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--exp-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    # Scope is deliberately development fold 0; do not inspect the holdout here.
    if args.query_csv.name != "fold0_query.csv" or args.query_csv.parent.name != "val_v2":
        raise ValueError("night diagnostic requires the frozen val_v2 development query CSV")
    cv2.setNumThreads(1)
    queries = pd.read_csv(args.query_csv)

    def measure(image_id):
        frame = cv2.imread(str(args.images_dir / f"{image_id}.jpg"))
        if frame is None:
            raise ValueError(f"cannot decode development frame {image_id}")
        small = cv2.resize(frame, (480, 270), interpolation=cv2.INTER_AREA)
        brightness = float(cv2.cvtColor(small, cv2.COLOR_BGR2HSV)[..., 2].mean())
        return {"image_id": image_id, "frame_brightness": brightness, "night": brightness < 60}

    with ThreadPoolExecutor(max_workers=8) as pool:
        attributes = pd.DataFrame(list(pool.map(measure, queries.image_id)))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    attributes_path = args.out_dir / "night_fold0.csv"
    attributes.to_csv(attributes_path, index=False)
    result = {
        "protocol": "val_v2", "fold": 0,
        "criterion": "Complete frame resized to 480x270 with INTER_AREA; mean HSV V < 60/255",
        "source": "eda/extract_features.py:80-88 and eda/KEY_FINDINGS.md",
        "posthoc_diagnostic_only": True, "used_for_model_or_threshold_selection": False,
        "holdout_fold4_used": False, "queries_measured": len(attributes),
        "night_queries": int(attributes.night.sum()), "experiments": {},
    }
    for directory in args.exp_dirs:
        metadata, expected_map, _ = describe(directory)
        if (metadata["protocol"], metadata["fold"]) != ("val_v2", 0):
            raise ValueError(f"{directory}: evaluation protocol differs")
        ap = load_ap(directory)
        measured_map = float(ap.AP.mean())
        if not math.isfinite(measured_map) or abs(measured_map - expected_map) > 1e-7:
            raise ValueError(f"{directory}: APs do not reproduce the recorded mAP@10")
        if set(ap.image_id) != set(attributes.image_id):
            raise ValueError(f"{directory}: query set differs from development fold 0")
        merged = ap.merge(attributes, on="image_id", validate="one_to_one")
        night = merged[merged.night]
        result["experiments"][directory.name] = {
            "night_valid_queries": int(night.AP.notna().sum()),
            "night_vehicles": int(night.vehicle_id.nunique()),
            "night_mAP10": float(night.AP.mean()),
            "day_mAP10": float(merged[~merged.night].AP.mean()),
            "all_mAP10": float(merged.AP.mean()),
        }
    with attributes_path.open("rb") as stream:
        result["attributes_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    (args.out_dir / "night_slices.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
