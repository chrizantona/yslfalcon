"""Paired bootstrap of the mAP gain between two experiments evaluated on the same fold (queries matched by image_id).

Each experiment dir provides per_query_ap_mean.csv (seed-averaged) or val_fold*/per_query_ap.csv (runs without seeds).
Usage: PYTHONPATH=src python -m reid.compare artifacts/EXP-000 artifacts/EXP-001
"""
import argparse
from pathlib import Path

import pandas as pd

from reid.metrics import paired_bootstrap


def load_ap(exp_dir):
    exp_dir = Path(exp_dir)
    candidates = [exp_dir / "per_query_ap_mean.csv", *sorted(exp_dir.glob("val_fold*/per_query_ap.csv"))]
    return pd.read_csv(next(p for p in candidates if p.exists()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("base")
    parser.add_argument("new")
    args = parser.parse_args()
    merged = load_ap(args.base).merge(load_ap(args.new)[["image_id", "AP"]], on="image_id", suffixes=("_base", "_new"))
    gain, low, high = paired_bootstrap(merged.AP_base.to_numpy(), merged.AP_new.to_numpy(), merged.vehicle_id.to_numpy())
    print(f"{Path(args.new).name} vs {Path(args.base).name}: mAP {merged.AP_base.mean():.4f} -> {merged.AP_new.mean():.4f}, "
          f"gain {gain:+.4f}, 95% CI [{low:+.4f}, {high:+.4f}], {len(merged)} queries")


if __name__ == "__main__":
    main()
