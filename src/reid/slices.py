"""mAP by slice for one experiment: shows where a gain comes from, not just the average.

Slices follow the EDA: is the query's viewpoint present in another camera's gallery frame, cameras 89/90, smallest
boxes (bottom quartile of the fold), boxes truncated by the frame border, taxi / carsharing liveries.

Usage: PYTHONPATH=src python -m reid.slices --exp-dir artifacts/EXP-003 --val-config configs/val_v1.yaml \
           --attributes protocol/attributes.csv
Writes <exp-dir>/slices.csv.
"""
import argparse
from pathlib import Path

import pandas as pd

from reid.compare import load_ap
from reid.data import load_yaml

DOMINANT_CAMERAS = [89, 90]
SLICES = ["view_in_other_cam", "cams_89_90", "small_box", "truncated", "taxi_livery"]


def slice_table(ap_df, gallery, attributes):
    """ap_df: per-query image_id, vehicle_id, camera_id, AP. Returns slice / group / n_queries / mAP."""
    queries = ap_df.merge(attributes, on="image_id", how="left")
    gallery = gallery.merge(attributes[["image_id", "view"]], on="image_id", how="left")

    views = {}
    for row in gallery.itertuples():
        views.setdefault(row.vehicle_id, []).append((row.camera_id, row.view))
    queries["view_in_other_cam"] = [any(view == q.view and camera != q.camera_id
                                        for camera, view in views.get(q.vehicle_id, []))
                                    for q in queries.itertuples()]
    queries["cams_89_90"] = queries.camera_id.isin(DOMINANT_CAMERAS)
    queries["small_box"] = queries.area_k <= queries.area_k.quantile(0.25)
    queries["taxi_livery"] = queries.taxi.eq("taxi")

    rows = []
    for name in SLICES:
        for value, group in queries.groupby(name):
            rows.append({"slice": name, "group": bool(value), "n_queries": int(group.AP.notna().sum()),
                         "mAP": group.AP.mean()})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-dir", required=True)
    parser.add_argument("--val-config", required=True)
    parser.add_argument("--attributes", default="protocol/attributes.csv")
    parser.add_argument("--fold", type=int)
    args = parser.parse_args()
    val_cfg = load_yaml(args.val_config)
    fold = args.fold if args.fold is not None else val_cfg["dev_fold"]

    gallery = pd.read_csv(Path(val_cfg["out_dir"]) / f"fold{fold}_gallery.csv")
    table = slice_table(load_ap(args.exp_dir), gallery, pd.read_csv(args.attributes))
    out = Path(args.exp_dir) / "slices.csv"
    table.to_csv(out, index=False)
    print(f"{Path(args.exp_dir).name}, fold {fold}")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
