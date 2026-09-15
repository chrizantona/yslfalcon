"""Frozen validation protocol: vehicle-grouped folds and a per-fold pseudo-test that mirrors the test recipe.

Test recipe (reverse-engineered in eda/): gallery = one random frame per (vehicle, camera), query = all other frames.
Writes to <out_dir>: folds.csv, fold{k}_query.csv and fold{k}_gallery.csv (rows shuffled), fold{k}_no_match_vehicles.csv
(vehicles whose gallery frames are dropped when calibrating the refusal mode). Never overwrites an existing protocol.

Usage: PYTHONPATH=src python -m reid.protocol --data-config configs/data.yaml --config configs/val_v1.yaml
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from reid.data import load_csv, load_yaml

DOMINANT_CAMERAS = {89, 90}  # 21% of train images; balanced across folds


def seen_by_dominant_cameras(df):
    return df.groupby("vehicle_id").camera_id.agg(lambda cams: bool(DOMINANT_CAMERAS & set(cams)))


def assign_folds(train, n_folds, seed):
    """Fold per row: vehicles are shuffled within strata (seen by cameras 89/90 or not) and dealt round-robin."""
    stratum = seen_by_dominant_cameras(train)
    rng = np.random.default_rng(seed)
    fold_of = {}
    for _, vehicles in stratum.groupby(stratum):
        for k, vehicle in enumerate(rng.permutation(vehicles.index.to_numpy())):
            fold_of[vehicle] = k % n_folds
    return train.vehicle_id.map(fold_of)


def pseudo_test(fold_df, seed):
    gallery_idx = fold_df.groupby(["vehicle_id", "camera_id"]).sample(1, random_state=seed).index
    query = fold_df.drop(gallery_idx).sample(frac=1, random_state=seed)  # shuffled: row order must not leak identity
    gallery = fold_df.loc[gallery_idx].sample(frac=1, random_state=seed)
    return query, gallery


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    data_cfg, cfg = load_yaml(args.data_config), load_yaml(args.config)

    out = Path(cfg["out_dir"])
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} already exists: the protocol is frozen, create a new version instead")
    out.mkdir(parents=True, exist_ok=True)

    train = load_csv(data_cfg["dataset_dir"], "train")
    train["fold"] = assign_folds(train, cfg["n_folds"], cfg["split_seed"])
    train.to_csv(out / "folds.csv", index=False)

    rng = np.random.default_rng(cfg["no_match_seed"])
    summary = []
    for k in range(cfg["n_folds"]):
        fold_df = train[train.fold == k].drop(columns="fold")
        query, gallery = pseudo_test(fold_df, cfg["pseudo_test_seed"])
        query.to_csv(out / f"fold{k}_query.csv", index=False)
        gallery.to_csv(out / f"fold{k}_gallery.csv", index=False)
        vehicles = np.sort(fold_df.vehicle_id.unique())
        no_match = np.sort(rng.choice(vehicles, round(cfg["no_match_frac"] * len(vehicles)), replace=False))
        pd.DataFrame({"vehicle_id": no_match}).to_csv(out / f"fold{k}_no_match_vehicles.csv", index=False)
        summary.append({
            "fold": k,
            "vehicles": len(vehicles),
            "images": len(fold_df),
            "query": len(query),
            "gallery": len(gallery),
            "share_cams_89_90": round(seen_by_dominant_cameras(fold_df).mean(), 3),
            "no_match_vehicles": len(no_match),
        })
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
