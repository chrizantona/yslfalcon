"""Top-1 confidence and correctness on every fold, from the EXP-202 run that never saw that fold, with and without the
query's own (vehicle, camera) frames in the gallery (EXP-209).

Both settings use the single gallery of the test (the no-pair vehicles removed, ~20% of queries without a pair):
  test  - as the test is built: one frame per (vehicle, camera), so the query's own camera is always represented;
  clean - that frame is removed for every query before ranking and before the z-score: only cross-camera evidence.
Pair and correctness as evaluate.py counts them: a pair is a cross-camera frame of the vehicle; the top candidate is
right whenever it is the same vehicle (in the test setting that includes the own-camera frame).
One npz per fold: has_pair, vehicle, conf_{test,clean}, right_{test,clean}; reid.threshold_mix reads them.

Usage: PYTHONPATH=src python -m reid.clean_gallery --data-config configs/data.yaml --val-config configs/val_v2.yaml \
           --out artifacts/EXP-209 [--folds 0 1 2 3 4]
"""
import argparse
import json
from pathlib import Path

import numpy as np

from reid.data import load_yaml
from reid.embed import pick_device
from reid.evaluate import load_fold
from reid.metrics import reid_eval_top10
from reid.official_check import realistic


def run_of(fold):
    """The {config, checkpoint} of the EXP-202 run trained on every fold but this one."""
    (spec,) = load_yaml(f"configs/exp202_calib_fold{fold}.yaml")["candidates"].values()
    return {"config": spec["config"], "checkpoint": spec["checkpoint"]}


def top1(sim, allowed):
    """Top-1 index and its z-score over the allowed gallery items of every query."""
    s = np.where(allowed, sim, np.nan)
    z = (s - np.nanmean(s, 1, keepdims=True)) / np.maximum(np.nanstd(s, 1, keepdims=True), 1e-12)
    z = np.where(allowed, z, -np.inf)
    return z.argmax(1), z.max(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--val-config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = parser.parse_args()
    data_cfg, val_cfg = load_yaml(args.data_config), load_yaml(args.val_config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = pick_device()
    for fold in args.folds:
        query, gallery_full, no_match = load_fold(Path(val_cfg["out_dir"]), fold)
        gallery = gallery_full[~gallery_full.vehicle_id.isin(no_match)].reset_index(drop=True)
        spec = run_of(fold)
        sim, confidence, _, _ = realistic(spec, query, gallery, data_cfg, device)
        q_pid, q_cam = query.vehicle_id.to_numpy(), query.camera_id.to_numpy()
        g_pid, g_cam = gallery.vehicle_id.to_numpy(), gallery.camera_id.to_numpy()
        same_vehicle = q_pid[:, None] == g_pid[None, :]
        own_camera = same_vehicle & (q_cam[:, None] == g_cam[None, :])
        has_pair = (same_vehicle & ~own_camera).any(1)
        top_test, conf_test = top1(sim, np.ones_like(own_camera))
        assert np.allclose(conf_test, confidence, atol=1e-4), "z-score differs from reid.official_check"
        top_clean, conf_clean = top1(sim, ~own_camera)
        top10, _ = reid_eval_top10(sim, q_pid, q_cam, g_pid, g_cam)
        np.savez_compressed(out / f"fold{fold}.npz", has_pair=has_pair, vehicle=q_pid,
                            conf_test=conf_test, right_test=g_pid[top_test] == q_pid,
                            conf_clean=conf_clean, right_clean=g_pid[top_clean] == q_pid)
        print(json.dumps({"fold": fold, "checkpoint": spec["checkpoint"], "queries": len(query),
                          "gallery": len(gallery), "with_pair": int(has_pair.sum()),
                          "mAP10": round(float(top10["mAP10"]), 6), "R1": round(float(top10["R1"]), 6)}), flush=True)


if __name__ == "__main__":
    main()
