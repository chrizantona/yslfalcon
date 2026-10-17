"""Refusal thresholds fitted the way the test is scored, and how much fitting them on one fold costs (EXP-178).

Three corrections over what our thresholds were fitted on until now:
  - F1 as the organisers' evaluate.py counts it: the top candidate is right whenever it is the same vehicle, junk
    (same vehicle and same camera) included;
  - one gallery, as the test is built: the no-pair vehicles are simply absent, so alpha-QE cannot lean on them;
  - an estimate of over-fitting: the plateau around the best threshold, and vehicle-grouped split-half validation
    inside the fold - fit on half the vehicles, score on the other half.
Our own reimplementation is checked first against the official numbers at the old thresholds.

This feature-branch CLI retains the legacy EXP-178 candidate comparison. The final EXP-202 leave-one-fold-out
calibration lives on main; the additive diagnostics here do not replace that recipe or its serving threshold.

Usage: PYTHONPATH=src python -m reid.calibrate_refusal --data-config configs/data.yaml \
           --val-config configs/val_v2.yaml --config configs/exp178_refusal_calibration.yaml
"""
import argparse
import json
from pathlib import Path

import numpy as np

from reid.data import load_yaml
from reid.embed import pick_device
from reid.evaluate import load_fold
from reid.metrics import official_refusal_diagnostics, reid_eval_top10
from reid.official_check import realistic


def per_query(similarity, confidence, query, gallery):
    """What the refusal score needs from every query: its confidence, whether a pair exists, whether the top is right."""
    q_pid, q_cam = query.vehicle_id.to_numpy(), query.camera_id.to_numpy()
    g_pid, g_cam = gallery.vehicle_id.to_numpy(), gallery.camera_id.to_numpy()
    top = np.argsort(-similarity, axis=1, kind="stable")[:, 0]
    has_pair = ((g_pid[None, :] == q_pid[:, None]) & (g_cam[None, :] != q_cam[:, None])).any(1)
    return confidence, has_pair, g_pid[top] == q_pid      # official: same vehicle is right, junk included


def score(conf, has_pair, right, threshold, rows=None):
    """0.7 F1 + 0.3 TNR at a threshold over a subset of queries, counted as evaluate.py counts them."""
    if rows is not None:
        conf, has_pair, right = conf[rows], has_pair[rows], right[rows]
    answered = conf >= threshold
    tp = (answered & has_pair & right).sum()
    fp = (answered & ~(has_pair & right)).sum()
    fn = (~answered & has_pair).sum()
    tn = (~answered & ~has_pair).sum()
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    tnr = tn / max((~has_pair).sum(), 1)
    return 0.7 * f1 + 0.3 * tnr, f1, tnr


def fit(conf, has_pair, right, rows=None):
    """Best threshold over every distinct confidence (the higher one on ties), with its score."""
    grid = np.unique(conf if rows is None else conf[rows])
    scores = np.array([score(conf, has_pair, right, t, rows)[0] for t in grid])
    best = np.flatnonzero(scores == scores.max())[-1]
    return grid[best], scores[best], grid, scores


def plateau(grid, scores, tolerance):
    """Width of the threshold range whose score stays within tolerance of the best - and its middle."""
    inside = grid[scores >= scores.max() - tolerance]
    return float(inside.max() - inside.min()), float((inside.max() + inside.min()) / 2)


def split_half(conf, has_pair, right, vehicle, n_splits, seed):
    """Fit on half the vehicles, score on the other half, both ways round; the gap is the cost of fitting."""
    rng = np.random.default_rng(seed)
    vehicles = np.unique(vehicle)
    inside, outside, centre_outside = [], [], []
    for _ in range(n_splits):
        half = rng.permutation(vehicles)[: len(vehicles) // 2]
        a = np.isin(vehicle, half)
        for fit_rows, test_rows in ((a, ~a), (~a, a)):
            t, s_in, grid, scores = fit(conf, has_pair, right, fit_rows)
            inside.append(s_in)
            outside.append(score(conf, has_pair, right, t, test_rows)[0])
            centre_outside.append(score(conf, has_pair, right, plateau(grid, scores, 0.002)[1], test_rows)[0])
    inside, outside, centre_outside = map(np.array, (inside, outside, centre_outside))
    return {"fit_half": float(inside.mean()), "other_half_argmax": float(outside.mean()),
            "other_half_plateau_centre": float(centre_outside.mean()),
            "optimism": float((inside - outside).mean()), "optimism_sd": float((inside - outside).std())}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--val-config", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    data_cfg, val_cfg, cfg = (load_yaml(p) for p in (args.data_config, args.val_config, args.config))
    device = pick_device()
    query, gallery_full, no_match = load_fold(Path(val_cfg["out_dir"]), cfg["eval_fold"])
    gallery = gallery_full[~gallery_full.vehicle_id.isin(no_match)].reset_index(drop=True)
    out = Path("artifacts") / cfg["exp_id"]
    out.mkdir(parents=True, exist_ok=True)

    results = {}
    for name, spec in cfg["candidates"].items():
        similarity, confidence, _, _ = realistic(spec, query, gallery, data_cfg, device)
        conf, has_pair, right = per_query(similarity, confidence, query, gallery)
        top10, _ = reid_eval_top10(similarity, query.vehicle_id.to_numpy(), query.camera_id.to_numpy(),
                                   gallery.vehicle_id.to_numpy(), gallery.camera_id.to_numpy())
        old = score(conf, has_pair, right, spec["threshold"])
        t, best, grid, scores = fit(conf, has_pair, right)
        _, f1, tnr = score(conf, has_pair, right, t)
        width_1, centre = plateau(grid, scores, 0.001)
        width_5, _ = plateau(grid, scores, 0.005)
        halves = split_half(conf, has_pair, right, query.vehicle_id.to_numpy(), cfg["splits"], cfg["seed"])
        results[name] = {
            "mAP10": float(top10["mAP10"]),
            "old_threshold": {"threshold": spec["threshold"], "refusal": float(old[0]), "F1": float(old[1]),
                              "TNR": float(old[2]),
                              **official_refusal_diagnostics(conf, has_pair, right, spec["threshold"])},
            "refitted": {"threshold": float(t), "refusal": float(best), "F1": float(f1), "TNR": float(tnr),
                         "points": 45 * float(top10["mAP10"]) + 10 * float(best),
                         **official_refusal_diagnostics(conf, has_pair, right, t)},
            "plateau": {"within_0.001": width_1, "within_0.005": width_5, "centre": centre},
            "split_half": halves}
        print(json.dumps({"candidate": name, **results[name]}), flush=True)
    (out / "summary.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
