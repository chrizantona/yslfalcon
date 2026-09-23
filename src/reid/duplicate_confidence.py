"""A refusal confidence that ignores near-duplicate gallery frames, and the test-set diagnostic it enables (EXP-207).

Our protocol builds a query without a pair by dropping its whole vehicle from the gallery, so nothing similar is left
and the top-1 z-score collapses - which is what the threshold of EXP-178/EXP-202 separates. The real test does not
look like that: its z-scores match the full-gallery setting of a fold, where 94% of queries have a junk top-1 (the
same vehicle from the same camera, a near-duplicate of the query). Two very different worlds produce that picture:

  A - every test query really has a cross-camera pair, and refusing 5% of them is right;
  B - the queries without a pair keep their own same-camera frame in the gallery, so they look maximally confident
      and our z-score is blind to them, which would cost most of the refusal points.

The confidence here drops every gallery frame whose cosine to the query is above tau (a near-duplicate is almost
always the same shot from the same camera) and scores the best of the rest:

    conf(q) = (max_{g: cos(q,g) <= tau} cos(q,g) - mean_g cos(q,g)) / std_g cos(q,g).

Scenario B is simulated on a fold we hold labels for: for every no-match vehicle only its frame from one camera stays
in the gallery, and only that vehicle's queries from the same camera stay in the query set. Both confidences are then
scored on both scenarios, and the winner is applied to the real test file: a left mode there means we live in B.

Usage: PYTHONPATH=src python -m reid.duplicate_confidence --val-config configs/val_v2.yaml \
           --config configs/exp207_duplicate_confidence.yaml
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def duplicate_aware(sim, tau):
    """z-score of the best gallery frame that is not a near-duplicate of the query; the plain z-score when tau is None
    or when every frame is a duplicate."""
    baseline = sim.mean(1), np.maximum(sim.std(1), 1e-12)
    if tau is None:
        return (sim.max(1) - baseline[0]) / baseline[1]
    kept = np.where(sim > tau, -np.inf, sim).max(1)
    best = np.where(np.isfinite(kept), kept, sim.max(1))
    return (best - baseline[0]) / baseline[1]


def same_camera_only(query, gallery, no_match):
    """Scenario B: every no-match vehicle keeps one camera, in the gallery and in the query set, so its queries have a
    junk twin in the gallery and no cross-camera pair. Returns (query rows, gallery rows)."""
    keep_q, keep_g = np.ones(len(query), bool), np.ones(len(gallery), bool)
    for vehicle in no_match:
        rows = query.vehicle_id == vehicle
        if not rows.any():
            continue
        camera = query.loc[rows, "camera_id"].value_counts().idxmax()   # the camera with the most queries
        keep_q &= ~(rows & (query.camera_id != camera)).to_numpy()
        keep_g &= ~((gallery.vehicle_id == vehicle) & (gallery.camera_id != camera)).to_numpy()
    return keep_q, keep_g


def refusal(conf, has_pair, right, threshold):
    """0.7 F1 + 0.3 TNR at a threshold, counted as the organisers do (query level, top candidate)."""
    answered = conf >= threshold
    tp = int((answered & has_pair & right).sum())
    fp = int((answered & ~(has_pair & right)).sum())
    fn = int((~answered & has_pair).sum())
    tn = int((~answered & ~has_pair).sum())
    f1 = 2 * tp / max(2 * tp + fp + fn, 1)
    tnr = tn / max(int((~has_pair).sum()), 1)
    return {"score": 0.7 * f1 + 0.3 * tnr, "F1": f1, "TNR": tnr, "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def best_threshold(conf, has_pair, right):
    grid = np.unique(conf)
    scores = np.array([refusal(conf, has_pair, right, t)["score"] for t in grid])
    inside = grid[scores >= scores.max() - 0.001]
    return float(grid[np.flatnonzero(scores == scores.max())[-1]]), float((inside.min() + inside.max()) / 2), \
        float(scores.max())


def scenario(query, gallery, q_emb, g_emb, no_match, taus, carried):
    """Both confidences on one query/gallery pair: what a query needs is a cross-camera frame of its own vehicle."""
    sim = q_emb @ g_emb.T
    q_pid, q_cam = query.vehicle_id.to_numpy(), query.camera_id.to_numpy()
    g_pid, g_cam = gallery.vehicle_id.to_numpy(), gallery.camera_id.to_numpy()
    same = g_pid[None] == q_pid[:, None]
    has_pair = (same & (g_cam[None] != q_cam[:, None])).any(1)
    top = np.argsort(-sim, axis=1, kind="stable")
    junk = same[np.arange(len(query))[:, None], top] & (g_cam[top] == q_cam[:, None])
    decisive = top[np.arange(len(query)), np.argmax(~junk, axis=1)]        # the first candidate that is not junk
    right = g_pid[decisive] == q_pid
    out = {"queries": int(len(query)), "gallery": int(len(gallery)),
           "no_pair_share": float((~has_pair).mean()),
           "junk_top1_share": float(junk[:, 0].mean())}
    for tau in [None] + list(taus):
        conf = duplicate_aware(sim, tau)
        top_t, centre, best = best_threshold(conf, has_pair, right)
        out["plain" if tau is None else f"tau{tau}"] = {
            "best_threshold": top_t, "plateau_centre": centre, "best_score": best,
            "at_plateau_centre": refusal(conf, has_pair, right, centre),
            "at_carried": refusal(conf, has_pair, right, carried),
            "conf_quantiles": [round(float(v), 3) for v in np.percentile(conf, [1, 5, 10, 20, 50])],
            "conf_no_pair_median": float(np.median(conf[~has_pair])) if (~has_pair).any() else None,
            "conf_pair_median": float(np.median(conf[has_pair]))}
    return out


def main():
    from reid.data import load_yaml          # imported here so the functions above need no torch
    from reid.evaluate import load_fold

    parser = argparse.ArgumentParser()
    parser.add_argument("--val-config", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    val_cfg, cfg = load_yaml(args.val_config), load_yaml(args.config)
    out_dir = Path("artifacts") / cfg["exp_id"]
    out_dir.mkdir(parents=True, exist_ok=True)
    taus, carried = cfg["taus"], cfg["threshold"]

    query, gallery_full, no_match = load_fold(Path(val_cfg["out_dir"]), cfg["fold"])
    emb = Path(cfg["embeddings_dir"])
    q_emb = np.load(emb / "query_emb.npy").astype(np.float32)
    g_emb = np.load(emb / "gallery_emb.npy").astype(np.float32)
    q_emb /= np.linalg.norm(q_emb, axis=1, keepdims=True)
    g_emb /= np.linalg.norm(g_emb, axis=1, keepdims=True)

    report = {}
    keep_g = ~gallery_full.vehicle_id.isin(no_match).to_numpy()           # scenario C: the vehicle is gone entirely
    report["scenario_C_vehicle_removed"] = scenario(
        query, gallery_full[keep_g].reset_index(drop=True), q_emb, g_emb[keep_g], no_match, taus, carried)
    keep_q, keep_b = same_camera_only(query, gallery_full, no_match)      # scenario B: its same-camera twin stays
    report["scenario_B_same_camera_twin"] = scenario(
        query[keep_q].reset_index(drop=True), gallery_full[keep_b].reset_index(drop=True),
        q_emb[keep_q], g_emb[keep_b], no_match, taus, carried)
    report["full_gallery"] = scenario(query, gallery_full, q_emb, g_emb, no_match, taus, carried)

    test = np.load(cfg["test_embeddings"]).astype(np.float32)             # the real test: no labels, distributions only
    test /= np.linalg.norm(test, axis=1, keepdims=True)
    tq, tg = test[: cfg["n_test_query"]], test[cfg["n_test_query"]:]
    sim = tq @ tg.T
    report["test"] = {"queries": int(len(tq)), "gallery": int(len(tg)),
                      "top1_above_tau": {str(t): float((sim.max(1) > t).mean()) for t in taus}}
    for tau in [None] + list(taus):
        conf = duplicate_aware(sim, tau)
        report["test"]["plain" if tau is None else f"tau{tau}"] = {
            "quantiles": [round(float(v), 3) for v in np.percentile(conf, [1, 5, 10, 20, 30, 50])],
            "below_carried": float((conf < carried).mean()),
            "histogram": [int(v) for v in np.histogram(conf, bins=np.arange(0, 14.5, 0.5))[0]]}
    (out_dir / "summary.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
