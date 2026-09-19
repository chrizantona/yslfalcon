"""Refusal with other per-query confidences than the top-1 z-score, on saved fold embeddings (EXP-074).

The ranking and the candidates do not change; only the number the refusal threshold is put on does. Each confidence
gets its best threshold on the same fold, exactly as reid.evaluate does for the z-score, so the scores compare like for
like. Confidences:
- z_top1 - z-score of the top-1 similarity against the query's own gallery distribution (current);
- cos_top1 - raw cosine of the top-1;
- cos_margin - top-1 cosine minus the best cosine of any other vehicle;
- z_margin - the same margin in z-score units.
The refusal gallery drops the no-match vehicles, as in reid.evaluate. A paired bootstrap over query vehicles gives the
95% CI of each score against z_top1 at the thresholds found on the full fold.

Usage: PYTHONPATH=src python -m reid.refusal_confidence --val-config configs/val_v2.yaml \
           --emb-dir artifacts/EXP-054/val_fold0 --fold 0 --out artifacts/EXP-074/refusal_confidence.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

from reid.data import load_yaml
from reid.evaluate import load_fold
from reid.metrics import pick_threshold, refusal_curve, zscore


def confidences(sim, g_pid):
    """Per-query confidences on a query x gallery cosine matrix."""
    z = zscore(sim)
    rows = np.arange(len(sim))
    top = sim.argmax(1)
    other = g_pid[None, :] != g_pid[top][:, None]  # gallery items of other vehicles than the top-1
    best_other = np.where(other, sim, -np.inf).max(1)
    best_other_z = np.where(other, z, -np.inf).max(1)
    return {"z_top1": z[rows, top], "cos_top1": sim[rows, top],
            "cos_margin": sim[rows, top] - best_other, "z_margin": z[rows, top] - best_other_z}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-config", required=True)
    parser.add_argument("--emb-dir", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args()
    query, gallery, no_match = load_fold(Path(load_yaml(args.val_config)["out_dir"]), args.fold)
    emb_dir = Path(args.emb_dir)
    q = np.load(emb_dir / "query_emb.npy").astype(np.float32)
    g = np.load(emb_dir / "gallery_emb.npy").astype(np.float32)
    keep = ~gallery.vehicle_id.isin(no_match).to_numpy()
    q_pid, q_cam = query.vehicle_id.to_numpy(), query.camera_id.to_numpy()
    g_pid, g_cam = gallery.vehicle_id.to_numpy()[keep], gallery.camera_id.to_numpy()[keep]
    sim = q @ g[keep].T

    results, per_query = {}, {}
    for name, conf in confidences(sim, g_pid).items():
        best = pick_threshold(refusal_curve(sim, q_pid, q_cam, g_pid, g_cam, confidence=conf))
        results[name] = {key: float(value) for key, value in best.items()}
        per_query[name] = conf >= best["threshold"]

    # paired bootstrap over query vehicles of score(name) - score(z_top1) at the fixed thresholds
    order = np.argsort(-zscore(sim), axis=1, kind="stable")[:, :10]
    same = g_pid[order] == q_pid[:, None]
    junk = same & (g_cam[order] == q_cam[:, None])
    correct = g_pid[order[np.arange(len(q_pid)), np.argmax(~junk, axis=1)]] == q_pid
    has_pair = ((g_pid[None, :] == q_pid[:, None]) & (g_cam[None, :] != q_cam[:, None])).any(1)

    def score(answered, idx):
        a, p, c = answered[idx], has_pair[idx], correct[idx]
        tp, fp = (a & p & c).sum(), (a & ~(p & c)).sum()
        fn, tn = (~a & p).sum(), (~a & ~p).sum()
        return 0.7 * 2 * tp / max(2 * tp + fp + fn, 1) + 0.3 * tn / max((~p).sum(), 1)

    vehicles = np.unique(q_pid)
    by_vehicle = {v: np.flatnonzero(q_pid == v) for v in vehicles}
    rng = np.random.default_rng(0)
    draws = [np.concatenate([by_vehicle[v] for v in rng.choice(vehicles, len(vehicles))]) for _ in range(args.n_boot)]
    for name in results:
        diffs = [score(per_query[name], idx) - score(per_query["z_top1"], idx) for idx in draws]
        results[name]["vs_z_top1"] = results[name]["score"] - results["z_top1"]["score"]
        results[name]["vs_z_top1_ci95"] = [float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"emb_dir": str(emb_dir), "fold": args.fold, "n_boot": args.n_boot, "bootstrap_seed": 0,
                               "confidences": results}, indent=2))
    print(json.dumps({k: {m: round(v[m], 4) for m in ("score", "F1", "TNR", "threshold", "vs_z_top1")}
                      | {"ci95": [round(x, 4) for x in v["vs_z_top1_ci95"]]} for k, v in results.items()}, indent=1))


if __name__ == "__main__":
    main()
