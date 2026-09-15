"""Retrieval metrics (Market-1501 protocol), refusal-mode curve and paired bootstrap."""
import numpy as np
import pandas as pd


def reid_eval(sim, q_pid, q_cam, g_pid, g_cam, chunk=1024):
    """mAP, Rank-1, Rank-5, mINP for a (n_query, n_gallery) similarity matrix.

    Gallery items with the same vehicle AND camera as the query are junk and ignored; other same-camera items stay
    negatives. Returns (summary, per-query AP with NaN for queries that have no cross-camera match)."""
    n = len(q_pid)
    ap, first, inp, junk10 = (np.full(n, np.nan) for _ in range(4))
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        order = np.argsort(-sim[s:e], axis=1, kind="stable")
        match = g_pid[order] == q_pid[s:e, None]
        junk = match & (g_cam[order] == q_cam[s:e, None])
        good = match & ~junk
        rank = np.cumsum(~junk, axis=1)  # 1-based rank among non-junk items
        hits = np.cumsum(good, axis=1)
        npos = good.sum(1)
        rows = np.arange(e - s)
        last = rank[rows, good.shape[1] - 1 - good[:, ::-1].argmax(1)]
        with np.errstate(invalid="ignore", divide="ignore"):
            ap[s:e] = np.where(npos > 0, np.where(good, hits / rank, 0).sum(1) / npos, np.nan)
            first[s:e] = np.where(npos > 0, rank[rows, good.argmax(1)], np.nan)
            inp[s:e] = np.where(npos > 0, npos / last, np.nan)
        junk10[s:e] = junk[:, :10].sum(1)
    ok = ~np.isnan(ap)
    summary = {
        "mAP": float(ap[ok].mean()),
        "R1": float(np.mean(first[ok] <= 1)),
        "R5": float(np.mean(first[ok] <= 5)),
        "mINP": float(inp[ok].mean()),
        "junk_in_top10": float(junk10[ok].mean()),
        "n_queries": int(ok.sum()),
    }
    return summary, ap


def zscore(sim):
    """Each similarity standardised against its query's similarity distribution over the gallery."""
    return (sim - sim.mean(1, keepdims=True)) / np.maximum(sim.std(1, keepdims=True), 1e-12)


def refusal_curve(sim, q_pid, q_cam, g_pid, g_cam, max_candidates=10, n_thresholds=201):
    """Query-level precision / recall / F1 / TNR of the candidate mode over a grid of z-score thresholds.

    A query is answered if its best candidate has z >= t; accepted candidates are its top items with z >= t.
    The answer is correct only if an accepted candidate is the same vehicle from another camera (same-camera matches
    don't count). Queries whose vehicle has no other-camera frame in the gallery are the negatives for TNR."""
    z = zscore(sim)
    order = np.argsort(-z, axis=1, kind="stable")[:, :max_candidates]
    z_top = np.take_along_axis(z, order, 1)
    hit = (g_pid[order] == q_pid[:, None]) & (g_cam[order] != q_cam[:, None])
    has_match = ((g_pid[None, :] == q_pid[:, None]) & (g_cam[None, :] != q_cam[:, None])).any(1)
    rows = []
    for t in np.unique(np.quantile(z_top[:, 0], np.linspace(0, 1, n_thresholds))):
        accepted = z_top >= t
        answered = accepted.any(1)
        tp = (accepted & hit).any(1).sum()
        precision = tp / max(answered.sum(), 1)
        recall = tp / max(has_match.sum(), 1)
        rows.append({
            "threshold": float(t),
            "precision": precision,
            "recall": recall,
            "F1": 2 * precision * recall / max(precision + recall, 1e-12),
            "TNR": (~answered & ~has_match).sum() / max((~has_match).sum(), 1),
            "answered": answered.mean(),
        })
    return pd.DataFrame(rows)


def pick_threshold(curve, min_tnr):
    """Best-F1 row among thresholds with TNR >= min_tnr (all rows if none qualifies)."""
    eligible = curve[curve.TNR >= min_tnr]
    return (eligible if len(eligible) else curve).sort_values("F1", kind="stable").iloc[-1]


def paired_bootstrap(ap_base, ap_new, vehicle_ids, n_boot=2000, seed=0):
    """Mean AP gain of ap_new over ap_base and its 95% CI; vehicles are resampled with all their queries."""
    ok = ~(np.isnan(ap_base) | np.isnan(ap_new))
    diff = (ap_new - ap_base)[ok]
    per_vehicle = pd.DataFrame({"v": np.asarray(vehicle_ids)[ok], "d": diff}).groupby("v").d.agg(["sum", "size"])
    idx = np.random.default_rng(seed).integers(0, len(per_vehicle), (n_boot, len(per_vehicle)))
    boot = per_vehicle["sum"].to_numpy()[idx].sum(1) / per_vehicle["size"].to_numpy()[idx].sum(1)
    low, high = np.percentile(boot, [2.5, 97.5])
    return float(diff.mean()), float(low), float(high)
