"""Retrieval metrics (official mAP@10, full-ranking mAP and mINP), candidate-mode scoring and paired bootstrap."""
import numpy as np
import pandas as pd

TOP_K = 10  # candidates per query in submission.csv


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


def reid_eval_top10(sim, q_pid, q_cam, g_pid, g_cam, k=10):
    """Main metric of the challenge: mAP@10, Rank-1 and Rank-5 from the top-10 list that submission.csv holds.

    Junk (same vehicle AND camera) is dropped from the list and the rest keeps its order; with only ten ids in the
    file nothing moves up from rank 11 (unconfirmed until the organisers' evaluate.py is available). AP@10 sums the
    precision at every positive in the list and divides by min(n_pos, 10), n_pos = cross-camera positives in the
    whole gallery. Queries without a positive are excluded. Returns (summary, per-query AP@10 with NaN)."""
    order = np.argsort(-sim, axis=1, kind="stable")[:, :k]
    same = g_pid[order] == q_pid[:, None]
    junk = same & (g_cam[order] == q_cam[:, None])
    good = same & ~junk
    rank = np.cumsum(~junk, axis=1)
    hits = np.cumsum(good, axis=1)
    n_pos = ((g_pid[None, :] == q_pid[:, None]) & (g_cam[None, :] != q_cam[:, None])).sum(1)
    ok = n_pos > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        ap = np.where(ok, np.where(good, hits / rank, 0).sum(1) / np.minimum(n_pos, k), np.nan)
    first_good = np.where(good.any(1), rank[np.arange(len(q_pid)), good.argmax(1)], k + 1)
    summary = {
        "mAP10": float(np.nanmean(ap)),
        "R1": float(np.mean(first_good[ok] <= 1)),
        "R5": float(np.mean(first_good[ok] <= 5)),
        "n_queries": int(ok.sum()),
    }
    return summary, ap


def zscore(sim):
    """Each similarity standardised against its query's similarity distribution over the gallery."""
    return (sim - sim.mean(1, keepdims=True)) / np.maximum(sim.std(1, keepdims=True), 1e-12)


def refusal_curve(sim, q_pid, q_cam, g_pid, g_cam, n_thresholds=201, confidence=None, grid=None):
    """Candidate mode scored as the organisers do, for a grid of thresholds on the top-1 z-score, or on another
    per-query confidence when one is given (reid.refusal_confidence); the ranking stays the same.

    A query is answered when its best gallery item has z >= t; answered queries list their top items in candidates.csv.
    The decisive candidate is the first listed one that is not junk (same vehicle AND camera), as for mAP@10; the
    strict variant takes the very first one, junk included, and counts junk as wrong. Query level, micro-averaged:
    TP - a cross-camera pair exists and the decisive candidate is that vehicle; FP - answered but the candidate is
    wrong or no pair exists; FN - a pair exists but no answer; TN - no pair and no answer.
    F1 = 2TP / (2TP + FP + FN), TNR = TN / (TN + FP) over queries without a pair, score = 0.7 F1 + 0.3 TNR."""
    z = zscore(sim)
    order = np.argsort(-z, axis=1, kind="stable")[:, :TOP_K]
    rows = np.arange(len(q_pid))
    same = g_pid[order] == q_pid[:, None]
    junk = same & (g_cam[order] == q_cam[:, None])
    decisive = order[rows, np.argmax(~junk, axis=1)]
    correct = g_pid[decisive] == q_pid
    correct_strict = same[:, 0] & ~junk[:, 0]
    has_pair = ((g_pid[None, :] == q_pid[:, None]) & (g_cam[None, :] != q_cam[:, None])).any(1)
    top_z = z[rows, order[:, 0]] if confidence is None else np.asarray(confidence)

    def confusion(answered, right):
        tp = (answered & has_pair & right).sum()
        fp = (answered & ~(has_pair & right)).sum()
        fn = (~answered & has_pair).sum()
        tn = (~answered & ~has_pair).sum()
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        tnr = tn / max((~has_pair).sum(), 1)
        return tp, fp, fn, f1, tnr

    curve = []
    # grid: score these exact thresholds instead of the quantiles - a holdout keeps the threshold fitted elsewhere
    for t in (np.unique(np.quantile(top_z, np.linspace(0, 1, n_thresholds))) if grid is None else np.atleast_1d(grid)):
        answered = top_z >= t
        tp, fp, fn, f1, tnr = confusion(answered, correct)
        f1_strict = confusion(answered, correct_strict)[3]
        curve.append({"threshold": float(t), "score": 0.7 * f1 + 0.3 * tnr, "F1": f1, "TNR": tnr,
                      "precision": tp / max(tp + fp, 1), "recall": tp / max(tp + fn, 1), "answered": answered.mean(),
                      "F1_strict": f1_strict, "score_strict": 0.7 * f1_strict + 0.3 * tnr})
    return pd.DataFrame(curve)


def pick_threshold(curve):
    """The row with the best organisers' score 0.7 F1 + 0.3 TNR (the higher threshold on ties)."""
    return curve.sort_values(["score", "threshold"], kind="stable").iloc[-1]


def paired_bootstrap(ap_base, ap_new, vehicle_ids, n_boot=2000, seed=0):
    """Mean AP gain of ap_new over ap_base and its 95% CI; vehicles are resampled with all their queries."""
    ok = ~(np.isnan(ap_base) | np.isnan(ap_new))
    diff = (ap_new - ap_base)[ok]
    per_vehicle = pd.DataFrame({"v": np.asarray(vehicle_ids)[ok], "d": diff}).groupby("v").d.agg(["sum", "size"])
    idx = np.random.default_rng(seed).integers(0, len(per_vehicle), (n_boot, len(per_vehicle)))
    boot = per_vehicle["sum"].to_numpy()[idx].sum(1) / per_vehicle["size"].to_numpy()[idx].sum(1)
    low, high = np.percentile(boot, [2.5, 97.5])
    return float(diff.mean()), float(low), float(high)
