"""Refusal threshold over the five out-of-fold runs and the two gallery settings of reid.clean_gallery (EXP-209).

Score = 0.7 F1 + 0.3 TNR as evaluate.py counts it. The threshold is the centre of the plateau (within 0.001 of the
best) of the score averaged over the five folds and, with weight w, over the two settings (w = share of 'clean').
Its out-of-sample cost is checked fold by fold: fit on four folds, score on the fifth, against the current threshold;
at the main weight and at w = 1, the setting of the closed test (organisers, 20.10.2026).
Thresholds are those of one seed; the soup's is 0.03 higher (EXP-202).

Usage: python -m reid.threshold_mix --dir artifacts/EXP-209
"""
import argparse
import json
from pathlib import Path

import numpy as np

SETTINGS = ("test", "clean")
GRID = np.round(np.arange(4.0, 9.0 + 1e-9, 0.01), 2)
CURRENT_SINGLE = 7.053   # median of the five EXP-202 plateau centres; the submitted 7.0828 is this plus the soup offset
SOUP_OFFSET = 0.03       # one seed -> soup of three, measured by EXP-161 -> EXP-164
WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)
MAIN_WEIGHT = 0.5


def counts(conf, has_pair, right, t):
    answered = conf >= t
    good = has_pair & right
    tp, fp = (answered & good).sum(-1), (answered & ~good).sum(-1)
    fn, tn = (~answered & has_pair).sum(-1), (~answered & ~has_pair).sum(-1)
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    tnr = tn / max(int((~has_pair).sum()), 1)
    return {"recall": tp / np.maximum(tp + fn, 1), "precision": tp / np.maximum(tp + fp, 1), "F1": f1, "TNR": tnr,
            "refusal": 0.7 * f1 + 0.3 * tnr}


def curve(run):
    conf, has_pair, right = run
    return counts(conf[None, :], has_pair, right, GRID[:, None])


def centre(scores, tolerance=0.001):
    inside = GRID[scores >= scores.max() - tolerance]
    return float(round((inside.min() + inside.max()) / 2, 3))


def mixed(curves, folds, w):
    return np.mean([(1 - w) * curves[k]["test"]["refusal"] + w * curves[k]["clean"]["refusal"] for k in folds], 0)


def summary_at(runs, t):
    """Mean over folds (and min..max of the refusal score) of every metric, per setting, at threshold t."""
    out = {}
    for s in SETTINGS:
        per_fold = [counts(*runs[k][s], t) for k in runs]
        out[s] = {m: round(float(np.mean([p[m] for p in per_fold])), 4) for m in per_fold[0]}
        out[s]["refusal_range"] = [round(float(min(p["refusal"] for p in per_fold)), 4),
                                   round(float(max(p["refusal"] for p in per_fold)), 4)]
    out["mean_of_settings"] = round((out["test"]["refusal"] + out["clean"]["refusal"]) / 2, 4)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="the fold*.npz of reid.clean_gallery; the report goes there")
    root = Path(parser.parse_args().dir)
    runs = {}
    for path in sorted(root.glob("fold*.npz")):
        z = np.load(path)
        runs[int(path.stem[4:])] = {s: (z[f"conf_{s}"], z["has_pair"], z[f"right_{s}"]) for s in SETTINGS}
    folds = sorted(runs)
    curves = {k: {s: curve(runs[k][s]) for s in SETTINGS} for k in folds}
    report = {"folds": folds}

    report["own_optimum_per_fold"] = {k: {s: centre(curves[k][s]["refusal"]) for s in SETTINGS} for k in folds}
    report["by_weight_of_clean"] = {}
    for w in WEIGHTS:
        t = centre(mixed(curves, folds, w))
        report["by_weight_of_clean"][w] = {"threshold": t, **summary_at(runs, t)}

    for w, key in ((MAIN_WEIGHT, "leave_one_fold_out"), (1.0, "leave_one_fold_out_clean_only")):
        lofo = []
        for k in folds if len(folds) > 1 else []:
            t = centre(mixed(curves, [j for j in folds if j != k], w))
            new, old = ({s: float(counts(*runs[k][s], th)["refusal"]) for s in SETTINGS} for th in (t, CURRENT_SINGLE))
            lofo.append({"fold": k, "threshold_from_other_folds": t,
                         **{f"delta_{s}": round(new[s] - old[s], 4) for s in SETTINGS}})
        report[key] = lofo
        report[f"{key}_mean_delta"] = {s: round(float(np.mean([r[f"delta_{s}"] for r in lofo])), 4) for s in SETTINGS}

    chosen = report["by_weight_of_clean"][MAIN_WEIGHT]["threshold"]
    table = sorted({6.0, 6.25, 6.5, 6.75, CURRENT_SINGLE, chosen})
    report["at_thresholds"] = {t: summary_at(runs, t) for t in table}
    report["chosen_single_seed"] = chosen
    report["chosen_for_soup"] = round(chosen + SOUP_OFFSET, 3)
    (root / "threshold_report.json").write_text(json.dumps(report, indent=1, default=float) + "\n")
    print(json.dumps(report, indent=1, default=float))


if __name__ == "__main__":
    main()
