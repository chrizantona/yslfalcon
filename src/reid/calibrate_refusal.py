"""The refusal threshold of the submission (EXP-202): five out-of-sample fits, combined into one number.

The submitted model trains on every fold, so no fold is left to fit its threshold on. The frozen recipe is therefore
trained five more times, each time without one fold (configs/train/threshold_loo_fold{k}.yaml, one seed), and each
run's threshold is fitted on the fold it never saw, the way the test is scored:
  - one gallery without the no-pair vehicles, so about 20% of the queries have no pair, as in the closed test;
  - confidence = z-score of the top-1 cosine similarity against the query's own similarities over the gallery;
  - F1 and TNR counted as the organisers' evaluate.py counts them: query level, the top candidate is right whenever it
    is the same vehicle (junk included); score = 0.7 F1 + 0.3 TNR.
A run's threshold is the centre of the plateau where the score stays within plateau_tolerance of its best. The
submitted threshold is the median of the five centres plus soup_offset: a soup of three seeds scores its top-1 higher
than one seed does, by 0.03 as measured on the pair EXP-161 -> EXP-164 (experiments branch).

Trimmed copy of reid.calibrate_refusal on the `experiments` branch (per_query, score, fit and plateau are unchanged;
the split-half diagnostics, which do not enter the number, are left out), with the five folds combined in one run.

Usage: PYTHONPATH=src python -m reid.calibrate_refusal --data-config configs/data.yaml \
           --val-config configs/train/val_v2_loo.yaml --config configs/train/threshold_calibration.yaml
"""
import argparse
import json
from pathlib import Path

import numpy as np

from reid.data import load_yaml
from reid.embed import embed_boxes, pick_device
from reid.evaluate import load_fold
from reid.inference import load_trained_encoder
from reid.metrics import reid_eval_top10, zscore


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


def combine(centres, offset):
    """The submitted threshold: the median of the out-of-sample plateau centres plus the seed-to-soup offset."""
    return float(np.median(centres)) + offset


def similarity_and_confidence(spec, query, gallery, data_cfg, device):
    """Cosine similarity of one leave-one-fold-out model over one gallery and its top-1 z-score, fp32 as in EXP-202."""
    model_cfg = load_yaml(spec["config"])["model"]
    encoder, _, mean, std = load_trained_encoder(model_cfg, Path(data_cfg["checkpoint_dir"]) / spec["checkpoint"],
                                                 data_cfg.get("external_weights_dir"))
    encoder = encoder.eval().to(device)
    images_dir = Path(data_cfg["dataset_dir"]) / "images"

    def embed(rows):
        emb = embed_boxes(encoder, rows, images_dir, model_cfg["input_hw"], mean, std, data_cfg["crop_max_side"],
                          device=device, batch_size=128, num_workers=8)
        return emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)  # as EXP-202 did, once more

    similarity = embed(query) @ embed(gallery).T
    return similarity, zscore(similarity).max(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--val-config", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    data_cfg, val_cfg, cfg = (load_yaml(p) for p in (args.data_config, args.val_config, args.config))
    device = pick_device()
    out = Path("artifacts") / cfg["exp_id"]
    out.mkdir(parents=True, exist_ok=True)

    folds = {}
    for fold, spec in cfg["folds"].items():  # the model of each entry never saw this fold
        query, gallery_full, no_match = load_fold(Path(val_cfg["out_dir"]), fold)
        gallery = gallery_full[~gallery_full.vehicle_id.isin(no_match)].reset_index(drop=True)
        similarity, confidence = similarity_and_confidence(spec, query, gallery, data_cfg, device)
        conf, has_pair, right = per_query(similarity, confidence, query, gallery)
        top10, _ = reid_eval_top10(similarity, query.vehicle_id.to_numpy(), query.camera_id.to_numpy(),
                                   gallery.vehicle_id.to_numpy(), gallery.camera_id.to_numpy())
        t, best, grid, scores = fit(conf, has_pair, right)
        _, f1, tnr = score(conf, has_pair, right, t)
        width, centre = plateau(grid, scores, cfg["plateau_tolerance"])
        folds[fold] = {"checkpoint": spec["checkpoint"], "mAP10": float(top10["mAP10"]),
                       "best": {"threshold": float(t), "refusal": float(best), "F1": float(f1), "TNR": float(tnr)},
                       "plateau": {"width": width, "centre": centre}}
        print(json.dumps({"fold": fold, **folds[fold]}), flush=True)

    centres = [f["plateau"]["centre"] for f in folds.values()]
    result = {"folds": folds, "median_centre": float(np.median(centres)), "spread": float(np.ptp(centres)),
              "soup_offset": cfg["soup_offset"], "threshold": combine(centres, cfg["soup_offset"])}
    (out / "threshold.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "folds"}), flush=True)


if __name__ == "__main__":
    main()
