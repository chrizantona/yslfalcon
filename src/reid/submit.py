"""Test submission: embeddings.npy (all query rows, then all gallery rows, csv order), submission.csv (top-10 for every
query, refusals included) and candidates.csv (query_id, gallery_id, confidence; a refused query has no rows).

Writes to artifacts/<exp_id>/submission/. The refusal threshold is refusal.threshold from the config, or the one
calibrated by reid.evaluate if that is null.

Usage: PYTHONPATH=src python -m reid.submit --data-config configs/data.yaml --config configs/exp000_clip_zeroshot.yaml
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from reid.data import load_csv, load_yaml
from reid.embed import ClipEmbedder, embed_boxes, pick_device
from reid.inference import load_trained_encoder
from reid.benchmark import InferencePrecision
from reid.metrics import TOP_K, zscore
from reid import view_head


def build_submission(query_emb, gallery_emb, query_ids, gallery_ids, threshold, max_candidates=TOP_K, penalty=None):
    """(submission, candidates) frames.

    penalty (nq x ng) is subtracted from the similarity for ranking only - the viewpoint penalty of reid.view_head.
    A query is answered when its top-1 z-score *before* the penalty is at least threshold, because that is the
    quantity the threshold of EXP-202 was calibrated on; a refused query gets no rows, as the organisers read
    candidates.csv. Their evaluate.py sorts a query's candidates by confidence and judges the highest one, ignoring
    the rest, so one row per query is enough and leaves the choice unambiguous."""
    sim = query_emb @ gallery_emb.T
    ranking = sim if penalty is None else sim - penalty
    if penalty is not None and max_candidates > 1:
        raise ValueError("with a penalty the listed order and the confidences disagree: use max_candidates 1")
    order = np.argsort(-ranking, axis=1, kind="stable")
    top = order[:, :TOP_K]                        # a gallery smaller than ten leaves shorter rows, as in their example
    submission = pd.DataFrame(gallery_ids[top], columns=[f"gallery_id_{i}" for i in range(1, top.shape[1] + 1)])
    submission.insert(0, "query_id", query_ids)

    confidence = zscore(sim).max(1)
    rows = [{"query_id": query_id, "gallery_id": gallery_ids[j], "confidence": round(float(confidence[qi]), 4)}
            for qi, query_id in enumerate(query_ids) if confidence[qi] >= threshold
            for j in order[qi, :max_candidates]]
    return submission, pd.DataFrame(rows, columns=["query_id", "gallery_id", "confidence"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    data_cfg, cfg = load_yaml(args.data_config), load_yaml(args.config)

    threshold = cfg["refusal"].get("threshold")
    if threshold is None:
        calibration_dir = cfg["refusal"].get("calibration_dir", Path("artifacts") / cfg["exp_id"] / f"val_fold{cfg['eval_fold']}")
        calibration = Path(calibration_dir) / "threshold.json"
        threshold = json.loads(calibration.read_text())["threshold"]

    dataset_dir = Path(data_cfg["dataset_dir"])
    query, gallery = load_csv(dataset_dir, "query"), load_csv(dataset_dir, "gallery")
    device = pick_device()
    checkpoint_name = cfg.get("checkpoint", cfg.get("checkpoint_name"))
    if checkpoint_name:
        checkpoint = Path(data_cfg["checkpoint_dir"]) / checkpoint_name
        encoder, _, mean, std = load_trained_encoder(cfg["model"], checkpoint,
                                                    data_cfg.get("external_weights_dir"), cfg.get("checkpoint_sha256"))
        precision = cfg.get("inference", {}).get("precision", "fp32")
        encoder = InferencePrecision(encoder, precision, device).eval().to(device)

        def embed(rows):
            return embed_boxes(encoder, rows, dataset_dir / "images", cfg["model"]["input_hw"], mean, std,
                               data_cfg["crop_max_side"], device=device, **cfg["embed"])
    else:
        embedder = ClipEmbedder(**cfg["model"], device=device)

        def embed(rows):
            return embedder.embed_boxes(rows, dataset_dir / "images", data_cfg["crop_max_side"], **cfg["embed"])

    view = cfg.get("view")            # the viewpoint penalty of EXP-192/204; the head is a 1152x3 matrix
    if view is None:
        query_emb, gallery_emb, penalty = embed(query), embed(gallery), None
    else:
        if not checkpoint_name:
            raise ValueError("the viewpoint penalty needs a trained encoder, not a zero-shot CLIP tower")
        head = view_head.load(Path(data_cfg["checkpoint_dir"]) / view["head"])
        query_emb, q_tokens = view_head.mean_patch_tokens(encoder, lambda: embed(query))
        gallery_emb, g_tokens = view_head.mean_patch_tokens(encoder, lambda: embed(gallery))
        q_views, g_views = view_head.views(q_tokens, head), view_head.views(g_tokens, head)
        penalty = view_head.penalty(q_views, g_views, view["delta"])
        print(f"views: query {dict(zip(*np.unique(q_views, return_counts=True)))}, "
              f"gallery {dict(zip(*np.unique(g_views, return_counts=True)))}; "
              f"penalised pairs {float((penalty > 0).mean()):.1%}")

    out = Path("artifacts") / cfg["exp_id"] / "submission"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "embeddings.npy", np.concatenate([query_emb, gallery_emb]))
    submission, candidates = build_submission(query_emb, gallery_emb, query.image_id.to_numpy(),
                                              gallery.image_id.to_numpy(), threshold,
                                              cfg["refusal"]["max_candidates"], penalty)
    submission.to_csv(out / "submission.csv", index=False, header=False)  # the organisers' format has no header
    candidates.to_csv(out / "candidates.csv", index=False)

    answered = candidates.query_id.nunique()
    print(f"threshold {threshold:.3f} · {len(query)} queries · refused {1 - answered / len(query):.1%} · "
          f"{len(candidates)} candidate rows")


if __name__ == "__main__":
    main()
