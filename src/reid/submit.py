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


def build_submission(query_emb, gallery_emb, query_ids, gallery_ids, threshold, max_candidates=TOP_K):
    """(submission, candidates) frames. A query is answered when its top-1 z-score >= threshold; it then lists its top
    max_candidates items with their z-scores, so a same-camera twin dropped by the scorer still leaves a candidate.
    A refused query gets no rows, as the organisers read candidates.csv."""
    sim = query_emb @ gallery_emb.T
    order = np.argsort(-sim, axis=1, kind="stable")
    submission = pd.DataFrame(gallery_ids[order[:, :TOP_K]], columns=[f"gallery_id_{i}" for i in range(1, TOP_K + 1)])
    submission.insert(0, "query_id", query_ids)

    z = zscore(sim)
    rows = [{"query_id": query_id, "gallery_id": gallery_ids[j], "confidence": round(float(z[qi, j]), 4)}
            for qi, query_id in enumerate(query_ids) if z[qi, order[qi, 0]] >= threshold
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

    query_emb, gallery_emb = embed(query), embed(gallery)

    out = Path("artifacts") / cfg["exp_id"] / "submission"
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "embeddings.npy", np.concatenate([query_emb, gallery_emb]))
    submission, candidates = build_submission(query_emb, gallery_emb, query.image_id.to_numpy(),
                                              gallery.image_id.to_numpy(), threshold, cfg["refusal"]["max_candidates"])
    submission.to_csv(out / "submission.csv", index=False)
    candidates.to_csv(out / "candidates.csv", index=False)

    answered = candidates.query_id.nunique()
    print(f"threshold {threshold:.3f} · {len(query)} queries · refused {1 - answered / len(query):.1%} · "
          f"{len(candidates)} candidate rows")


if __name__ == "__main__":
    main()
