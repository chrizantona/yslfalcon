"""Evaluate an embedding model on one fold of the frozen protocol and calibrate the refusal threshold.

Writes to artifacts/<exp_id>/val_fold<k>/: metrics.json, threshold.json, refusal_curve.csv, per_query_ap.csv,
query_emb.npy, gallery_emb.npy.

Usage: PYTHONPATH=src python -m reid.evaluate --data-config configs/data.yaml --val-config configs/val_v1.yaml \
           --config configs/exp000_clip_zeroshot.yaml
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from reid.data import load_yaml
from reid.embed import ClipEmbedder, pick_device
from reid.metrics import pick_threshold, refusal_curve, reid_eval


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--val-config", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    data_cfg, val_cfg, cfg = load_yaml(args.data_config), load_yaml(args.val_config), load_yaml(args.config)

    fold = cfg["eval_fold"]
    protocol_dir = Path(val_cfg["out_dir"])
    out = Path("artifacts") / cfg["exp_id"] / f"val_fold{fold}"
    out.mkdir(parents=True, exist_ok=True)

    query = pd.read_csv(protocol_dir / f"fold{fold}_query.csv")
    gallery = pd.read_csv(protocol_dir / f"fold{fold}_gallery.csv")
    embedder = ClipEmbedder(**cfg["model"], device=pick_device())
    images_dir = Path(data_cfg["dataset_dir"]) / "images"
    query_emb = embedder.embed_boxes(query, images_dir, data_cfg["crop_max_side"], **cfg["embed"])
    gallery_emb = embedder.embed_boxes(gallery, images_dir, data_cfg["crop_max_side"], **cfg["embed"])
    np.save(out / "query_emb.npy", query_emb)
    np.save(out / "gallery_emb.npy", gallery_emb)

    q_pid, q_cam = query.vehicle_id.to_numpy(), query.camera_id.to_numpy()
    g_pid, g_cam = gallery.vehicle_id.to_numpy(), gallery.camera_id.to_numpy()
    sim = query_emb @ gallery_emb.T
    retrieval, ap = reid_eval(sim, q_pid, q_cam, g_pid, g_cam)
    query.assign(AP=ap)[["image_id", "vehicle_id", "camera_id", "AP"]].to_csv(out / "per_query_ap.csv", index=False)

    no_match = pd.read_csv(protocol_dir / f"fold{fold}_no_match_vehicles.csv").vehicle_id
    keep = ~gallery.vehicle_id.isin(no_match).to_numpy()
    refusal = cfg["refusal"]
    curve = refusal_curve(sim[:, keep], q_pid, q_cam, g_pid[keep], g_cam[keep], refusal["max_candidates"])
    curve.to_csv(out / "refusal_curve.csv", index=False)
    best = pick_threshold(curve, refusal["min_tnr"])

    metrics = {
        "exp_id": cfg["exp_id"],
        "protocol": val_cfg["version"],
        "fold": fold,
        "n_query": len(query),
        "n_gallery": len(gallery),
        "retrieval": retrieval,
        "refusal": {key: float(value) for key, value in best.items()},
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "threshold.json").write_text(json.dumps({"score": "zscore", "threshold": float(best.threshold)}, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
