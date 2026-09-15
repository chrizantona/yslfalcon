"""Evaluate an embedding model on one fold of the frozen protocol and calibrate the refusal threshold.

load_fold / evaluate_fold / save_fold_results are shared with reid.train, so every experiment is scored the same way.
The CLI evaluates a zero-shot CLIP config and writes to artifacts/<exp_id>/val_fold<k>/: metrics.json, threshold.json,
refusal_curve.csv, per_query_ap.csv, query_emb.npy, gallery_emb.npy.

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


def load_fold(protocol_dir, fold):
    """(query, gallery, vehicle ids removed from the gallery for refusal calibration) of one protocol fold."""
    protocol_dir = Path(protocol_dir)
    return (pd.read_csv(protocol_dir / f"fold{fold}_query.csv"),
            pd.read_csv(protocol_dir / f"fold{fold}_gallery.csv"),
            pd.read_csv(protocol_dir / f"fold{fold}_no_match_vehicles.csv").vehicle_id)


def evaluate_fold(embed_fn, query, gallery, no_match_vehicles, refusal_cfg):
    """Retrieval metrics and refusal calibration for embed_fn(df) -> L2-normalised embeddings."""
    query_emb, gallery_emb = embed_fn(query), embed_fn(gallery)
    q_pid, q_cam = query.vehicle_id.to_numpy(), query.camera_id.to_numpy()
    g_pid, g_cam = gallery.vehicle_id.to_numpy(), gallery.camera_id.to_numpy()
    sim = query_emb @ gallery_emb.T
    retrieval, ap = reid_eval(sim, q_pid, q_cam, g_pid, g_cam)
    keep = ~gallery.vehicle_id.isin(no_match_vehicles).to_numpy()
    curve = refusal_curve(sim[:, keep], q_pid, q_cam, g_pid[keep], g_cam[keep], refusal_cfg["max_candidates"])
    best = pick_threshold(curve, refusal_cfg["min_tnr"])
    return {"retrieval": retrieval, "refusal": {key: float(value) for key, value in best.items()}, "ap": ap,
            "curve": curve, "query_emb": query_emb, "gallery_emb": gallery_emb}


def save_fold_results(out, result, query, header, emb_dtype=np.float32):
    """Write metrics.json, threshold.json, refusal_curve.csv, per_query_ap.csv and the embeddings; return the metrics."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "query_emb.npy", result["query_emb"].astype(emb_dtype))
    np.save(out / "gallery_emb.npy", result["gallery_emb"].astype(emb_dtype))
    query.assign(AP=result["ap"])[["image_id", "vehicle_id", "camera_id", "AP"]].to_csv(out / "per_query_ap.csv", index=False)
    result["curve"].to_csv(out / "refusal_curve.csv", index=False)
    metrics = {**header, "retrieval": result["retrieval"], "refusal": result["refusal"]}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    threshold = {"score": "zscore", "threshold": result["refusal"]["threshold"]}
    (out / "threshold.json").write_text(json.dumps(threshold, indent=2))
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--val-config", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    data_cfg, val_cfg, cfg = load_yaml(args.data_config), load_yaml(args.val_config), load_yaml(args.config)

    fold = cfg["eval_fold"]
    query, gallery, no_match = load_fold(val_cfg["out_dir"], fold)
    embedder = ClipEmbedder(**cfg["model"], device=pick_device())
    images_dir = Path(data_cfg["dataset_dir"]) / "images"

    def embed(df):
        return embedder.embed_boxes(df, images_dir, data_cfg["crop_max_side"], **cfg["embed"])

    result = evaluate_fold(embed, query, gallery, no_match, cfg["refusal"])
    header = {"exp_id": cfg["exp_id"], "protocol": val_cfg["version"], "fold": fold,
              "n_query": len(query), "n_gallery": len(gallery)}
    metrics = save_fold_results(Path("artifacts") / cfg["exp_id"] / f"val_fold{fold}", result, query, header)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
