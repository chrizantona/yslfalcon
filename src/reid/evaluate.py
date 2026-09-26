"""Score an embedding model on one fold of the frozen protocol and calibrate the refusal threshold there.

Shared by reid.train, so every dev evaluation is scored the same way: mAP@10, R1, R5 as the organisers score
submission.csv, full-ranking mAP and mINP for reference, and the candidate mode with the threshold that maximises
0.7 F1 + 0.3 TNR. Trimmed copy of reid.evaluate on the `experiments` branch: the fold helpers without the CLI that
scored public reference checkpoints.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from reid.metrics import pick_threshold, refusal_curve, reid_eval, reid_eval_top10


def load_fold(protocol_dir, fold):
    """(query, gallery, vehicle ids removed from the gallery for refusal calibration) of one protocol fold."""
    protocol_dir = Path(protocol_dir)
    return (pd.read_csv(protocol_dir / f"fold{fold}_query.csv"),
            pd.read_csv(protocol_dir / f"fold{fold}_gallery.csv"),
            pd.read_csv(protocol_dir / f"fold{fold}_no_match_vehicles.csv").vehicle_id)


def evaluate_fold(embed_fn, query, gallery, no_match_vehicles):
    """Retrieval metrics and refusal calibration for embed_fn(df) -> L2-normalised embeddings, cosine similarity.

    The refusal is calibrated on the gallery without the no-match vehicles, so about 20% of the queries have no pair,
    as in the closed test."""
    query_emb, gallery_emb = embed_fn(query), embed_fn(gallery)
    q_pid, q_cam = query.vehicle_id.to_numpy(), query.camera_id.to_numpy()
    g_pid, g_cam = gallery.vehicle_id.to_numpy(), gallery.camera_id.to_numpy()
    sim = query_emb @ gallery_emb.T
    top10, ap = reid_eval_top10(sim, q_pid, q_cam, g_pid, g_cam)
    full, ap_full = reid_eval(sim, q_pid, q_cam, g_pid, g_cam)
    retrieval = {**top10, "mAP": full["mAP"], "mINP": full["mINP"], "junk_in_top10": full["junk_in_top10"]}
    keep = ~gallery.vehicle_id.isin(no_match_vehicles).to_numpy()
    curve = refusal_curve(query_emb @ gallery_emb[keep].T, q_pid, q_cam, g_pid[keep], g_cam[keep])
    best = pick_threshold(curve)
    refusal = {key: float(value) for key, value in best.items()}
    refusal["no_pair_share"] = float(query.vehicle_id.isin(no_match_vehicles).mean())
    return {"retrieval": retrieval, "refusal": refusal, "ap": ap, "ap_full": ap_full, "curve": curve,
            "query_emb": query_emb, "gallery_emb": gallery_emb}


def save_fold_results(out, result, query, header, emb_dtype=np.float32, save_embeddings=True):
    """Write metrics.json, threshold.json, refusal_curve.csv, per_query_ap.csv and the embeddings; return the metrics."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if save_embeddings:
        np.save(out / "query_emb.npy", result["query_emb"].astype(emb_dtype))
        np.save(out / "gallery_emb.npy", result["gallery_emb"].astype(emb_dtype))
    per_query = query.assign(AP=result["ap"], AP_full=result["ap_full"])
    per_query[["image_id", "vehicle_id", "camera_id", "AP", "AP_full"]].to_csv(out / "per_query_ap.csv", index=False)
    result["curve"].to_csv(out / "refusal_curve.csv", index=False)
    metrics = {**header, "retrieval": result["retrieval"], "refusal": result["refusal"]}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    threshold = {"score": "zscore of the top-1 similarity", "threshold": result["refusal"]["threshold"]}
    (out / "threshold.json").write_text(json.dumps(threshold, indent=2))
    return metrics
