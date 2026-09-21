"""Score candidates with the organisers' evaluate.py on a fold we hold labels for, next to our own scorer.

The real test has one gallery in which about a fifth of the queries have no pair; ranking and refusal are both scored
on it. Our val_v2 protocol scores retrieval on the whole fold gallery and refusal on the gallery with the no-pair
vehicles removed. To separate the two kinds of difference, every candidate is written out in the official format for
the single-gallery setting (the reduced gallery, as the test is built), scored by evaluate.py and by reid.metrics on
the same files, and our protocol number is printed alongside.

Usage: PYTHONPATH=src python -m reid.official_check --data-config configs/data.yaml --val-config configs/val_v2.yaml \
           --config configs/exp177_official_check.yaml --evaluate /path/to/evaluate.py
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from reid.data import load_yaml
from reid.embed import pick_device
from reid.evaluate import load_fold
from reid.gallery_transforms import alpha_qe, normalise
from reid.holdout import embeddings
from reid.inference import load_trained_encoder
from reid.metrics import TOP_K, refusal_curve, reid_eval_top10, zscore
from reid.rerank_pairs import PairReranker, candidate_scores, plain_loader, reorder, token_bank


def official_files(out, query, gallery, similarity, confidence, threshold, query_emb, gallery_emb):
    """submission.csv (no header), candidates.csv (one row per answered query), embeddings.npy, ground truth."""
    out.mkdir(parents=True, exist_ok=True)
    order = np.argsort(-similarity, axis=1, kind="stable")
    q_ids, g_ids = query.image_id.to_numpy(), gallery.image_id.to_numpy()
    top = pd.DataFrame(g_ids[order[:, :TOP_K]])
    top.insert(0, "query_id", q_ids)
    top.to_csv(out / "submission.csv", index=False, header=False)          # the spec says: no header
    answered = confidence >= threshold
    pd.DataFrame({"query_id": q_ids[answered], "gallery_id": g_ids[order[answered, 0]],
                  "confidence": np.round(confidence[answered], 4)}).to_csv(out / "candidates.csv", index=False)
    np.save(out / "embeddings.npy", np.concatenate([query_emb, gallery_emb]).astype(np.float32))
    pd.concat([query.assign(split="query"), gallery.assign(split="gallery")])[
        ["image_id", "vehicle_id", "camera_id", "split"]].to_csv(out / "ground_truth.csv", index=False)
    query[["image_id"]].assign(x=0, y=0, w=1, h=1).to_csv(out / "test_query.csv", index=False)
    gallery[["image_id"]].assign(x=0, y=0, w=1, h=1).to_csv(out / "test_gallery.csv", index=False)


def run_official(evaluate, out):
    report = out / "report.json"
    subprocess.run([sys.executable, str(evaluate), "--gt", str(out / "ground_truth.csv"),
                    "--submission", str(out / "submission.csv"), "--candidates", str(out / "candidates.csv"),
                    "--embeddings", str(out / "embeddings.npy"), "--query", str(out / "test_query.csv"),
                    "--gallery", str(out / "test_gallery.csv"), "--json", str(report)],
                   check=True, capture_output=True, text=True)
    return json.loads(report.read_text())


def ours(similarity, confidence, threshold, query, gallery):
    """reid.metrics on exactly the files written above: the same gallery and the same threshold."""
    q_pid, q_cam = query.vehicle_id.to_numpy(), query.camera_id.to_numpy()
    g_pid, g_cam = gallery.vehicle_id.to_numpy(), gallery.camera_id.to_numpy()
    top10, _ = reid_eval_top10(similarity, q_pid, q_cam, g_pid, g_cam)
    row = refusal_curve(similarity, q_pid, q_cam, g_pid, g_cam, confidence=confidence, grid=threshold).iloc[0]
    return {"mAP10": float(top10["mAP10"]), "R1": float(top10["R1"]), "F1": float(row["F1"]),
            "TNR": float(row["TNR"]), "F1_strict": float(row["F1_strict"])}


def realistic(spec, query, gallery, data_cfg, device):
    """(similarity, confidence, query_emb, gallery_emb) of one candidate against a single gallery, as at test time:
    alpha-QE, when the candidate has it, expands over that same gallery and the re-ranker reorders its top window."""
    images_dir = Path(data_cfg["dataset_dir"]) / "images"
    model_cfg = load_yaml(spec["config"])["model"]
    encoder, _, mean, std = load_trained_encoder(model_cfg, Path(data_cfg["checkpoint_dir"]) / spec["checkpoint"],
                                                 data_cfg.get("external_weights_dir"))
    encoder = encoder.eval().to(device)
    query_emb = normalise(embeddings(encoder, query, images_dir, model_cfg, data_cfg, mean, std, device))
    gallery_emb = normalise(embeddings(encoder, gallery, images_dir, model_cfg, data_cfg, mean, std, device))
    if spec.get("alpha_qe"):
        query_emb, gallery_emb = alpha_qe(query_emb, gallery_emb, spec["alpha_qe"]["k"],
                                          spec["alpha_qe"]["alpha"])
    similarity = query_emb @ gallery_emb.T
    confidence = zscore(similarity).max(1)          # the encoder's top-1 z-score, unchanged by the re-ranker
    if spec.get("reranker"):
        crops, hw = data_cfg["crops_dir"], model_cfg["input_hw"]
        q_tok = token_bank(encoder[0], plain_loader(query, crops, hw, mean, std, 64), device)
        g_tok = token_bank(encoder[0], plain_loader(gallery, crops, hw, mean, std, 64), device)
        saved = torch.load(Path(data_cfg["checkpoint_dir"]) / spec["reranker"]["checkpoint"], map_location=device)
        model = PairReranker(q_tok.shape[-1], saved["config"]["width"], saved["config"]["blocks"],
                             saved["config"]["heads"], saved["config"].get("dropout", 0.0)).to(device)
        model.load_state_dict(saved["state_dict"])
        picked = np.argsort(-similarity, axis=1, kind="stable")[:, :spec["reranker"]["window"]]
        similarity = reorder(similarity, picked, candidate_scores(model, q_tok, g_tok, picked, device),
                             spec["reranker"]["weight"])
    del encoder
    torch.cuda.empty_cache()
    return similarity, confidence, query_emb, gallery_emb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--val-config", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--evaluate", required=True, help="the organisers' evaluate.py")
    args = parser.parse_args()
    data_cfg, val_cfg, cfg = (load_yaml(p) for p in (args.data_config, args.val_config, args.config))
    device = pick_device()
    query, gallery_full, no_match = load_fold(Path(val_cfg["out_dir"]), cfg["eval_fold"])
    keep = ~gallery_full.vehicle_id.isin(no_match).to_numpy()
    gallery = gallery_full[keep].reset_index(drop=True)                     # one gallery, as in the real test
    root = Path("artifacts") / cfg["exp_id"]
    print(json.dumps({"fold": cfg["eval_fold"], "queries": len(query), "gallery": len(gallery),
                      "queries_without_pair": int(query.vehicle_id.isin(no_match).sum())}), flush=True)

    results = {}
    for name, spec in cfg["candidates"].items():
        similarity, confidence, query_emb, gallery_emb = realistic(spec, query, gallery, data_cfg, device)
        out = root / name
        official_files(out, query, gallery, similarity, confidence, spec["threshold"], query_emb, gallery_emb)
        official = run_official(Path(args.evaluate), out)
        mine = ours(similarity, confidence, spec["threshold"], query, gallery)
        results[name] = {"official": official, "ours_same_files": mine}
        f1, tnr = official["candidates"]["F1"], official["candidates"]["TNR"]
        print(json.dumps({"candidate": name,
                          "official": {"mAP10": round(official["ranking"]["mAP@10"], 6),
                                       "R1": round(official["ranking"]["Rank-1"], 6), "F1": round(f1, 6),
                                       "TNR": round(tnr, 6), "refusal": round(0.7 * f1 + 0.3 * tnr, 6),
                                       "PR_AUC": round(official["candidates"]["PR-AUC"], 6)},
                          "ours_same_files": {k: round(v, 6) for k, v in mine.items()}}), flush=True)
    (root / "summary.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
