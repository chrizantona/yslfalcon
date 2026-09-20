"""Fold 4 evaluation of candidates declared in advance, with every constant carried over from fold 0 (EXP-171).

The holdout is worth something only because no decision was ever taken on it. So nothing is fitted here: the refusal
threshold, the alpha-QE parameters and the re-ranker's blending weight are the values chosen on fold 0 and are used
as they are. What a re-fitted threshold would have given is reported too, clearly apart, for information only - it
takes part in no comparison.

Usage: PYTHONPATH=src python -m reid.holdout --data-config configs/data.yaml --val-config configs/val_v2.yaml \
           --config configs/exp171_fold4.yaml
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from reid.data import load_yaml
from reid.embed import embed_boxes, pick_device
from reid.evaluate import load_fold
from reid.gallery_transforms import alpha_qe, normalise
from reid.inference import load_trained_encoder
from reid.metrics import pick_threshold, refusal_curve, reid_eval, reid_eval_top10
from reid.rerank_pairs import PairReranker, candidate_scores, plain_loader, reorder, token_bank


def embeddings(encoder, rows, images_dir, model_cfg, data_cfg, mean, std, device):
    return embed_boxes(encoder, rows, images_dir, model_cfg["input_hw"], mean, std, data_cfg["crop_max_side"],
                       device=device, batch_size=128, num_workers=8)


def score(similarity, reduced, query, gallery, keep, threshold):
    """Retrieval on the whole gallery, refusal on the reduced one - at the threshold carried over from fold 0."""
    q_pid, q_cam = query.vehicle_id.to_numpy(), query.camera_id.to_numpy()
    g_pid, g_cam = gallery.vehicle_id.to_numpy(), gallery.camera_id.to_numpy()
    top10, ap = reid_eval_top10(similarity, q_pid, q_cam, g_pid, g_cam)
    full, _ = reid_eval(similarity, q_pid, q_cam, g_pid, g_cam)
    carried = refusal_curve(reduced, q_pid, q_cam, g_pid[keep], g_cam[keep], grid=threshold).iloc[0]
    refitted = pick_threshold(refusal_curve(reduced, q_pid, q_cam, g_pid[keep], g_cam[keep]))
    return {"mAP10": float(top10["mAP10"]), "R1": float(top10["R1"]), "R5": float(top10["R5"]),
            "mAP": float(full["mAP"]), "mINP": float(full["mINP"]),
            "refusal": float(carried["score"]), "F1": float(carried["F1"]), "TNR": float(carried["TNR"]),
            "threshold_carried": float(threshold),
            "points": 45 * float(top10["mAP10"]) + 10 * float(carried["score"]),
            "refitted_here_only_for_information": {"refusal": float(refitted["score"]),
                                                   "threshold": float(refitted["threshold"])}}, ap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--val-config", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    data_cfg, val_cfg, cfg = (load_yaml(p) for p in (args.data_config, args.val_config, args.config))
    device = pick_device()
    fold = cfg["eval_fold"]
    if fold != val_cfg["holdout_fold"]:
        raise ValueError("this module exists for the holdout fold only")
    query, gallery, no_match = load_fold(Path(val_cfg["out_dir"]), fold)
    keep = ~gallery.vehicle_id.isin(no_match).to_numpy()
    images_dir = Path(data_cfg["dataset_dir"]) / "images"
    out = Path("artifacts") / cfg["exp_id"]
    out.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"fold": fold, "queries": len(query), "gallery": len(gallery),
                      "no_pair_share": round(float(query.vehicle_id.isin(no_match).mean()), 4)}), flush=True)

    results = {}
    for name, spec in cfg["candidates"].items():
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
        reduced = query_emb @ gallery_emb[keep].T

        if not spec.get("reranker"):
            metrics, ap = score(similarity, reduced, query, gallery, keep, spec["threshold"])
            results[name] = metrics
        else:
            tower = encoder[0]
            crops = data_cfg["crops_dir"]
            hw = model_cfg["input_hw"]
            query_tokens = token_bank(tower, plain_loader(query, crops, hw, mean, std, 64), device)
            gallery_tokens = token_bank(tower, plain_loader(gallery, crops, hw, mean, std, 64), device)
            window = spec["reranker"]["window"]
            picked = np.argsort(-similarity, axis=1, kind="stable")[:, :window]
            picked_reduced = np.argsort(-reduced, axis=1, kind="stable")[:, :window]
            per_seed, aps = {}, []
            for checkpoint in spec["reranker"]["checkpoints"]:
                saved = torch.load(Path(data_cfg["checkpoint_dir"]) / checkpoint, map_location=device)
                model = PairReranker(query_tokens.shape[-1], saved["config"]["width"], saved["config"]["blocks"],
                                     saved["config"]["heads"], saved["config"].get("dropout", 0.0)).to(device)
                model.load_state_dict(saved["state_dict"])
                weight = spec["reranker"]["weight"]
                ranked = reorder(similarity, picked,
                                 candidate_scores(model, query_tokens, gallery_tokens, picked, device), weight)
                ranked_reduced = reorder(reduced, picked_reduced,
                                         candidate_scores(model, query_tokens, gallery_tokens[torch.as_tensor(keep)],
                                                          picked_reduced, device), weight)
                per_seed[checkpoint], ap = score(ranked, ranked_reduced, query, gallery, keep, spec["threshold"])
                aps.append(ap)
                print(json.dumps({"candidate": name, "reranker": checkpoint,
                                  **{k: round(v, 6) for k, v in per_seed[checkpoint].items()
                                     if isinstance(v, float)}}), flush=True)
            results[name] = {key: float(np.mean([s[key] for s in per_seed.values()]))
                             for key in ("mAP10", "R1", "R5", "mAP", "mINP", "refusal", "F1", "TNR", "points")}
            results[name]["per_reranker"] = per_seed
            ap = np.nanmean(aps, axis=0)  # the seed-averaged AP, as the bootstrap expects
        query.assign(AP=ap).to_csv(out / f"per_query_ap_{name}.csv", index=False)
        print(json.dumps({"candidate": name, **{k: round(v, 6) for k, v in results[name].items()
                                                if isinstance(v, float)}}), flush=True)
        del encoder
        torch.cuda.empty_cache()

    (out / "summary.json").write_text(json.dumps({"exp_id": cfg["exp_id"], "protocol": val_cfg["version"],
                                                  "fold": fold, "candidates": results}, indent=2) + "\n")
    order = sorted(results, key=lambda name: -results[name]["points"])
    print(json.dumps({"ranking_by_points": [(name, round(results[name]["points"], 4)) for name in order]}), flush=True)


if __name__ == "__main__":
    main()
