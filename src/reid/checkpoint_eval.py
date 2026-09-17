"""Dev-fold evaluation of a trained checkpoint, including single-image flip TTA."""
import argparse
import json
from pathlib import Path

import numpy as np

from reid.benchmark import InferencePrecision
from reid.data import load_yaml
from reid.embed import embed_boxes, pick_device
from reid.evaluate import evaluate_fold, load_fold, save_fold_results
from reid.inference import load_trained_encoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--val-config", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    data, val, cfg = load_yaml(args.data_config), load_yaml(args.val_config), load_yaml(args.config)
    checkpoint = Path(data["checkpoint_dir"]) / cfg["checkpoint"]
    encoder, _, mean, std = load_trained_encoder(cfg["model"], checkpoint,
                                                data.get("external_weights_dir"), cfg.get("checkpoint_sha256"))
    device = pick_device()
    precision = cfg.get("inference", {}).get("precision", "fp32")
    encoder = InferencePrecision(encoder, precision, device).eval().to(device)
    query, gallery, no_match = load_fold(val["out_dir"], cfg["eval_fold"])

    def embed(rows):
        return embed_boxes(encoder, rows, Path(data["dataset_dir"]) / "images", cfg["model"]["input_hw"],
                           mean, std, data["crop_max_side"], device=device, **cfg["embed"])

    result = evaluate_fold(embed, query, gallery, no_match)
    header = {"exp_id": cfg["exp_id"], "protocol": val["version"], "fold": cfg["eval_fold"],
              "checkpoint": checkpoint.name, "checkpoint_mb": round(checkpoint.stat().st_size / 1e6, 1),
              "precision": precision, "flip_tta": bool(cfg["embed"].get("flip_tta")),
              "n_query": len(query), "n_gallery": len(gallery)}
    metrics = save_fold_results(Path("artifacts") / cfg["exp_id"] / f"val_fold{cfg['eval_fold']}",
                                result, query, header, emb_dtype=np.float16)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
