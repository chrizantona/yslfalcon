"""Uniform weight soup: average the weights of fine-tuning runs that started from the same checkpoint.

One model at inference, so latency and size stay those of a single run. The soup is saved to checkpoint_dir under a
random name and scored on the dev fold; writes artifacts/<exp_id>/val_fold<k>/ like reid.evaluate.

Usage: PYTHONPATH=src python -m reid.soup --data-config configs/data.yaml --val-config configs/val_v1.yaml \
           --config configs/exp016_soup_exp010_seeds.yaml
"""
import argparse
import json
import uuid
from pathlib import Path

import numpy as np
import torch

from reid.data import load_yaml
from reid.embed import embed_boxes, pick_device
from reid.evaluate import evaluate_fold, load_fold, save_fold_results
from reid.models import ReIDModel, build_backbone


def average_state_dicts(states):
    """Floating-point tensors are averaged; integer buffers (BN batch counters) are taken from the first state."""
    return {k: torch.stack([s[k].float() for s in states]).mean(0).to(v.dtype) if v.is_floating_point() else v
            for k, v in states[0].items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--val-config", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    data_cfg, val_cfg, cfg = load_yaml(args.data_config), load_yaml(args.val_config), load_yaml(args.config)
    fold, device = cfg["eval_fold"], pick_device()
    query, gallery, no_match = load_fold(val_cfg["out_dir"], fold)

    checkpoints = Path(data_cfg["checkpoint_dir"])
    soup = average_state_dicts([torch.load(checkpoints / name, map_location="cpu")["state_dict"]
                                for name in cfg["checkpoints"]])
    backbone, feat_dim, mean, std = build_backbone(cfg["model"])
    model = ReIDModel(backbone, feat_dim, soup["classifier.weight"].shape[0])
    model.load_state_dict(soup)
    name = f"{uuid.uuid4().hex}.pt"  # weights stay on the pod; the name goes to the card and the registry
    torch.save({"exp_id": cfg["exp_id"], "config": cfg, "state_dict": soup}, checkpoints / name)
    model = model.eval().to(device)

    images_dir = Path(data_cfg["dataset_dir"]) / "images"

    def embed(df):
        return embed_boxes(model, df, images_dir, cfg["model"]["input_hw"], mean, std, data_cfg["crop_max_side"],
                           device=device, **cfg["embed"])

    result = evaluate_fold(embed, query, gallery, no_match)
    header = {"exp_id": cfg["exp_id"], "protocol": val_cfg["version"], "fold": fold, "n_query": len(query),
              "n_gallery": len(gallery), "members": cfg["checkpoints"], "checkpoint": name,
              "checkpoint_mb": round((checkpoints / name).stat().st_size / 1e6, 1)}
    metrics = save_fold_results(Path("artifacts") / cfg["exp_id"] / f"val_fold{fold}", result, query, header,
                                emb_dtype=np.float16)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
