"""Uniform weight soup: average the weights of fine-tuning runs that started from the same checkpoint.

One model at inference, so latency and size stay those of a single run. The submitted encoder is the soup of the three
seeds of configs/train/stage2_finetune_all_folds.yaml (EXP-206); they trained on every fold, so there is nothing
left to score it on. Trimmed copy of reid.soup on the `experiments` branch, without the architectures of rejected
experiments and without the dev-fold scoring the submitted soup does not run.

Usage: PYTHONPATH=src python -m reid.soup --data-config configs/data.yaml --config configs/train/stage3_soup.yaml
"""
import argparse
import json
from pathlib import Path

import torch

from reid.data import load_yaml
from reid.models import ReIDModel, build_backbone, checkpoint_name


def average_state_dicts(states):
    """Floating-point tensors are averaged; integer buffers (BN batch counters) are taken from the first state."""
    return {k: torch.stack([s[k].float() for s in states]).mean(0).to(v.dtype) if v.is_floating_point() else v
            for k, v in states[0].items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    data_cfg, cfg = load_yaml(args.data_config), load_yaml(args.config)

    checkpoints = Path(data_cfg["checkpoint_dir"])
    soup = average_state_dicts([torch.load(checkpoints / name, map_location="cpu")["state_dict"]
                                for name in cfg["checkpoints"]])
    # rebuilding the architecture around the average proves the soup loads as one model
    backbone, feat_dim, _, _ = build_backbone(cfg["model"], data_cfg.get("external_weights_dir"), load_pretrained=False)
    ReIDModel(backbone, feat_dim, soup["classifier.weight"].shape[0]).load_state_dict(soup)
    name = checkpoint_name(cfg)
    if (checkpoints / name).exists():
        raise FileExistsError(checkpoints / name)
    torch.save({"exp_id": cfg["exp_id"], "config": cfg, "state_dict": soup}, checkpoints / name)

    metrics = {"exp_id": cfg["exp_id"], "members": cfg["checkpoints"], "checkpoint": name,
               "checkpoint_mb": round((checkpoints / name).stat().st_size / 1e6, 1)}
    out = Path("artifacts") / cfg["exp_id"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
