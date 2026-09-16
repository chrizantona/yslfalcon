"""Score a trained checkpoint with the benchmark precision, through the unchanged evaluate_fold protocol."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from reid.benchmark import InferencePrecision
from reid.data import load_yaml
from reid.embed import embed_boxes, pick_device
from reid.evaluate import evaluate_fold, load_fold, save_fold_results
from reid.models import ReIDModel, build_backbone


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--val-config", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    data, val, cfg = load_yaml(args.data_config), load_yaml(args.val_config), load_yaml(args.config)
    checkpoint = Path(data["checkpoint_dir"]) / cfg["checkpoint"]
    state = torch.load(checkpoint, map_location="cpu")["state_dict"]
    backbone, dim, mean, std = build_backbone(cfg["model"], data.get("external_weights_dir"), load_pretrained=False)
    model = ReIDModel(backbone, dim, state["classifier.weight"].shape[0])
    model.load_state_dict(state)
    device = pick_device()
    precision = cfg["benchmark"]["precision"]
    encoder = InferencePrecision(model, precision, device).eval().to(device)
    query, gallery, no_match = load_fold(val["out_dir"], cfg["eval_fold"])

    def embed(rows):
        return embed_boxes(encoder, rows, Path(data["dataset_dir"]) / "images", cfg["model"]["input_hw"],
                           mean, std, data["crop_max_side"], device=device, **cfg["embed"])

    result = evaluate_fold(embed, query, gallery, no_match)
    header = {"exp_id": cfg["exp_id"], "protocol": val["version"], "fold": cfg["eval_fold"],
              "checkpoint": checkpoint.name, "precision": precision, "n_query": len(query), "n_gallery": len(gallery)}
    metrics = save_fold_results(Path("artifacts") / cfg["exp_id"] / "val_fold0", result, query, header,
                                emb_dtype=np.float16)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
