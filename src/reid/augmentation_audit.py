"""Render and measure the exact augmentation recipe before a costly training run.

The report separates each transform and the full stochastic recipe. Cosine is measured by the unmodified pretrained
tower between an original crop and its augmented view; it is a guardrail, not a replacement for visual inspection.

Usage: PYTHONPATH=src python -m reid.augmentation_audit --data-config configs/data.yaml \
           --config configs/exp167_siglip2_giant_vehicleid_teacher.yaml --source vehicleid_train \
           --out artifacts/EXP-167/augmentation_audit --device cuda
"""
import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from reid.data import load_yaml
from reid.external import load_table
from reid.models import build_backbone
from reid.train import CropDataset


def _plain(train):
    return {**train, "flip_p": 0, "pad": 0, "erasing_p": 0, "color_jitter": None}


def audit_recipes(train):
    """The original plus isolated transforms and the exact full training recipe."""
    plain = _plain(train)
    jitter = dict(train.get("color_jitter") or {})
    if jitter:
        jitter["p"] = 1.0
    return {
        "original": plain,
        "flip": {**plain, "flip_p": 1.0},
        "pad_crop": {**plain, "pad": train.get("pad", 0)},
        "color": {**plain, "color_jitter": jitter or None},
        "erase": {**plain, "erasing_p": 1.0, "erasing_scale": train.get("erasing_scale", [0.02, 0.08]),
                  "erasing_ratio": train.get("erasing_ratio", [0.5, 2.0])},
        "full": train,
    }


def _to_bgr(tensor, mean, std):
    rgb = (tensor.permute(1, 2, 0).float().numpy() * np.asarray(std) + np.asarray(mean)).clip(0, 1)
    return cv2.cvtColor((rgb * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR)


def _label(image, text):
    canvas = np.full((image.shape[0] + 26, image.shape[1], 3), 245, np.uint8)
    canvas[:image.shape[0]] = image
    cv2.putText(canvas, text, (4, image.shape[0] + 18), cv2.FONT_HERSHEY_SIMPLEX, .42, (20, 20, 20), 1,
                cv2.LINE_AA)
    return canvas


@torch.inference_mode()
def embed(encoder, tensors, device, batch_size):
    chunks = []
    for start in range(0, len(tensors), batch_size):
        x = torch.stack(tensors[start:start + batch_size]).to(device)
        with torch.autocast(device.split(":")[0], dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            chunks.append(F.normalize(encoder(x).float(), dim=1).cpu())
    return torch.cat(chunks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--source", default="vehicleid_train")
    parser.add_argument("--out", required=True)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=166)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    data, cfg, out = load_yaml(args.data_config), load_yaml(args.config), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    spec = data["external"][args.source]
    table = load_table(spec["dir"], spec.get("table", "train.csv"))
    # One frame per sampled identity maximises the range of colours, viewpoints and overlays we inspect.
    table = table.groupby("vehicle_id", sort=False).sample(n=1, random_state=args.seed)
    table = table.sample(n=min(args.samples, len(table)), random_state=args.seed).reset_index(drop=True)
    table["label"] = np.arange(len(table))
    table["crops_dir"] = str(Path(spec["dir"]) / "images")

    encoder, _, mean, std = build_backbone(cfg["model"], data.get("external_weights_dir"))
    encoder = encoder.eval().to(args.device)
    recipes = audit_recipes(cfg["train"])
    views, metadata = [], []
    for column, (name, recipe) in enumerate(recipes.items()):
        dataset = CropDataset(table, None, cfg["model"]["input_hw"], mean, std, recipe)
        for row in range(len(table)):
            random.seed(args.seed * 100000 + row * 100 + column)
            torch.manual_seed(args.seed * 100000 + row * 100 + column)
            views.append(dataset[row][0])
            metadata.append((row, name))
    features = embed(encoder, views, args.device, args.batch_size)
    feature_by_key = {key: feature for key, feature in zip(metadata, features)}

    rows, scores = [], {name: [] for name in recipes if name != "original"}
    view_by_key = {key: view for key, view in zip(metadata, views)}
    for row in range(len(table)):
        tiles = []
        reference = feature_by_key[row, "original"]
        for name in recipes:
            cosine = 1.0 if name == "original" else float(reference @ feature_by_key[row, name])
            if name != "original":
                scores[name].append(cosine)
            tiles.append(_label(_to_bgr(view_by_key[row, name], mean, std), f"{name} cos={cosine:.4f}"))
        rows.append(np.concatenate(tiles, axis=1))
    cv2.imwrite(str(out / "montage.jpg"), np.concatenate(rows, axis=0))

    report = {"source": args.source, "images": len(table), "seed": args.seed,
              "recipe": {key: cfg["train"].get(key) for key in ("flip_p", "pad", "color_jitter", "erasing_p",
                                                                     "erasing_scale", "erasing_ratio")},
              "cosine": {name: {"mean": float(np.mean(values)), "p05": float(np.quantile(values, .05)),
                                  "min": float(np.min(values))} for name, values in scores.items()}}
    (out / "report.json").write_text(json.dumps(report, indent=2))
    table[["image_id", "vehicle_id"]].to_csv(out / "samples.csv", index=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
