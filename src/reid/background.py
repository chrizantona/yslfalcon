"""Background replacement as a training augmentation: the vehicle, cut out by its SAM mask, over a random street scene.

At inference nothing changes - the model sees ordinary crops; the augmentation only keeps it from leaning on the
background of a camera. Two things are prepared once, from files that already exist:
- a mask per training crop, read off the difference between the ordinary crop and its SAM-masked twin (reid.sam_masks,
  then reid.crops with configs/data_sam.yaml): a pixel is background where the masked crop is the fill grey and the
  ordinary one is not;
- a bank of backgrounds, random windows of full frames of the training folds only, so no evaluation scene leaks in.

For pretraining on an external set (train.source) the masks come from reid.sam_masks --external and are only checked
here; the bank is still cut from our training folds, background.bank_folds.

Usage: PYTHONPATH=src python -m reid.background --data-config configs/data.yaml --val-config configs/val_v2.yaml \
           --config configs/exp185_background_swap.yaml
"""
import argparse
import os
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from reid.data import load_yaml
from reid.external import load_table, source_names
from reid.sam_masks import FILL


def vehicle_mask(crop, masked, tol=12, feather=1.5):
    """Float HxW mask of the vehicle, 1 inside, from an ordinary crop and its SAM-masked twin (both HxWx3 uint8)."""
    grey = (np.abs(masked.astype(np.int16) - FILL) <= tol).all(2)
    changed = (np.abs(masked.astype(np.int16) - crop.astype(np.int16)) > tol).any(2)
    vehicle = (~(grey & changed)).astype(np.uint8)
    vehicle = cv2.morphologyEx(vehicle, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))  # JPEG specks
    return cv2.GaussianBlur(vehicle.astype(np.float32), (0, 0), feather) if feather else vehicle.astype(np.float32)


def paste(crop, mask, background):
    """The vehicle of crop over background, resized to the crop; mask is float HxW in [0, 1]."""
    h, w = crop.shape[:2]
    bg = cv2.resize(background, (w, h), interpolation=cv2.INTER_AREA)
    alpha = mask[:, :, None]
    return (alpha * crop + (1 - alpha) * bg).round().astype(np.uint8)


def save(path, image, *params):
    tmp = str(path)[: -len(path.suffix)] + ".tmp" + path.suffix
    cv2.imwrite(tmp, image, list(params))
    os.replace(tmp, path)  # no half-written files if the job is killed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--val-config", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    data_cfg, val_cfg, cfg = (load_yaml(p) for p in (args.data_config, args.val_config, args.config))
    tcfg = cfg["train"]
    spec = tcfg["background"]
    folds = spec.get("bank_folds", tcfg.get("folds"))
    table = pd.read_csv(Path(val_cfg["out_dir"]) / "folds.csv")
    table = table[table.fold.isin(folds)]
    bank_dir = Path(data_cfg["backgrounds_dir"])
    bank_dir.mkdir(parents=True, exist_ok=True)
    if tcfg.get("source"):
        check_external_masks(data_cfg, source_names(tcfg))
    else:
        derive_masks(data_cfg, table)
    make_bank(data_cfg, spec, table, folds, bank_dir)


def check_external_masks(data_cfg, sources):
    for source in sources:
        spec = data_cfg["external"][source]
        ids = load_table(spec["dir"], spec.get("table", "train.csv")).image_id
        missing = [i for i in ids if not (Path(spec["masks_dir"]) / f"{i}.png").exists()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} of {len(ids)} crops of {source} have no mask in "
                                    f"{spec['masks_dir']}: run reid.sam_masks --external {source} first")
        print(f"{source}: masks for all {len(ids)} crops", flush=True)


def derive_masks(data_cfg, table):
    masks_dir = Path(data_cfg["masks_dir"])
    masks_dir.mkdir(parents=True, exist_ok=True)
    crops, masked_crops = Path(data_cfg["crops_dir"]), Path(data_cfg["masked_crops_dir"])
    shares = []
    for image_id in table.image_id:
        path = masks_dir / f"{image_id}.png"
        if path.exists():
            continue
        crop, masked = cv2.imread(str(crops / f"{image_id}.jpg")), cv2.imread(str(masked_crops / f"{image_id}.jpg"))
        if masked is None or masked.shape != crop.shape:
            raise FileNotFoundError(f"no SAM-masked crop of {image_id} matching {crops}: run reid.sam_masks and "
                                    f"reid.crops with configs/data_sam.yaml first")
        mask = vehicle_mask(crop, masked, feather=0)
        shares.append(mask.mean())
        save(path, (mask * 255).astype(np.uint8))
    if shares:
        print(f"{len(shares)} masks, vehicle share of the crop: median {np.median(shares):.2f}, "
              f"under 0.3: {np.sum(np.asarray(shares) < 0.3)}", flush=True)


def make_bank(data_cfg, spec, table, folds, bank_dir):
    rng = random.Random(0)
    frames = sorted(table.image_id)
    images_dir = Path(data_cfg["dataset_dir"]) / "images"
    made = 0
    for k in range(spec["bank_size"]):
        path = bank_dir / f"{k:05d}.jpg"
        if path.exists():
            continue
        frame = cv2.imread(str(images_dir / f"{rng.choice(frames)}.jpg"))
        fh, fw = frame.shape[:2]
        h = int(fh * rng.uniform(0.2, 0.6))
        w = min(fw, int(h * rng.uniform(0.6, 1.8)))
        y, x = rng.randint(0, fh - h), rng.randint(0, fw - w)
        window = frame[y:y + h, x:x + w]
        scale = min(1.0, data_cfg["crop_max_side"] / max(h, w))
        window = cv2.resize(window, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
        save(path, window, cv2.IMWRITE_JPEG_QUALITY, 90)
        made += 1
    print(f"background bank: {made} new windows from {len(frames)} frames of folds {folds}", flush=True)


if __name__ == "__main__":
    main()
