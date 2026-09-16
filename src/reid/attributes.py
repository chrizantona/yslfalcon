"""Per-image attributes for slice analysis: CLIP zero-shot view / colour / taxi plus box geometry.

Dataset-level and independent of the fold split, so it is generated once and committed as protocol/attributes.csv.

Usage: PYTHONPATH=src python -m reid.attributes --data-config configs/data.yaml --config configs/exp000_clip_zeroshot.yaml
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import open_clip
import pandas as pd
import torch
import torch.nn.functional as F
from open_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD

from reid.data import CSV_FILES, load_csv, load_yaml
from reid.embed import embed_boxes, pick_device

PROMPTS = {
    "view": (["front", "back", "side"], "a photo of the {} of a car"),
    "color": (["white", "black", "silver", "gray", "red", "blue", "green", "yellow", "orange", "brown", "beige"],
              "a photo of a {} car"),
    "taxi": (["taxi", "private"], "a photo of a {} car"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", default="protocol/attributes.csv")
    args = parser.parse_args()
    data_cfg, cfg = load_yaml(args.data_config), load_yaml(args.config)

    dataset_dir = Path(data_cfg["dataset_dir"])
    df = pd.concat([load_csv(dataset_dir, split).assign(split=split) for split in CSV_FILES], ignore_index=True)
    frame_h, frame_w = cv2.imread(str(dataset_dir / "images" / f"{df.image_id.iloc[0]}.jpg")).shape[:2]

    device = pick_device()
    model, _, _ = open_clip.create_model_and_transforms(cfg["model"]["name"], pretrained=cfg["model"]["pretrained"])
    tokenizer = open_clip.get_tokenizer(cfg["model"]["name"])
    model = model.eval().to(device)
    preprocess = getattr(model.visual, "preprocess_cfg", None) or {}
    mean = preprocess.get("mean", OPENAI_DATASET_MEAN)
    std = preprocess.get("std", OPENAI_DATASET_STD)
    emb = embed_boxes(model.visual, df, dataset_dir / "images", cfg["model"]["input_hw"], mean, std,
                      data_cfg["crop_max_side"], device=device, **cfg["embed"])

    with torch.no_grad():
        for name, (labels, template) in PROMPTS.items():
            tokens = tokenizer([template.format(label) for label in labels]).to(device)
            text = F.normalize(model.encode_text(tokens).float(), dim=-1).cpu().numpy()
            df[name] = np.array(labels)[(emb @ text.T).argmax(1)]

    df["area_k"] = df.w * df.h / 1e3
    df["aspect"] = df.w / df.h
    df["truncated"] = (df.x <= 0) | (df.y <= 0) | (df.x + df.w >= frame_w) | (df.y + df.h >= frame_h)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    columns = ["image_id", "split", "area_k", "aspect", "truncated", "view", "color", "taxi"]
    df[columns].to_csv(out, index=False)
    print(f"{len(df)} rows -> {out}")
    for name in PROMPTS:
        print(name, df[name].value_counts(normalize=True).round(3).to_dict())


if __name__ == "__main__":
    main()
