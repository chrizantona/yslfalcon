"""CLIP zero-shot viewpoint labels (front / back / side) of the training frames: the targets of the viewpoint head.

The labeller is the public CLIP ViT-B/16 (laion2b_s34b_b88k) of configs/train/clip_labeller.yaml; it sees each box on
its own and never an identity. Only train.csv is labelled - the head is fitted on training frames alone, and at test
time the view comes from the head, not from CLIP. The labels are committed as protocol/attributes.csv.

Trimmed copy of reid.attributes on the `experiments` branch, which also labelled colour and taxi livery and the test
files for slice analysis; the view prompt, the model and the crops are unchanged.

Usage: PYTHONPATH=src python -m reid.attributes --data-config configs/data.yaml --config configs/train/clip_labeller.yaml
"""
import argparse
from pathlib import Path

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from open_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD

from reid.data import load_csv, load_yaml
from reid.embed import embed_boxes, pick_device

VIEW_LABELS, VIEW_TEMPLATE = ["front", "back", "side"], "a photo of the {} of a car"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", default="protocol/attributes.csv")
    args = parser.parse_args()
    data_cfg, cfg = load_yaml(args.data_config), load_yaml(args.config)

    dataset_dir = Path(data_cfg["dataset_dir"])
    df = load_csv(dataset_dir, "train")

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
        tokens = tokenizer([VIEW_TEMPLATE.format(label) for label in VIEW_LABELS]).to(device)
        text = F.normalize(model.encode_text(tokens).float(), dim=-1).cpu().numpy()
    df["view"] = np.array(VIEW_LABELS)[(emb @ text.T).argmax(1)]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df[["image_id", "view"]].to_csv(out, index=False)
    print(f"{len(df)} rows -> {out}")
    print("view", df.view.value_counts(normalize=True).round(3).to_dict())


if __name__ == "__main__":
    main()
