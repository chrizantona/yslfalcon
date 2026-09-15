"""One pass over all frames -> cached features for the EDA notebook.

Per image:
  * quality stats (brightness / saturation / blur / contrast of the crop and frame, file md5)
  * DINOv2 ViT-B/14 embedding of the bbox crop            (instance-level similarity)
  * CLIP ViT-B/16 embedding of the bbox crop              (zero-shot color / body type / view)
  * DINOv2 embedding of the full frame with the bbox masked (scene -> pseudo camera id)

Outputs (row-aligned with eda/cache/meta_stats.parquet) are written to eda/cache/.
Run:  .venv/bin/python eda/extract_features.py
"""
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np
import open_clip
import pandas as pd
import timm
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "BIG_DATA" / "dataset"
CACHE = ROOT / "eda" / "cache"

IN_MEAN, IN_STD = np.array([0.485, 0.456, 0.406], np.float32), np.array([0.229, 0.224, 0.225], np.float32)
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)
CROP_SIZE = (224, 224)   # squash, as in most ReID pipelines
SCENE_SIZE = (392, 224)  # 16:9-ish, multiples of the 14px patch

PROMPTS = {
    "color": ["white", "black", "silver", "gray", "red", "blue", "green", "yellow", "orange", "brown", "beige"],
    "body": ["sedan", "hatchback", "station wagon", "SUV", "minivan", "pickup truck", "cargo van", "box truck", "bus"],
    "view": ["front", "back", "side"],
    "taxi": ["taxi", "private"],
}
TEMPLATES = {
    "color": "a photo of a {} car",
    "body": "a photo of a {}",
    "view": "a photo of the {} of a car",
    "taxi": "a photo of a {} car",
}


def load_meta():
    tr = pd.read_csv(DATA / "train.csv").assign(split="train")
    q = pd.read_csv(DATA / "test_query.csv").assign(split="query")
    g = pd.read_csv(DATA / "test_gallery.csv").assign(split="gallery")
    df = pd.concat([tr, q, g], ignore_index=True)
    df["row"] = df.groupby("split").cumcount()  # position inside its own csv
    return df


def normalize(img_rgb, size, mean, std):
    x = cv2.resize(img_rgb, size, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    return torch.from_numpy(((x - mean) / std).transpose(2, 0, 1).copy())


class Frames(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        raw = (DATA / "images" / f"{r.image_id}.jpg").read_bytes()
        bgr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        H, W = bgr.shape[:2]
        x0, y0 = max(int(r.x), 0), max(int(r.y), 0)
        x1, y1 = min(int(r.x + r.w), W), min(int(r.y + r.h), H)
        crop = bgr[y0:y1, x0:x1]

        small = cv2.resize(bgr, (480, 270), interpolation=cv2.INTER_AREA)
        hsv_f = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hsv_c = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        gray_c = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        # blur measured at a fixed crop width so small and large boxes are comparable
        g256 = cv2.resize(gray_c, (256, max(1, round(256 * crop.shape[0] / crop.shape[1]))), interpolation=cv2.INTER_AREA)
        stats = dict(
            H=H, W=W, file_kb=len(raw) / 1024, md5=hashlib.md5(raw).hexdigest(),
            frame_brightness=float(hsv_f[..., 2].mean()), frame_saturation=float(hsv_f[..., 1].mean()),
            crop_brightness=float(hsv_c[..., 2].mean()), crop_saturation=float(hsv_c[..., 1].mean()),
            crop_contrast=float(gray_c.std()), crop_blur=float(cv2.Laplacian(g256, cv2.CV_64F).var()),
        )

        rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        masked = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        masked[y0:y1, x0:x1] = 124  # hide the vehicle so the embedding describes the scene only
        return (
            stats,
            normalize(rgb_crop, CROP_SIZE, IN_MEAN, IN_STD),
            normalize(rgb_crop, CROP_SIZE, CLIP_MEAN, CLIP_STD),
            normalize(masked, SCENE_SIZE, IN_MEAN, IN_STD),
        )


def collate(batch):
    stats, a, b, c = zip(*batch)
    return list(stats), torch.stack(a), torch.stack(b), torch.stack(c)


def worker_init(_):
    cv2.setNumThreads(1)


@torch.no_grad()
def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    df = load_meta()

    dino = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True, num_classes=0,
                             img_size=224, dynamic_img_size=True).eval().to(dev)
    clip, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="laion2b_s34b_b88k")
    clip = clip.eval().to(dev)
    tok = open_clip.get_tokenizer("ViT-B-16")

    text = {}
    for k, labels in PROMPTS.items():
        t = clip.encode_text(tok([TEMPLATES[k].format(l) for l in labels]).to(dev))
        text[k] = torch.nn.functional.normalize(t, dim=-1).cpu().numpy()
    np.savez(CACHE / "clip_text.npz", **text)
    (CACHE / "clip_text_labels.json").write_text(json.dumps(PROMPTS, indent=1))

    loader = DataLoader(Frames(df), batch_size=128, num_workers=min(32, os.cpu_count()), collate_fn=collate,
                        worker_init_fn=worker_init)
    all_stats, e_dino, e_clip, e_scene = [], [], [], []
    l2 = lambda t: torch.nn.functional.normalize(t.float(), dim=-1).cpu().numpy().astype(np.float16)
    for stats, x_dino, x_clip, x_scene in tqdm(loader, mininterval=10):
        all_stats += stats
        e_dino.append(l2(dino(x_dino.to(dev))))
        e_clip.append(l2(clip.encode_image(x_clip.to(dev))))
        e_scene.append(l2(dino(x_scene.to(dev))))

    pd.concat([df, pd.DataFrame(all_stats)], axis=1).to_parquet(CACHE / "meta_stats.parquet")
    np.save(CACHE / "emb_dino_crop.npy", np.concatenate(e_dino))
    np.save(CACHE / "emb_clip_crop.npy", np.concatenate(e_clip))
    np.save(CACHE / "emb_dino_scene.npy", np.concatenate(e_scene))
    print("done", len(df))


if __name__ == "__main__":
    main()
