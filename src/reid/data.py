"""Config / annotation loading and box cropping shared by preprocessing, evaluation and inference."""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import Dataset


CSV_FILES = {"train": "train.csv", "query": "test_query.csv", "gallery": "test_gallery.csv"}


def load_yaml(path):
    return yaml.safe_load(Path(path).read_text())


def load_csv(dataset_dir, split):
    return pd.read_csv(Path(dataset_dir) / CSV_FILES[split])


def crop_box(frame, box, max_side=None):
    """Cut box = (x, y, w, h) in frame pixels out of an HxWxC frame; downsize so the longer side is <= max_side."""
    frame_h, frame_w = frame.shape[:2]
    x, y, w, h = (int(v) for v in box)
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, frame_w), min(y + h, frame_h)
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"box {box} does not intersect a {frame_w}x{frame_h} frame")
    crop = frame[y0:y1, x0:x1]
    longer = max(crop.shape[:2])
    if max_side and longer > max_side:
        scale = max_side / longer
        size = (max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale)))
        crop = cv2.resize(crop, size, interpolation=cv2.INTER_AREA)
    return crop


def to_model_input(crop_bgr, input_hw, mean, std):
    """Squash a BGR crop to input_hw and return a normalised RGB CHW float tensor."""
    h, w = input_hw
    upscale = crop_bgr.shape[0] < h or crop_bgr.shape[1] < w
    resized = cv2.resize(crop_bgr, (w, h), interpolation=cv2.INTER_LINEAR if upscale else cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    # mean/std may arrive as plain tuples; without the cast numpy would promote the image to float64
    normalised = (rgb - np.asarray(mean, np.float32)) / np.asarray(std, np.float32)
    return torch.from_numpy(normalised.transpose(2, 0, 1).copy())


class BoxDataset(Dataset):
    """Rows with image_id, x, y, w, h -> model inputs of the boxed vehicle, cropped from the full frame."""

    def __init__(self, df, images_dir, input_hw, mean, std, max_side):
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.input_hw, self.max_side = tuple(input_hw), max_side
        self.mean, self.std = np.asarray(mean, np.float32), np.asarray(std, np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        path = self.images_dir / f"{r.image_id}.jpg"
        frame = cv2.imread(str(path))
        if frame is None:
            raise FileNotFoundError(path)
        crop = crop_box(frame, (r.x, r.y, r.w, r.h), self.max_side)
        return to_model_input(crop, self.input_hw, self.mean, self.std)
