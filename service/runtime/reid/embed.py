"""Embeddings for annotated boxes: crop from the full frame -> squash -> encoder -> L2 normalisation."""
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from reid.data import BoxDataset
from reid.models import build_backbone


def pick_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _worker_init(_):
    cv2.setNumThreads(1)


class FlipTTA(torch.nn.Module):
    """Mean of normalised embeddings of one frame and its reflection; queries remain independent."""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, x):
        original = F.normalize(self.encoder(x).float(), dim=-1)
        reflected = F.normalize(self.encoder(x.flip(-1)).float(), dim=-1)
        return F.normalize((original + reflected) * 0.5, dim=-1)


@torch.no_grad()
def embed_boxes(encoder, df, images_dir, input_hw, mean, std, max_side, batch_size, num_workers, device, flip_tta=False):
    """float32 L2-normalised encoder embeddings aligned with the rows of df."""
    dataset = BoxDataset(df, images_dir, input_hw, mean, std, max_side)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, worker_init_fn=_worker_init)
    if flip_tta:
        encoder = FlipTTA(encoder).eval()
    chunks = [F.normalize(encoder(x.to(device)).float(), dim=-1).cpu().numpy() for x in loader]
    return np.concatenate(chunks).astype(np.float32)


class ClipEmbedder:
    """Zero-shot CLIP visual tower."""

    def __init__(self, name, pretrained, input_hw, device):
        visual, _, self.mean, self.std = build_backbone({"backbone": "clip", "name": name, "pretrained": pretrained})
        self.visual = visual.eval().to(device)
        self.input_hw, self.device = input_hw, device

    def embed_boxes(self, df, images_dir, max_side, batch_size, num_workers):
        return embed_boxes(self.visual, df, images_dir, self.input_hw, self.mean, self.std, max_side,
                           batch_size, num_workers, self.device)
