"""CLIP image embeddings for annotated boxes: crop -> squash -> CLIP visual tower -> L2 normalisation."""
import cv2
import numpy as np
import open_clip
import torch
from open_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from torch.utils.data import DataLoader

from reid.data import BoxDataset


def pick_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def _worker_init(_):
    cv2.setNumThreads(1)


class ClipEmbedder:
    def __init__(self, name, pretrained, input_hw, device):
        model, _, _ = open_clip.create_model_and_transforms(name, pretrained=pretrained)
        preprocess = getattr(model.visual, "preprocess_cfg", None) or {}
        self.mean = preprocess.get("mean", OPENAI_DATASET_MEAN)
        self.std = preprocess.get("std", OPENAI_DATASET_STD)
        self.visual = model.visual.eval().to(device)  # the text tower is not used
        self.input_hw, self.device = input_hw, device

    @torch.no_grad()
    def embed_boxes(self, df, images_dir, max_side, batch_size, num_workers):
        """float32 L2-normalised embeddings aligned with the rows of df."""
        dataset = BoxDataset(df, images_dir, self.input_hw, self.mean, self.std, max_side)
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, worker_init_fn=_worker_init)
        chunks = [torch.nn.functional.normalize(self.visual(x.to(self.device)), dim=-1).cpu().numpy() for x in loader]
        return np.concatenate(chunks).astype(np.float32)
