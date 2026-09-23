"""Offline model adapters. The preview descriptor never makes identity claims."""

from io import BytesIO

import numpy as np
from PIL import Image

from .core import FeatureBundle, normalise
from .images import crop_image


class PreviewEncoder:
    mode = "preview"
    model_id = "preview-rgb-layout-v1-NOT-REID"
    dimension = 192
    rerank_window = 0
    calibration_policy = None
    real_model_connected = False
    device = "cpu"
    precision = "float32"
    weights_bytes = 0

    def encode(self, image: bytes, bbox=None):
        with Image.open(BytesIO(crop_image(image, bbox))) as frame:
            values = np.asarray(frame.resize((8, 8), Image.Resampling.BILINEAR), dtype=np.float32)
        # Positive floor gives even a fully black image a defined unit descriptor.
        return FeatureBundle(normalise((values / 255.0 + 0.01).ravel()), None, self.model_id)

    def rerank(self, query, candidates, similarities):
        return np.asarray(similarities, dtype=np.float32).copy()


class RepositoryEncoder:
    """Reserved adapter boundary. Real model integration is intentionally disabled."""

    def __init__(self, manifest_path: str, device="cpu"):
        raise RuntimeError(
            "Real model integration is disabled. Use the explicit preview encoder; "
            "a trained adapter can be connected through the inference port later."
        )


def redistribute_scores(similarities, logits, weight):
    """Same rank blending and value permutation as reid.rerank_pairs.reorder."""
    similarities, logits = np.asarray(similarities), np.asarray(logits)
    if similarities.shape != logits.shape or similarities.ndim != 1 or not np.isfinite(logits).all():
        raise RuntimeError("Invalid pairwise logits")
    encoder_rank = np.argsort(np.argsort(-similarities))
    pair_rank = np.argsort(np.argsort(-logits))
    mixed = (1 - weight) * encoder_rank + weight * pair_rank
    places = np.argsort(np.argsort(mixed, kind="stable"))
    return np.sort(similarities)[::-1][places].astype(np.float32)
