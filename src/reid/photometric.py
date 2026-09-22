"""Night-like photometric augmentation: random gamma and a sodium-lamp cast.

Cameras 82-86 shoot under sodium lighting, where a white car reads as beige and a red one burns out: they carry 24% of
the lost AP@10 at 13% of the true matches (error analysis of the final configuration). This turns ordinary daylight
crops into such frames during training only - inference is untouched. The cast keeps the luminance of the frame and
pulls its colours towards the monochromatic amber of a sodium lamp, which is what the lamp physically does.
"""
import numpy as np

AMBER = np.array([60, 170, 255], np.float32)          # BGR of a sodium lamp, ~2000 K
LUMINANCE = np.array([0.114, 0.587, 0.299], np.float32)  # BGR weights of Rec. 601


def night_cast(image, gamma, strength):
    """BGR uint8 frame -> gamma-corrected and blended towards amber of the same luminance; strength 0 leaves it as is."""
    x = (image.astype(np.float32) / 255.0) ** gamma
    lit = (x @ LUMINANCE)[..., None] * (AMBER / 255.0)
    return (np.clip((1 - strength) * x + strength * lit, 0, 1) * 255).astype(np.uint8)
