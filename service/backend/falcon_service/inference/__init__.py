"""Stable inference port shared by HTTP, workers and the batch CLI."""

from .core import (
    Encoder,
    ExactIndex,
    FeatureBundle,
    calibration_needs_population,
    rank_candidates,
    rank_gallery,
)
from .encoders import PreviewEncoder, RepositoryEncoder
from .images import crop_image, validate_image
from .remote import RemoteEncoder

__all__ = [
    "Encoder",
    "FeatureBundle",
    "ExactIndex",
    "PreviewEncoder",
    "RepositoryEncoder",
    "RemoteEncoder",
    "calibration_needs_population",
    "rank_candidates",
    "rank_gallery",
    "validate_image",
    "crop_image",
]
