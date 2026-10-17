"""Backbones and the BNNeck ReID head."""
import math
import hashlib
from pathlib import Path

import open_clip
import timm
import torch
import torch.nn.functional as F
from open_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from timm.data import resolve_data_config
from torch import nn

from reid.teachers import load_fastreid

from ._model_backbones import build_backbone
from ._model_checkpoints import checkpoint_name, interpolate_position_embedding, load_backbone
from ._model_heads import MarginClassifier, ReIDModel
from ._model_pooling import MultiQueryPool, apply_drop_path

# Keep existing full-model checkpoints and pickled callables on their original import path.
for _definition in (
    build_backbone, MultiQueryPool, apply_drop_path, checkpoint_name,
    interpolate_position_embedding, load_backbone, MarginClassifier, ReIDModel,
):
    _definition.__module__ = __name__
del _definition
