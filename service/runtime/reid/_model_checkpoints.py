"""Checkpoint naming and compatible backbone weight loading."""
import math
from pathlib import Path

import torch
import torch.nn.functional as F


def checkpoint_name(cfg, seed=None):
    """Optional readable checkpoint name; legacy experiments keep their random names."""
    import uuid
    name = cfg.get("checkpoint_name", f"{uuid.uuid4().hex}.pt").format(seed=seed)
    if Path(name).name != name or not name.endswith(".pt"):
        raise ValueError("checkpoint_name must be a .pt filename without a directory")
    return name


def interpolate_position_embedding(position, target):
    """Bicubic patch-grid interpolation; preserve the class token exactly."""
    if position.shape == target.shape:
        return position
    old_side, new_side = math.isqrt(position.shape[0] - 1), math.isqrt(target.shape[0] - 1)
    if (position.ndim != 2 or target.ndim != 2 or position.shape[1] != target.shape[1]
            or old_side ** 2 != position.shape[0] - 1 or new_side ** 2 != target.shape[0] - 1):
        raise ValueError("positional embeddings require square patch grids and one class token")
    grid = position[1:].float().T.reshape(1, position.shape[1], old_side, old_side)
    grid = F.interpolate(grid, size=(new_side, new_side), mode="bicubic", align_corners=False, antialias=True)
    return torch.cat([position[:1], grid.reshape(position.shape[1], -1).T.to(position.dtype)])


def load_backbone(model, checkpoint_path, interpolate_position=False):
    """Copy the backbone weights of a reid.train checkpoint into model; the head (BNNeck, classifier) is left as is."""
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)["state_dict"]
    backbone = {k[len("backbone."):]: v for k, v in state.items() if k.startswith("backbone.")}
    for key, value in model.backbone.state_dict().items():
        if key.endswith("query_offsets"):
            backbone.setdefault(key, value)  # a single-query pretraining: the offsets keep their initialisation
    if interpolate_position:
        backbone["positional_embedding"] = interpolate_position_embedding(
            backbone["positional_embedding"], model.backbone.positional_embedding)
    model.backbone.load_state_dict(backbone)
