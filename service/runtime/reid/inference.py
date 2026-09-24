"""Load a complete trained visual encoder and BNNeck without fetching its original public weights."""
import hashlib

import torch

from reid.models import build_backbone


def load_trained_encoder(model_cfg, checkpoint, external_weights_dir=None, sha256=None):
    if sha256:
        with open(checkpoint, "rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != sha256:
                raise ValueError(f"checkpoint checksum mismatch: {checkpoint}")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = saved["state_dict"]
    backbone, dim, mean, std = build_backbone(model_cfg, external_weights_dir, load_pretrained=False)
    backbone.load_state_dict({k[len("backbone."):]: v for k, v in state.items() if k.startswith("backbone.")})
    neck = torch.nn.BatchNorm1d(dim)
    neck.load_state_dict({k[len("bnneck."):]: v for k, v in state.items() if k.startswith("bnneck.")})
    return torch.nn.Sequential(backbone, neck), dim, mean, std
