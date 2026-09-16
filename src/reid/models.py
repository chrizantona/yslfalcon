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


def build_backbone(cfg, external_weights_dir=None, load_pretrained=True):
    """(module mapping images to a global feature, feature dim, normalisation mean, std) for a model config."""
    if cfg["backbone"] == "fastreid_r50ibn":  # public FastReID checkpoint as the start; its BN head is dropped
        model, feat_dim, mean, std = load_fastreid(Path(external_weights_dir) / cfg["weights"])
        model.bottleneck = None  # ReIDModel adds a fresh BNNeck, as for the other backbones
        return model, feat_dim, mean, std
    if cfg["backbone"] == "clip":
        pretrained = cfg["pretrained"]
        kwargs = {}
        if cfg.get("weights") or not load_pretrained:
            metadata = open_clip.get_pretrained_cfg(cfg["name"], pretrained)
            kwargs = {f"image_{key}": metadata[key] for key in ("mean", "std", "interpolation", "resize_mode")
                      if key in metadata}
        if cfg.get("weights") and load_pretrained:
            path = Path(external_weights_dir) / cfg["weights"]
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != cfg["sha256"]:
                raise ValueError(f"checksum mismatch: {path}")
            # A local checkpoint does not carry the original tag's preprocessing metadata.
            pretrained = str(path)
        if not load_pretrained:
            pretrained = None  # a complete trained state follows; no extra public weights at inference
        model, _, _ = open_clip.create_model_and_transforms(cfg["name"], pretrained=pretrained, **kwargs)
        preprocess = getattr(model.visual, "preprocess_cfg", None) or {}
        mean, std = preprocess.get("mean", OPENAI_DATASET_MEAN), preprocess.get("std", OPENAI_DATASET_STD)
        return model.visual, model.visual.output_dim, mean, std
    if cfg["backbone"] == "timm":
        model = timm.create_model(cfg["name"], pretrained=True, num_classes=0)
        if cfg.get("last_stride_1"):  # keep a 2x larger final feature map, as in the ReID bag of tricks
            block = model.layer4[0]
            block.conv2.stride = (1, 1)
            block.downsample[0].stride = (1, 1)
        data_cfg = resolve_data_config({}, model=model)
        return model, model.num_features, data_cfg["mean"], data_cfg["std"]
    raise ValueError(f"unknown backbone {cfg['backbone']}")


def checkpoint_name(cfg, seed=None):
    """Optional readable checkpoint name; legacy experiments keep their random names."""
    import uuid
    name = cfg.get("checkpoint_name", f"{uuid.uuid4().hex}.pt").format(seed=seed)
    if Path(name).name != name or not name.endswith(".pt"):
        raise ValueError("checkpoint_name must be a .pt filename without a directory")
    return name


def load_backbone(model, checkpoint_path):
    """Copy the backbone weights of a reid.train checkpoint into model; the head (BNNeck, classifier) is left as is."""
    state = torch.load(checkpoint_path, map_location="cpu")["state_dict"]
    model.backbone.load_state_dict({k[len("backbone."):]: v for k, v in state.items() if k.startswith("backbone.")})


class MarginClassifier(nn.Module):
    """Cosine classifier with a margin on the target class: CosFace s * (cos - m), ArcFace s * cos(theta + m).

    Works on the same L2-normalised embedding that retrieval compares, so the ID loss optimises the retrieval metric."""

    def __init__(self, feat_dim, n_classes, kind, scale, margin):
        super().__init__()
        assert kind in ("arcface", "cosface")
        self.kind, self.scale, self.margin = kind, scale, margin
        self.weight = nn.Parameter(torch.empty(n_classes, feat_dim).normal_(std=0.01))

    def forward(self, emb, labels):
        with torch.autocast(emb.device.type, enabled=False):  # float32: bfloat16 cosines near 1 are too coarse for margins
            cos = F.linear(F.normalize(emb.float(), dim=1), F.normalize(self.weight.float(), dim=1)).clamp(-1, 1)
            target = cos.gather(1, labels[:, None])
            if self.kind == "cosface":
                target_m = target - self.margin
            else:
                sin = (1 - target ** 2).clamp(min=1e-6).sqrt()
                shifted = target * math.cos(self.margin) - sin * math.sin(self.margin)
                # past theta = pi - m, cos(theta + m) would grow again; keep the penalty monotonic as in insightface
                target_m = torch.where(target > math.cos(math.pi - self.margin), shifted,
                                       target - math.sin(math.pi - self.margin) * self.margin)
            return self.scale * cos.scatter(1, labels[:, None], target_m)


class ReIDModel(nn.Module):
    """Backbone + BNNeck. Training returns (feature for the triplet loss, ID logits); eval returns the embedding.

    head: {"type": "linear"} (default) or {"type": "arcface" | "cosface", "scale": s, "margin": m}."""

    def __init__(self, backbone, feat_dim, n_classes, head=None):
        super().__init__()
        head = head or {"type": "linear"}
        self.backbone = backbone
        self.bnneck = nn.BatchNorm1d(feat_dim)
        self.bnneck.bias.requires_grad_(False)
        if head["type"] == "linear":
            self.classifier = nn.Linear(feat_dim, n_classes, bias=False)
            nn.init.normal_(self.classifier.weight, std=0.001)
        else:
            self.classifier = MarginClassifier(feat_dim, n_classes, head["type"], head["scale"], head["margin"])

    def forward(self, x, labels=None):
        feat = self.backbone(x)
        emb = self.bnneck(feat)
        if not self.training:
            return emb
        if isinstance(self.classifier, MarginClassifier):  # the margin applies to the target class only
            return feat, self.classifier(emb, labels)
        return feat, self.classifier(emb)
