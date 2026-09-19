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
        if cfg.get("interpolate_positional_embedding"):
            kwargs["force_image_size"] = tuple(cfg["input_hw"])
        model, _, _ = open_clip.create_model_and_transforms(cfg["name"], pretrained=pretrained, **kwargs)
        preprocess = getattr(model.visual, "preprocess_cfg", None) or {}
        mean, std = preprocess.get("mean", OPENAI_DATASET_MEAN), preprocess.get("std", OPENAI_DATASET_STD)
        # timm-based towers (EVA02, SigLIP 2) have no output_dim; their projection ends at the model's embed_dim
        dim = getattr(model.visual, "output_dim", None) or open_clip.get_model_config(cfg["name"])["embed_dim"]
        return model.visual, dim, mean, std
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
    if interpolate_position:
        backbone["positional_embedding"] = interpolate_position_embedding(
            backbone["positional_embedding"], model.backbone.positional_embedding)
    model.backbone.load_state_dict(backbone)


class MarginClassifier(nn.Module):
    """Cosine classifier with a margin on the target class: CosFace s * (cos - m), ArcFace s * cos(theta + m).

    Works on the same L2-normalised embedding that retrieval compares, so the ID loss optimises the retrieval metric.
    sub_centers > 1 gives every class that many centres and scores it by its closest one (sub-centre ArcFace, Deng et
    al., ECCV 2020): a vehicle seen from the front and from the rear need not fit into one direction."""

    def __init__(self, feat_dim, n_classes, kind, scale, margin, sub_centers=1):
        super().__init__()
        assert kind in ("arcface", "cosface")
        self.kind, self.scale, self.margin = kind, scale, margin
        self.n_classes, self.sub_centers = n_classes, sub_centers
        self.weight = nn.Parameter(torch.empty(n_classes * sub_centers, feat_dim).normal_(std=0.01))

    def forward(self, emb, labels):
        with torch.autocast(emb.device.type, enabled=False):  # float32: bfloat16 cosines near 1 are too coarse for margins
            cos = F.linear(F.normalize(emb.float(), dim=1), F.normalize(self.weight.float(), dim=1)).clamp(-1, 1)
            if self.sub_centers > 1:  # centres are stored class-major: weight[c * sub_centers + j]
                cos = cos.view(len(cos), self.n_classes, self.sub_centers).amax(2)
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

    head: {"type": "linear"} (default) or {"type": "arcface" | "cosface", "scale": s, "margin": m, "sub_centers": k}."""

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
            self.classifier = MarginClassifier(feat_dim, n_classes, head["type"], head["scale"], head["margin"],
                                               head.get("sub_centers", 1))

    def forward(self, x, labels=None):
        feat = self.backbone(x)
        emb = self.bnneck(feat)
        if not self.training:
            return emb
        if isinstance(self.classifier, MarginClassifier):  # the margin applies to the target class only
            return feat, self.classifier(emb, labels)
        return feat, self.classifier(emb)
