"""Backbones and the BNNeck ReID head."""
import open_clip
import timm
from open_clip.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from timm.data import resolve_data_config
from torch import nn


def build_backbone(cfg):
    """(module mapping images to a global feature, feature dim, normalisation mean, std) for a model config."""
    if cfg["backbone"] == "clip":
        model, _, _ = open_clip.create_model_and_transforms(cfg["name"], pretrained=cfg["pretrained"])
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


class ReIDModel(nn.Module):
    """Backbone + BNNeck. Training returns (feature for the triplet loss, ID logits); eval returns the embedding."""

    def __init__(self, backbone, feat_dim, n_classes):
        super().__init__()
        self.backbone = backbone
        self.bnneck = nn.BatchNorm1d(feat_dim)
        self.bnneck.bias.requires_grad_(False)
        self.classifier = nn.Linear(feat_dim, n_classes, bias=False)
        nn.init.normal_(self.classifier.weight, std=0.001)

    def forward(self, x):
        feat = self.backbone(x)
        emb = self.bnneck(feat)
        return (feat, self.classifier(emb)) if self.training else emb
