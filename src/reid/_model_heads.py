"""Margin classifiers and the BNNeck ReID head."""
import math

import torch
import torch.nn.functional as F
from torch import nn


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
