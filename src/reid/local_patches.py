"""Local patch branch next to the frozen global descriptor of the EXP-054 SigLIP 2 soup (EXP-083, EXP-084).

The global branch is EXP-054 unchanged: SigLIP 2 trunk -> its MAP pooling -> BNNeck -> L2 = g. The local branch reads
the trunk's patch tokens (16 x 16 at 224 px, after the final norm), before MAP pools them:
- type attention: R learnable soft masks (a linear map from each token to R logits, softmax over tokens) give R weighted
  poolings; one shared projection to region_dim; BNNeck over the concatenation; each region L2-normalised, concatenated
  and L2-normalised again = l. Trained with ArcFace + SupCon on the concatenation, the rest frozen;
- type mean: l = L2(mean of the patch tokens), no parameters - a zero-training baseline.
The retrieval embedding is concat(sqrt(1 - alpha) g, sqrt(alpha) l): the cosine of two embeddings is
(1 - alpha) cos(g) + alpha cos(l), and alpha = 0 gives the EXP-054 ranking exactly."""
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from reid.inference import load_trained_encoder
from reid.models import MarginClassifier


class LocalPatchHead(nn.Module):
    def __init__(self, dim, cfg, n_classes=None, head=None):
        super().__init__()
        self.kind = cfg["type"]
        if self.kind not in ("attention", "mean"):
            raise ValueError(f"unknown local branch {self.kind}")
        if self.kind == "attention":
            self.regions, self.region_dim = cfg["regions"], cfg["region_dim"]
            self.masks = nn.Linear(dim, self.regions)
            self.projection = nn.Linear(dim, self.region_dim)
            self.neck = nn.BatchNorm1d(self.regions * self.region_dim)
            self.neck.bias.requires_grad_(False)
            if n_classes:
                self.classifier = MarginClassifier(self.regions * self.region_dim, n_classes, head["type"],
                                                   head["scale"], head["margin"])

    def features(self, tokens):  # B x N x C -> B x R*region_dim, before the BNNeck
        weights = self.masks(tokens).softmax(1)  # each region's mask sums to 1 over the tokens
        return self.projection(torch.einsum("bnr,bnc->brc", weights, tokens)).flatten(1)

    def embedding(self, tokens):  # unit-length local vector l
        if self.kind == "mean":
            return F.normalize(tokens.float().mean(1), dim=1)
        emb = self.neck(self.features(tokens)).float().view(len(tokens), self.regions, self.region_dim)
        return F.normalize(F.normalize(emb, dim=2).flatten(1), dim=1)


def tokens_and_global(encoder, x):
    """(patch tokens after the trunk's norm, EXP-054 embedding before L2) for encoder = Sequential(visual, BNNeck)."""
    visual, neck = encoder
    tokens = visual.trunk.forward_features(x)
    return tokens, neck(visual.head(visual.trunk.forward_head(tokens)))


def combine(g, l, alpha):
    return torch.cat([math.sqrt(1 - alpha) * F.normalize(g.float(), dim=1), math.sqrt(alpha) * l.float()], 1)


class LocalPatchEncoder(nn.Module):
    """Inference: frozen EXP-054 + the local head, concatenated at model.local.alpha; for reid.benchmark."""

    def __init__(self, encoder, head, alpha):
        super().__init__()
        self.encoder, self.head, self.alpha = encoder, head, alpha

    def forward(self, x):
        tokens, g = tokens_and_global(self.encoder, x)
        return combine(g, self.head.embedding(tokens), self.alpha)


def load_base(model_cfg, checkpoint_dir, weights_dir=None):
    """The frozen EXP-054 soup (visual + BNNeck), checksum verified."""
    return load_trained_encoder({key: value for key, value in model_cfg.items() if key != "local"},
                                Path(checkpoint_dir) / model_cfg["base_checkpoint"], weights_dir,
                                model_cfg["base_sha256"])


def load_local_encoder(model_cfg, checkpoint, weights_dir=None):
    """reid.inference entry: base from the checkpoint's directory, head weights from the checkpoint itself."""
    encoder, dim, mean, std = load_base(model_cfg, Path(checkpoint).parent, weights_dir)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)["state_dict"]
    head = LocalPatchHead(dim, model_cfg["local"])
    head.load_state_dict({k: v for k, v in state.items() if not k.startswith("classifier.")})
    size = dim + (head.regions * head.region_dim if head.kind == "attention" else dim)
    return LocalPatchEncoder(encoder, head, model_cfg["local"]["alpha"]), size, mean, std
