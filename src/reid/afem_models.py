"""Global SigLIP2 AFEM and independent OSNet-AIN late fusion, from EXP-054.

AFEM transforms one vector into one vector. ReID heads/losses live outside it.
The same crop/augmentation feeds both towers, with each tower's own normalisation.
"""
import hashlib
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from reid.models import MarginClassifier, ReIDModel, build_backbone
from reid.paper_models import GeM, IMAGENET_MEAN, IMAGENET_STD
from reid.vendor.osnet_ain import osnet_ain_x1_0


KINDS = {"siglip_afem", "osnet_ain", "siglip_osnet"}


def verified_state(path, sha256):
    with Path(path).open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != sha256:
            raise ValueError(f"checksum mismatch: {path}")
    return torch.load(path, map_location="cpu", weights_only=True)


class SemanticAFEM(nn.Module):
    """Main projection + small residual weighted sum of G independent full-vector transforms.

    Each scalar gate is a global trainable parameter; softmax keeps the mixture scale bounded.
    Nonzero small residual initialisation lets every branch/gate receive gradients immediately.
    Groups are learned transformations, never contiguous slices or spatial/patch attention.
    """

    def __init__(self, in_dim, out_dim, groups=32, residual_init=0.001, base=None):
        super().__init__()
        if groups < 1 or residual_init <= 0:
            raise ValueError("positive groups and residual_init are required")
        self.in_dim, self.out_dim = in_dim, out_dim
        self.main = base if base is not None else (nn.Identity() if in_dim == out_dim else nn.Linear(in_dim, out_dim))
        self.branches = nn.ModuleList([
            nn.Sequential(nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), nn.GELU()) for _ in range(groups)
        ])
        self.group_logits = nn.Parameter(torch.zeros(groups))
        self.residual_scale = nn.Parameter(torch.tensor(float(residual_init)))

    def forward(self, x):
        if x.ndim != 2 or x.shape[1] != self.in_dim:
            raise ValueError(f"AFEM expects [B, {self.in_dim}], got {tuple(x.shape)}")
        weights = self.group_logits.softmax(0).to(x.dtype)
        refined = torch.stack([branch(x) for branch in self.branches], 1)
        main = self.main(x)
        residual = self.residual_scale.to(x.dtype) * (refined * weights[None, :, None]).sum(1)
        if self.training:
            self.last_residual_ratio = (residual.detach().float().norm(dim=1) /
                                        main.detach().float().norm(dim=1).clamp_min(1e-8)).mean()
        return main + residual


class OSNetAppearance(nn.Module):
    """Standard AIN x1.0 spatial map -> bounded GeM (or GAP) -> independent projection."""

    def __init__(self, cfg, weights_dir=None, load_pretrained=True):
        super().__init__()
        self.network = osnet_ain_x1_0(num_classes=1000, pretrained=False)
        if load_pretrained:
            state = verified_state(Path(weights_dir) / cfg["weights"], cfg["sha256"])
            # The author's ImageNet checkpoint was saved with DataParallel.
            if state and all(key.startswith("module.") for key in state):
                state = {key[len("module."):]: value for key, value in state.items()}
            self.network.load_state_dict(state)
        # These original heads are not on our feature-map path and must not be trained/stored.
        self.network.fc = nn.Identity()
        self.network.classifier = nn.Identity()
        pooling = cfg.get("pooling", "gem")
        if pooling not in {"gem", "gap"}:
            raise ValueError("OSNet pooling must be gem or gap")
        self.pool = GeM(p=cfg.get("gem_p", 3.0), min_p=1.0) if pooling == "gem" else nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Linear(512, cfg["dim"])
        self.dim = cfg["dim"]

    def forward(self, x):
        return self.projection(self.pool(self.network.featuremaps(x)).flatten(1))


def siglip_tower(cfg, load_pretrained):
    tower_cfg = {"backbone": "clip", **cfg["model"]}
    tower, dim, mean, std = build_backbone(tower_cfg, load_pretrained=False)
    if load_pretrained:
        state = verified_state(cfg["checkpoint"], cfg["checkpoint_sha256"])
        if state["exp_id"] != "EXP-054":
            raise ValueError("SigLIP source must be the selected EXP-054 soup")
        saved = state["config"]["model"]
        if any(saved[key] != tower_cfg[key] for key in ["name", "pretrained", "input_hw"]):
            raise ValueError("SigLIP architecture/preprocessing differs from EXP-054")
        tower.load_state_dict({k[len("backbone."):]: v for k, v in state["state_dict"].items()
                               if k.startswith("backbone.")})
    return tower, dim, mean, std


class SemanticBackbone(nn.Module):
    def __init__(self, cfg, load_pretrained=True):
        super().__init__()
        self.encoder, self.source_dim, self.mean, self.std = siglip_tower(cfg["semantic"], load_pretrained)
        self.dim = cfg["embedding_dim"]
        base = nn.Linear(self.source_dim, self.dim)
        self.refinement = SemanticAFEM(self.source_dim, self.dim, base=base, **cfg["afem"]) if cfg.get("afem") else base

    def forward(self, x):
        return self.refinement(self.encoder(x))

    def optimizer_groups(self, cfg):
        return [{"params": list(self.encoder.parameters()), "lr": cfg["lr_backbone"], "branch": "siglip"},
                {"params": list(self.refinement.parameters()), "lr": cfg["lr_head"], "branch": "semantic_projection_afem"}]


class AppearanceBackbone(OSNetAppearance):
    def optimizer_groups(self, cfg):
        return [{"params": list(self.network.parameters()), "lr": cfg["lr_backbone"], "branch": "osnet"},
                {"params": [*self.pool.parameters(), *self.projection.parameters()], "lr": cfg["lr_head"],
                 "branch": "appearance_pool_projection"}]


class LateFusionBackbone(nn.Module):
    def __init__(self, cfg, weights_dir=None, load_pretrained=True):
        super().__init__()
        self.encoder, self.source_dim, self.mean, self.std = siglip_tower(cfg["semantic"], load_pretrained)
        base = nn.Linear(self.source_dim, cfg["branch_dim"])
        self.appearance = OSNetAppearance(cfg["appearance"], weights_dir, load_pretrained)
        # Initialise common projections before AFEM, keeping the ablation's new-head initialisation aligned.
        self.refinement = SemanticAFEM(self.source_dim, cfg["branch_dim"], base=base, **cfg["afem"]) if cfg.get("afem") else base
        if self.appearance.dim != cfg["branch_dim"]:
            raise ValueError("Both fusion branches must have the configured branch_dim")
        self.dim = cfg["branch_dim"] * 2
        scale = torch.tensor(self.std) / torch.tensor(IMAGENET_STD)
        shift = (torch.tensor(self.mean) - torch.tensor(IMAGENET_MEAN)) / torch.tensor(IMAGENET_STD)
        self.register_buffer("appearance_scale", scale.view(1, 3, 1, 1), persistent=False)
        self.register_buffer("appearance_shift", shift.view(1, 3, 1, 1), persistent=False)

    def fused_and_appearance(self, x):
        semantic = self.refinement(self.encoder(x))
        appearance = self.appearance(x * self.appearance_scale + self.appearance_shift)
        if self.training:
            self.last_branch_norms = {"semantic": semantic.detach().float().norm(dim=1).mean(),
                                      "appearance": appearance.detach().float().norm(dim=1).mean()}
        # Float32 norms under autocast; each tower contributes a unit-length descriptor to simple concat.
        fused = torch.cat([F.normalize(semantic.float(), dim=1), F.normalize(appearance.float(), dim=1)], 1)
        return fused, appearance

    def forward(self, x):
        return self.fused_and_appearance(x)[0]

    def optimizer_groups(self, cfg):
        return [{"params": list(self.encoder.parameters()), "lr": cfg["lr_backbone"], "branch": "siglip"},
                {"params": list(self.appearance.network.parameters()), "lr": cfg.get("lr_appearance", cfg["lr_backbone"]),
                 "branch": "osnet"},
                {"params": [*self.refinement.parameters(), *self.appearance.pool.parameters(),
                            *self.appearance.projection.parameters()], "lr": cfg["lr_head"], "branch": "new_projections_afem_pool"}]


class FusionReIDModel(ReIDModel):
    """Main ReID head on concat; a separate OSNet head for auxiliary training only."""

    def __init__(self, backbone, n_classes, head):
        super().__init__(backbone, backbone.dim, n_classes, head)
        self.auxiliary_neck = nn.BatchNorm1d(backbone.appearance.dim)
        self.auxiliary_neck.bias.requires_grad_(False)
        self.auxiliary_classifier = MarginClassifier(backbone.appearance.dim, n_classes, head["type"], head["scale"], head["margin"])

    def forward(self, x, labels=None):
        feat, appearance = self.backbone.fused_and_appearance(x)
        emb = self.bnneck(feat)
        if not self.training:
            return emb
        return feat, self.classifier(emb, labels), appearance, self.auxiliary_classifier(self.auxiliary_neck(appearance), labels)

    def imprinting_features(self, x):
        feat, appearance = self.backbone.fused_and_appearance(x)
        return [self.bnneck(feat), self.auxiliary_neck(appearance)]


def build_extended_model(cfg, n_classes, weights_dir=None, load_pretrained=True, head=None):
    kind = cfg["backbone"]
    if kind == "siglip_afem":
        backbone = SemanticBackbone(cfg, load_pretrained)
        model = ReIDModel(backbone, backbone.dim, n_classes, head)
        mean, std = backbone.mean, backbone.std
    elif kind == "osnet_ain":
        backbone = AppearanceBackbone(cfg["appearance"], weights_dir, load_pretrained)
        model = ReIDModel(backbone, backbone.dim, n_classes, head)
        mean, std = IMAGENET_MEAN, IMAGENET_STD
    elif kind == "siglip_osnet":
        backbone = LateFusionBackbone(cfg, weights_dir, load_pretrained)
        model = FusionReIDModel(backbone, n_classes, head)
        mean, std = backbone.mean, backbone.std
    else:
        raise ValueError(f"Unknown AFEM/OSNet architecture: {kind}")
    return model, backbone.dim, mean, std
