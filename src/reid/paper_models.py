"""Two published vehicle ReID architectures, reproduced at a colleague's request (EXP-055 and later).

clip_senet (CLIP-SENet): ResNeXt101-IBN-a appearance branch (T^a, 2048) and the TinyCLIP ViT-B/32 image encoder
(semantics T^s, 512) side by side; T^u = FC([T^s, T^a]) (2048), AFEM refines T^s, T = T^u + T^s'. One classifier on T.
convnext_glpe (ConvNeXt-L + OT-GFAM + GLPE): ConvNeXt-L (ImageNet-22k) up to stage 3 (768 x H/16 x W/16) -> windowed
and shifted-window gated attention with ONN projections -> GeM over the whole map and over 2 and 3 horizontal stripes:
six embeddings with their own classifiers; the retrieval embedding concatenates the six normalised ones (6 x 768).

Training mode returns (embeddings, logits), one entry per supervised embedding; eval mode returns the retrieval
embedding. Inputs arrive with the ImageNet mean/std; the TinyCLIP branch renormalises to CLIP's."""
import hashlib
from functools import partial
from pathlib import Path

import timm
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from timm.layers import resample_abs_pos_embed
from timm.models.vision_transformer import VisionTransformer
from torch import nn

from reid.ibn_resnext import ResNeXt101IBNa

IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
CLIP_MEAN, CLIP_STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)


def load_verified(weights_dir, entry):
    """State dict of a public weights file listed in the config as {file, sha256}."""
    path = Path(weights_dir) / entry["file"]
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != entry["sha256"]:
            raise ValueError(f"checksum mismatch: {path}")
    if path.suffix == ".safetensors":
        return load_file(path)
    return torch.load(path, map_location="cpu", weights_only=True)


def tinyclip_state_for_timm(state, grid):
    """HF CLIPVisionModel keys -> timm VisionTransformer keys; q/k/v are fused, positions resampled to grid."""
    out, prefix = {}, "vision_model."
    old_grid = int((state[prefix + "embeddings.position_embedding.weight"].shape[0] - 1) ** 0.5)
    out["cls_token"] = state[prefix + "embeddings.class_embedding"][None, None]
    out["patch_embed.proj.weight"] = state[prefix + "embeddings.patch_embedding.weight"]
    out["pos_embed"] = resample_abs_pos_embed(state[prefix + "embeddings.position_embedding.weight"][None],
                                              new_size=grid, old_size=(old_grid, old_grid), num_prefix_tokens=1)
    for ours, theirs in (("norm_pre", "pre_layrnorm"), ("norm", "post_layernorm")):
        for kind in ("weight", "bias"):
            out[f"{ours}.{kind}"] = state[f"{prefix}{theirs}.{kind}"]
    layers = {int(key.split(".")[3]) for key in state if key.startswith(prefix + "encoder.layers.")}
    for i in sorted(layers):
        src = f"{prefix}encoder.layers.{i}."
        for kind in ("weight", "bias"):
            out[f"blocks.{i}.attn.qkv.{kind}"] = torch.cat([state[f"{src}self_attn.{p}_proj.{kind}"] for p in "qkv"])
            for ours, theirs in (("norm1", "layer_norm1"), ("norm2", "layer_norm2"),
                                 ("attn.proj", "self_attn.out_proj"), ("mlp.fc1", "mlp.fc1"), ("mlp.fc2", "mlp.fc2")):
                out[f"blocks.{i}.{ours}.{kind}"] = state[f"{src}{theirs}.{kind}"]
    return out


class AFEM(nn.Module):
    """Adaptive fine-grained enhancement: Linear -> BN -> ReLU gives G+1 vectors; the G group vectors, weighted by
    learnable w_i (standard normal init), gate the kept vector: T^s' = f + f * sum_i(w_i * g_i). This is our reading of
    the paper's text; its formula as printed (sum of w_i * f(T^s)) collapses to a fixed per-channel scale."""

    def __init__(self, in_dim, dim, groups):
        super().__init__()
        self.groups, self.dim = groups, dim
        self.linear = nn.Sequential(nn.Linear(in_dim, (groups + 1) * dim), nn.BatchNorm1d((groups + 1) * dim),
                                    nn.ReLU(inplace=True))
        self.weights = nn.Parameter(torch.randn(groups, dim))

    def forward(self, ts):
        f = self.linear(ts).view(len(ts), self.groups + 1, self.dim)
        kept = f[:, 0]
        return kept + kept * (self.weights * f[:, 1:]).sum(1)


class CLIPSENet(nn.Module):
    def __init__(self, cfg, n_classes, weights_dir=None, load_pretrained=True):
        super().__init__()
        s, (h, w) = cfg["semantic"], cfg["input_hw"]
        self.cnn = ResNeXt101IBNa()
        self.clip = VisionTransformer(
            img_size=(h, w), patch_size=s["patch"], embed_dim=s["width"], depth=s["depth"], num_heads=s["heads"],
            mlp_ratio=s["mlp"] / s["width"], pre_norm=True, num_classes=0, global_pool="token", act_layer=nn.GELU,
            norm_layer=partial(nn.LayerNorm, eps=1e-5))
        self.clip_proj = nn.Linear(s["width"], s["dim"], bias=False)
        if load_pretrained:
            cnn = load_verified(weights_dir, cfg["appearance"])
            self.cnn.load_state_dict({k: v for k, v in cnn.items() if not k.startswith("fc.")})
            tiny = load_verified(weights_dir, s)
            self.clip.load_state_dict(tinyclip_state_for_timm(tiny, (h // s["patch"], w // s["patch"])))
            self.clip_proj.weight.data.copy_(tiny["visual_projection.weight"])
        dim = cfg["embedding_dim"]
        self.fuse = nn.Linear(s["dim"] + 2048, dim)
        self.afem = AFEM(s["dim"], dim, cfg["afem_groups"])
        self.classifiers = nn.ModuleList([nn.Linear(dim, n_classes)])
        scale = torch.tensor(IMAGENET_STD) / torch.tensor(CLIP_STD)
        self.register_buffer("clip_scale", scale.view(1, 3, 1, 1), persistent=False)
        self.register_buffer("clip_shift", ((torch.tensor(IMAGENET_MEAN) - torch.tensor(CLIP_MEAN)) /
                                            torch.tensor(CLIP_STD)).view(1, 3, 1, 1), persistent=False)

    def forward(self, x):
        ta = self.cnn(x).mean((2, 3))
        ts = self.clip_proj(self.clip(x * self.clip_scale + self.clip_shift))
        t = self.fuse(torch.cat([ts, ta], 1)) + self.afem(ts)
        if not self.training:
            return t
        return [t], [self.classifiers[0](t)]


class ONN(nn.Module):
    """Operational layer: tanh(sum_{q=1..Q} W_q x^q + b), element-wise powers; the bias is the q = 0 term."""

    def __init__(self, dim, order):
        super().__init__()
        self.powers = nn.ModuleList([nn.Linear(dim, dim, bias=q == 1) for q in range(1, order + 1)])

    def forward(self, x):
        return torch.tanh(sum(layer(x ** q) for q, layer in enumerate(self.powers, 1)))


def shifted_window_mask(h, w, window, shift, device):
    """Swin mask for the cyclically shifted map, windows x tokens x tokens: True where attention is allowed, i.e.
    both tokens come from the same image region rather than from opposite edges wrapped together by the shift."""
    region = torch.zeros(h, w, device=device)
    label = 0
    for rows in (slice(0, -window), slice(-window, -shift), slice(-shift, None)):
        for cols in (slice(0, -window), slice(-window, -shift), slice(-shift, None)):
            region[rows, cols] = label
            label += 1
    windows = region.view(h // window, window, w // window, window).permute(0, 2, 1, 3).reshape(-1, window * window)
    return windows[:, :, None] == windows[:, None, :]


class GFAM(nn.Module):
    """Gated window attention: a global gate (GAP of the OT-GFAM input -> MLP -> sigmoid) scales every window's
    tokens, Q/K/V come from ONN layers, attention runs inside 7x7 windows, cyclically shifted by (-3, -3) before and
    (+3, +3) after in the SW unit."""

    def __init__(self, dim, heads, window, shift, onn_order):
        super().__init__()
        self.heads, self.window, self.shift = heads, window, shift
        self.gate = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim), nn.Sigmoid())
        self.q, self.k, self.v = (ONN(dim, onn_order) for _ in range(3))

    def forward(self, x_norm, block_input):  # B x H x W x C
        b, h, w, c = x_norm.shape
        ws, n_windows = self.window, (h // self.window) * (w // self.window)
        gate = self.gate(block_input.mean((1, 2)))
        x = torch.roll(x_norm, (-self.shift, -self.shift), (1, 2)) if self.shift else x_norm
        x = x.view(b, h // ws, ws, w // ws, ws, c).permute(0, 1, 3, 2, 4, 5).reshape(b, n_windows, ws * ws, c)
        x = x * gate[:, None, None, :]

        def split(t):  # B x nW x T x C -> B x nW x heads x T x C/heads
            return t.view(b, n_windows, ws * ws, self.heads, c // self.heads).transpose(2, 3)

        mask = shifted_window_mask(h, w, ws, self.shift, x.device)[None, :, None] if self.shift else None
        out = F.scaled_dot_product_attention(split(self.q(x)), split(self.k(x)), split(self.v(x)), attn_mask=mask)
        out = out.transpose(2, 3).reshape(b, h // ws, w // ws, ws, ws, c).permute(0, 1, 3, 2, 4, 5).reshape(b, h, w, c)
        return torch.roll(out, (self.shift, self.shift), (1, 2)) if self.shift else out


class GFAMUnit(nn.Module):
    """Pre-norm residual unit: x + GFAM(LN(x)) + MLP(LN(x)), then LayerNorm; the gate of both units comes from the
    input of the OT-GFAM block."""

    def __init__(self, dim, heads, window, shift, onn_order):
        super().__init__()
        self.norm_in, self.norm_out = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = GFAM(dim, heads, window, shift, onn_order)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x, block_input):
        n = self.norm_in(x)
        return self.norm_out(x + self.attn(n, block_input) + self.mlp(n))


class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p, self.eps = nn.Parameter(torch.tensor(p)), eps

    def forward(self, x):  # B x C x h x w -> B x C
        return x.clamp(min=self.eps).pow(self.p).mean((2, 3)).pow(1 / self.p)


class ConvNeXtGLPE(nn.Module):
    STRIPES = ((7, 7), (5, 5, 4))  # rows per part of the 14-row stage-3 map at 224 px: 2 parts, 3 parts

    def __init__(self, cfg, n_classes, weights_dir=None, load_pretrained=True):
        super().__init__()
        convnext = timm.create_model("convnext_large", pretrained=False, num_classes=0)
        if load_pretrained:
            convnext.load_state_dict({k: v for k, v in load_verified(weights_dir, cfg["backbone_weights"]).items()
                                      if not k.startswith("head.fc.")})
        self.features = nn.Sequential(convnext.stem, *convnext.stages[:3])  # stage 3: 768 x H/16 x W/16
        # model.stage3_bn (EXP-059): BatchNorm over the stage-3 channels. Two of the 768 channels of the ImageNet-22k
        # model carry mean |x| ~1200 and ~680 against a median of 7, so as described the embeddings of all vehicles are
        # the same (cos 0.996 at the start, 1.000 in training) and the loss stays at chance (EXP-057)
        self.stage_norm = nn.BatchNorm2d(768) if cfg.get("stage3_bn") else nn.Identity()
        dim, g = 768, cfg["gfam"]
        # one OT-GFAM block: W-GFAM, then SW-GFAM (shift window // 2 = 3)
        self.units = nn.ModuleList([GFAMUnit(dim, g["heads"], g["window"], shift, g["onn_order"])
                                    for shift in (0, g["window"] // 2)])
        self.pools = nn.ModuleList([GeM() for _ in range(6)])
        self.classifiers = nn.ModuleList([nn.Linear(dim, n_classes, bias=False) for _ in range(6)])

    def forward(self, x):
        block_input = self.stage_norm(self.features(x)).permute(0, 2, 3, 1)  # B x 14 x 14 x 768
        f = block_input
        for unit in self.units:
            f = unit(f, block_input)
        f = f.permute(0, 3, 1, 2)  # F_ot, B x 768 x 14 x 14
        parts = [f, *torch.split(f, self.STRIPES[0], 2), *torch.split(f, self.STRIPES[1], 2)]
        vectors = [pool(part) for pool, part in zip(self.pools, parts)]
        embeddings = [F.normalize(v, dim=1) for v in vectors]  # global, 2 parts, 3 parts
        if not self.training:  # the six normalised embeddings concatenated, 6 x 768
            return torch.cat(embeddings, 1)
        return embeddings, [head(v) for head, v in zip(self.classifiers, vectors)]


ARCHS = {"clip_senet": CLIPSENet, "convnext_glpe": ConvNeXtGLPE}


def build_paper_model(cfg, n_classes, weights_dir=None, load_pretrained=True):
    """(model, embedding dim, mean, std) for a config with model.arch."""
    model = ARCHS[cfg["arch"]](cfg, n_classes, weights_dir, load_pretrained)
    return model, cfg.get("embedding_dim", 6 * 768), IMAGENET_MEAN, IMAGENET_STD
