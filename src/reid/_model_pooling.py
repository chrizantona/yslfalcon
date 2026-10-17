"""Attention pooling and stochastic depth for transformer towers."""
import torch
import torch.nn.functional as F
from torch import nn


class MultiQueryPool(nn.Module):
    """The tower's attention pooling with several queries instead of one, their outputs averaged.

    In SigLIP 2 the pretrained query lives in the bias of the q projection (norm 8.5; the latent itself is ~0), so
    queries that differ only in their latent would stay copies of it. Each query here is the pretrained one plus its
    own offset added after the projection; the first offset starts at zero, so query 0 is exactly the pretrained query.
    Offsets break the symmetry but do not by themselves make the queries look at different things: the attention maps
    are driven mostly by the tokens (a random offset twice the bias norm still leaves 85-89% of the map in place).
    What separates them is the penalty on their overlap, kept in self.overlap after every training forward: the mean
    cosine between the attention maps of different queries over heads and images, 1 when they coincide.
    The pooled feature keeps its size, so the head, the soup and the re-ranker are unchanged."""

    def __init__(self, pool, n_queries, offset_norm):
        super().__init__()
        if pool.pos_embed is not None or pool.latent.shape[1] != 1:
            raise ValueError("expected a single-query attention pool without a position embedding")
        # the pretrained modules under their own names: a checkpoint of the single-query tower loads as it is
        self.latent, self.q, self.kv, self.q_norm, self.k_norm = pool.latent, pool.q, pool.kv, pool.q_norm, pool.k_norm
        self.proj, self.proj_drop, self.norm, self.mlp = pool.proj, pool.proj_drop, pool.norm, pool.mlp
        self.num_heads, self.scale = pool.num_heads, pool.scale
        offsets = offset_norm * F.normalize(torch.randn(n_queries, pool.latent.shape[-1]), dim=-1)
        offsets[0] = 0
        self.query_offsets = nn.Parameter(offsets)
        self.overlap = None

    def forward(self, x):
        B, N, C = x.shape
        H = self.num_heads
        q = (self.q(self.latent[0]) + self.query_offsets).view(1, -1, H, C // H).transpose(1, 2)
        k, v = self.kv(x).reshape(B, N, 2, H, C // H).permute(2, 0, 3, 1, 4).unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        attn = ((q * self.scale) @ k.transpose(-2, -1)).softmax(-1)          # (B, H, queries, N)
        if self.training:
            maps = F.normalize(attn.float(), dim=-1)
            similarity = maps @ maps.transpose(-2, -1)                          # the diagonal is exactly 1
            n = similarity.shape[-1]
            self.overlap = ((similarity.sum((-2, -1)) - n) / (n * (n - 1))).mean()
        x = (attn @ v).transpose(1, 2).reshape(B, -1, C)
        x = self.proj_drop(self.proj(x))
        x = x + self.mlp(self.norm(x))
        return x.mean(1)


def apply_drop_path(trunk, rate):
    """Stochastic depth on a timm transformer tower: every block drops its residual branches with a probability that
    grows linearly from 0 at the first block to rate at the last, as timm does when built with drop_path_rate.

    Training only - at inference DropPath is the identity, so weights, speed and the state dict are unchanged."""
    from timm.layers import DropPath
    blocks = trunk.blocks
    for depth, block in enumerate(blocks):
        p = rate * depth / max(len(blocks) - 1, 1)
        for name in ("drop_path1", "drop_path2"):
            if not hasattr(block, name):
                raise ValueError(f"block {depth} has no {name}: this tower does not support stochastic depth")
            setattr(block, name, DropPath(p) if p > 0 else nn.Identity())
    return rate
