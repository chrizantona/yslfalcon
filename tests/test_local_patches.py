import math

import torch
import torch.nn.functional as F
from timm.models.vision_transformer import VisionTransformer
from torch import nn

from reid.local_patches import LocalPatchHead, combine, tokens_and_global


class Visual(nn.Module):  # the shape of open_clip's timm tower: trunk, then head
    def __init__(self):
        super().__init__()
        self.trunk = VisionTransformer(img_size=32, patch_size=8, embed_dim=16, depth=1, num_heads=2,
                                       global_pool="map", class_token=False, num_classes=0)
        self.head = nn.Sequential()

    def forward(self, x):
        return self.head(self.trunk(x))


def test_global_branch_is_the_unchanged_encoder_and_tokens_are_the_patches():
    torch.manual_seed(0)
    encoder = nn.Sequential(Visual(), nn.BatchNorm1d(16)).eval()
    x = torch.randn(3, 3, 32, 32)
    with torch.no_grad():
        tokens, g = tokens_and_global(encoder, x)
        assert tokens.shape == (3, 16, 16)
        assert torch.allclose(g, encoder(x), atol=1e-6)


def test_attention_head_gives_unit_regions_and_a_unit_local_vector():
    torch.manual_seed(0)
    head = LocalPatchHead(16, {"type": "attention", "regions": 4, "region_dim": 8}, n_classes=5,
                          head={"type": "arcface", "scale": 30.0, "margin": 0.5}).eval()
    tokens = torch.randn(3, 16, 16)
    assert head.features(tokens).shape == (3, 32)
    l = head.embedding(tokens)
    assert torch.allclose(l.norm(dim=1), torch.ones(3), atol=1e-6)
    assert torch.allclose(l.view(3, 4, 8).norm(dim=2), torch.full((3, 4), 0.5), atol=1e-6)  # regions weigh the same
    mean = LocalPatchHead(16, {"type": "mean"})
    assert torch.allclose(mean.embedding(tokens), F.normalize(tokens.mean(1), dim=1))


def test_combined_cosine_is_the_weighted_sum_and_alpha_zero_is_the_global_ranking():
    torch.manual_seed(0)
    g, l = torch.randn(4, 6), F.normalize(torch.randn(4, 3), dim=1)
    for alpha in (0.0, 0.1, 0.2):
        e = combine(g, l, alpha)
        expected = (1 - alpha) * F.normalize(g, dim=1) @ F.normalize(g, dim=1).T + alpha * l @ l.T
        assert torch.allclose(e @ e.T, expected, atol=1e-6)
    assert torch.equal(combine(g, l, 0.0)[:, 6:], torch.zeros(4, 3))
    assert math.isclose(combine(g, l, 0.2).norm(dim=1)[0].item(), 1.0, rel_tol=1e-6)


def test_drop_path_rate_grows_linearly_with_depth_and_is_identity_at_inference():
    from reid.models import apply_drop_path
    trunk = VisionTransformer(img_size=32, patch_size=8, embed_dim=16, depth=4, num_heads=2,
                              global_pool="map", class_token=False, num_classes=0)
    apply_drop_path(trunk, 0.3)
    rates = [getattr(block.drop_path1, "drop_prob", 0.0) for block in trunk.blocks]
    # the first block never drops, the last one drops at the configured rate; 0.3 / 3 is not exactly 0.1
    assert all(math.isclose(r, expected) for r, expected in zip(rates, [0.0, 0.1, 0.2, 0.3]))
    assert all(getattr(b.drop_path2, "drop_prob", 0.0) == r for b, r in zip(trunk.blocks, rates))
    trunk.eval()  # eval mode: DropPath must not change anything
    x = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        assert torch.equal(trunk(x), trunk(x))
