import torch
from timm.layers import AttentionPoolLatent
from torch import nn

from reid.models import MultiQueryPool, load_backbone
from reid.optim import optimizer_groups


def single_pool():
    """A single-query pool shaped like SigLIP 2's: the query sits in the q bias, the latent is small."""
    pool = AttentionPoolLatent(32, num_heads=4)
    pool.latent.data = 0.01 * torch.randn(1, 1, 32)
    pool.q.bias.data = torch.randn(32)
    return pool


class Trunk(nn.Module):
    def __init__(self, n_queries=1, offset_norm=1.0):
        super().__init__()
        self.attn_pool = single_pool()
        if n_queries > 1:
            self.attn_pool = MultiQueryPool(self.attn_pool, n_queries, offset_norm)


def test_zero_offsets_reproduce_the_single_query_pool():
    torch.manual_seed(0)
    pool = single_pool().eval()
    x = torch.randn(2, 10, 32)
    with torch.no_grad():
        before = pool(x)
        after = MultiQueryPool(pool, 4, offset_norm=0.0).eval()(x)
    assert after.shape == before.shape == (2, 32)            # the embedding keeps its size
    assert torch.allclose(before, after, atol=1e-5)


def test_the_first_query_is_the_pretrained_one_and_the_rest_are_offset():
    offsets = MultiQueryPool(single_pool(), 4, offset_norm=3.0).query_offsets
    assert offsets.shape == (4, 32)
    assert torch.equal(offsets[0], torch.zeros(32))
    assert torch.allclose(offsets[1:].norm(dim=-1), torch.full((3,), 3.0))


def test_overlap_is_one_for_copies_and_its_gradient_reaches_the_offsets():
    torch.manual_seed(0)
    x = torch.randn(2, 10, 32)
    copies = MultiQueryPool(single_pool(), 4, offset_norm=0.0).train()
    copies(x)
    assert torch.allclose(copies.overlap, torch.tensor(1.0), atol=1e-5)
    spread = MultiQueryPool(single_pool(), 4, offset_norm=5.0).train()
    spread(x)
    assert spread.overlap < 1
    spread.overlap.backward()
    assert (spread.query_offsets.grad[1:].abs().sum(-1) > 0).all()


def test_a_single_query_checkpoint_loads_and_the_offsets_keep_their_initialisation(tmp_path):
    single = Trunk()
    torch.save({"state_dict": {f"backbone.{k}": v for k, v in single.state_dict().items()}}, tmp_path / "pretrain.pt")
    model = nn.Module()
    model.backbone = Trunk(n_queries=4)
    offsets = model.backbone.attn_pool.query_offsets.detach().clone()
    load_backbone(model, tmp_path / "pretrain.pt")           # the VERI-Wild pretraining has a single query
    pool = model.backbone.attn_pool
    assert torch.equal(pool.q.bias, single.attn_pool.q.bias)
    assert torch.equal(pool.query_offsets, offsets)


def test_offsets_learn_at_the_head_rate():
    model = nn.Module()
    model.backbone, model.bnneck, model.classifier = Trunk(n_queries=4), nn.BatchNorm1d(32), nn.Linear(32, 5)
    backbone, head = optimizer_groups(model, {"lr_backbone": 1e-5, "lr_head": 5e-4})
    offsets = model.backbone.attn_pool.query_offsets
    assert any(p is offsets for p in head["params"]) and head["lr"] == 5e-4
    assert all(p is not offsets for p in backbone["params"])
    assert len(backbone["params"]) + len(head["params"]) == len(list(model.parameters()))
