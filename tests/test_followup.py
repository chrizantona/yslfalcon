from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from reid.embed import FlipTTA
from reid.inference import load_trained_encoder
from reid.models import ReIDModel, interpolate_position_embedding
from reid.optim import optimizer_groups
from reid.train import imprint_classifier


class TinyViT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Linear(2, 2)
        self.class_embedding = torch.nn.Parameter(torch.zeros(2))
        self.positional_embedding = torch.nn.Parameter(torch.zeros(5, 2))
        self.ln_pre = torch.nn.LayerNorm(2)
        self.transformer = torch.nn.Module()
        self.transformer.resblocks = torch.nn.ModuleList([torch.nn.Linear(2, 2) for _ in range(3)])
        self.ln_post = torch.nn.LayerNorm(2)
        self.proj = torch.nn.Parameter(torch.eye(2))


def test_layer_decay_covers_all_trainable_parameters_once_and_leaves_head_lr():
    model = ReIDModel(TinyViT(), 2, 3)
    groups = optimizer_groups(model, {"layer_decay": 0.9, "lr_backbone": 1e-5, "lr_head": 5e-4})
    parameters = [id(p) for g in groups for p in g["params"]]
    assert len(parameters) == len(set(parameters))
    assert set(parameters) == {id(p) for p in model.parameters() if p.requires_grad}
    assert [g["lr"] for g in groups] == pytest.approx([1e-5 * .9 ** 3, 1e-5 * .9 ** 2, 1e-5 * .9, 1e-5, 5e-4])
    assert any(p is model.backbone.proj for p in groups[-2]["params"])
    assert any(p is model.bnneck.weight for p in groups[-1]["params"])


def test_default_optimizer_keeps_original_two_groups():
    model = ReIDModel(torch.nn.Linear(2, 2), 2, 3)
    groups = optimizer_groups(model, {"lr_backbone": 1e-5, "lr_head": 5e-4})
    assert len(groups) == 2 and groups[0]["lr"] == 1e-5 and groups[1]["lr"] == 5e-4
    assert [id(p) for p in groups[0]["params"]] == [id(p) for p in model.backbone.parameters()]


def test_imprinting_uses_normalised_class_means_and_does_not_update_bn():
    model = ReIDModel(torch.nn.Identity(), 2, 2, {"type": "arcface", "scale": 30, "margin": .5})
    buffers = {name: b.clone() for name, b in model.bnneck.named_buffers()}
    batch = (torch.tensor([[2., 0.], [1., 1.], [0., 3.]]), torch.tensor([0, 0, 1]))
    result = imprint_classifier(model, [batch], "cpu")
    expected = F.normalize(torch.tensor([[1 + 2 ** -.5, 2 ** -.5], [0, 1]]), dim=1)
    assert torch.allclose(model.classifier.weight, expected, atol=1e-6)
    assert model.training and result["images"] == 3 and result["classes"] == 2
    assert all(torch.equal(b, buffers[n]) for n, b in model.bnneck.named_buffers())


def test_imprinting_rejects_missing_classes_without_modifying_weights():
    model = ReIDModel(torch.nn.Identity(), 2, 2, {"type": "arcface", "scale": 30, "margin": .5})
    before = model.classifier.weight.clone()
    with pytest.raises(ValueError, match="every training class"):
        imprint_classifier(model, [(torch.ones(1, 2), torch.tensor([0]))], "cpu")
    assert torch.equal(before, model.classifier.weight) and model.training


def test_positional_interpolation_preserves_class_token_and_constant_patch_grid():
    old = torch.cat([torch.tensor([[8., 9.]]), torch.ones(4, 2)])
    new = interpolate_position_embedding(old, torch.empty(10, 2))
    assert new.shape == (10, 2) and torch.equal(new[0], old[0])
    assert torch.allclose(new[1:], torch.ones(9, 2))
    assert interpolate_position_embedding(old, old) is old


def test_flip_tta_is_a_normalised_mean_and_each_query_is_independent():
    class Encoder(torch.nn.Module):
        def forward(self, x):
            return x.flatten(1)
    encoder = Encoder()
    x = torch.tensor([[[[1., 2.]]], [[[3., 4.]]]])
    model = FlipTTA(encoder)
    expected = F.normalize(F.normalize(encoder(x), dim=1) + F.normalize(encoder(x.flip(-1)), dim=1), dim=1)
    assert torch.allclose(model(x), expected)
    assert torch.equal(model(x)[0], model(x[:1])[0])
    assert torch.allclose(model(x).norm(dim=1), torch.ones(2))


def test_trained_inference_loads_backbone_and_bn_instead_of_public_weights(tmp_path, monkeypatch):
    original = ReIDModel(torch.nn.Linear(2, 3), 3, 4).eval()
    original.bnneck.running_mean.fill_(2)
    checkpoint = tmp_path / "exp024.pt"
    torch.save({"state_dict": original.state_dict()}, checkpoint)
    calls = []

    def build(cfg, external, load_pretrained):
        calls.append(load_pretrained)
        return torch.nn.Linear(2, 3), 3, (0.5,) * 3, (0.5,) * 3

    monkeypatch.setattr("reid.inference.build_backbone", build)
    loaded, dim, _, _ = load_trained_encoder({}, checkpoint)
    x = torch.ones(2, 2)
    assert calls == [False] and dim == 3
    assert torch.equal(loaded.eval()(x), original(x))
def test_single_checkpoint_gain_requires_positive_bootstrap_interval(monkeypatch):
    import pandas as pd
    from reid import research_compare

    def describe(path):
        return {"exp_id": path, "protocol": "val_v2", "fold": 0}, .5 if path == "base" else .5005, 0.

    def load_ap(path):
        return pd.DataFrame({"image_id": ["a", "b", "c", "d"], "vehicle_id": [1, 2, 3, 4],
                             "AP": [.5] * 4 if path == "base" else [.6, .6, .4, .402]})

    monkeypatch.setattr(research_compare, "describe", describe)
    monkeypatch.setattr(research_compare, "load_ap", load_ap)
    result = research_compare.compare("base", "new")
    assert result["gain"] > 0 and result["ci95"][0] < 0 < result["ci95"][1]
    assert result["decision"] == "inconclusive"
