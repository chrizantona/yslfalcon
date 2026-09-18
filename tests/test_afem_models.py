import hashlib

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from reid.afem_models import SemanticAFEM, OSNetAppearance, FusionReIDModel, build_extended_model
from reid.inference import load_trained_encoder
from reid.optim import optimizer_groups
from reid.paper_models import IMAGENET_MEAN, IMAGENET_STD
from reid.train import imprint_classifier, supervised_contrastive
from reid.vendor.osnet_ain import osnet_ain_x1_0


HEAD = {"type": "arcface", "scale": 30.0, "margin": 0.5}


def test_osnet_loads_verified_author_dataparallel_checkpoint_strictly(tmp_path):
    original = osnet_ain_x1_0(num_classes=1000, pretrained=False)
    path = tmp_path / "osnet.pth"
    torch.save({"module." + key: value for key, value in original.state_dict().items()}, path)
    cfg = {"weights": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "dim": 16}
    model = OSNetAppearance(cfg, tmp_path, True)
    assert torch.equal(model.network.conv1.conv.weight, original.conv1.conv.weight)
    path.write_bytes(b"wrong checkpoint")
    with pytest.raises(ValueError, match="checksum mismatch"):
        OSNetAppearance(cfg, tmp_path, True)


def test_semantic_source_must_really_be_exp054(monkeypatch, tmp_path):
    monkeypatch.setattr("reid.afem_models.build_backbone", lambda *a, **k: fake_tower({}, False))
    path = tmp_path / "wrong_exp.pt"
    torch.save({"exp_id": "EXP-043"}, path)
    from reid.afem_models import siglip_tower
    with pytest.raises(ValueError, match="EXP-054"):
        siglip_tower({"model": {}, "checkpoint": str(path),
                      "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}, True)


def test_afem_starts_near_residual_base_and_all_branches_and_gates_learn():
    torch.manual_seed(0)
    module = SemanticAFEM(24, 16, groups=32)
    x = torch.randn(8, 24, requires_grad=True)
    original = module.main(x)
    refined = module(x)
    assert (refined - original).norm() / original.norm() < 0.01
    (refined * torch.randn_like(refined)).sum().backward()
    assert module.group_logits.grad.abs().min() > 0
    assert module.residual_scale.grad.abs() > 0
    for branch in module.branches:
        grad = branch[0].weight.grad
        assert torch.isfinite(grad).all() and grad.abs().sum(0).min() > 0
    with pytest.raises(ValueError, match="expects"):
        module(torch.randn(2, 3, 24))


def test_afem_query_does_not_depend_on_other_queries():
    module = SemanticAFEM(24, 16, groups=32).eval()
    x = torch.randn(7, 24)
    assert torch.allclose(module(x)[2], module(x[2:3])[0], atol=1e-6)


class TinyAppearance(nn.Module):
    def __init__(self, cfg, *args):
        super().__init__()
        self.network = nn.Conv2d(3, 12, 1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Linear(12, cfg["dim"])
        self.dim = cfg["dim"]

    def forward(self, x):
        return self.projection(self.pool(self.network(x)).flatten(1))


def fake_tower(cfg, load_pretrained):
    return nn.Sequential(nn.Conv2d(3, 24, 1), nn.AdaptiveAvgPool2d(1), nn.Flatten()), 24, (0.5,) * 3, (0.5,) * 3


def fusion_config(afem=True):
    return {"backbone": "siglip_osnet", "semantic": {}, "branch_dim": 16,
            "appearance": {"dim": 16}, "afem": {"groups": 32, "residual_init": 0.001} if afem else None}


def patch_towers(monkeypatch):
    monkeypatch.setattr("reid.afem_models.siglip_tower", fake_tower)
    monkeypatch.setattr("reid.afem_models.OSNetAppearance", TinyAppearance)


def test_fusion_renormalises_same_pixels_and_balances_branches(monkeypatch):
    patch_towers(monkeypatch)
    model, dim, _, _ = build_extended_model(fusion_config(), 4, load_pretrained=False, head=HEAD)
    b = model.backbone
    pixels = torch.rand(8, 3, 12, 12)
    x = (pixels - 0.5) / 0.5
    seen = []
    hook = b.appearance.register_forward_pre_hook(lambda module, args: seen.append(args[0]))
    fused, app = b.fused_and_appearance(x)
    hook.remove()
    expected = (pixels - torch.tensor(IMAGENET_MEAN)[None, :, None, None]) / torch.tensor(IMAGENET_STD)[None, :, None, None]
    assert torch.allclose(seen[0], expected, atol=1e-6)
    assert dim == 32 and torch.allclose(fused[:, :16].norm(dim=1), torch.ones(8))
    assert torch.allclose(fused[:, 16:].norm(dim=1), torch.ones(8))
    assert app.shape == (8, 16)


def test_optimizer_covers_every_trainable_parameter_once_and_uses_separate_lrs(monkeypatch):
    patch_towers(monkeypatch)
    model, *_ = build_extended_model(fusion_config(), 4, load_pretrained=False, head=HEAD)
    groups = optimizer_groups(model, {"lr_backbone": 1e-5, "lr_appearance": 1e-5, "lr_head": 5e-4})
    parameters = [p for group in groups for p in group["params"]]
    assert len(parameters) == len({id(p) for p in parameters})
    assert {id(p) for p in parameters} == {id(p) for p in model.parameters() if p.requires_grad}
    assert {g["branch"]: g["lr"] for g in groups} == {
        "siglip": 1e-5, "osnet": 1e-5, "new_projections_afem_pool": 5e-4, "reid_heads": 5e-4}


def test_fusion_auxiliary_loss_reaches_osnet_without_semantic_branch(monkeypatch):
    patch_towers(monkeypatch)
    model, *_ = build_extended_model(fusion_config(), 4, load_pretrained=False, head=HEAD)
    outputs = model(torch.randn(8, 3, 12, 12), torch.arange(4).repeat_interleave(2))
    labels = torch.arange(4).repeat_interleave(2)
    loss = F.cross_entropy(outputs[3], labels) + supervised_contrastive(outputs[2], labels, 0.07)
    loss.backward()
    assert model.backbone.appearance.network.weight.grad.abs().sum() > 0
    assert model.auxiliary_classifier.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.backbone.encoder.parameters())
    assert all(p.grad is None for p in model.backbone.refinement.parameters())


def test_fusion_imprints_both_heads_without_updating_neck_statistics(monkeypatch):
    patch_towers(monkeypatch)
    model, *_ = build_extended_model(fusion_config(), 4, load_pretrained=False, head=HEAD)
    before = {k: v.clone() for k, v in model.state_dict().items() if "running" in k or "num_batches" in k}
    imprint_classifier(model, [(torch.randn(8, 3, 12, 12), torch.arange(4).repeat_interleave(2))], "cpu")
    for head in [model.classifier, model.auxiliary_classifier]:
        assert torch.allclose(head.weight.norm(dim=1), torch.ones(4), atol=1e-6)
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in before.items())
    assert model.training


def test_fusion_checkpoint_reload_is_offline_and_query_independent(monkeypatch, tmp_path):
    patch_towers(monkeypatch)
    cfg = fusion_config()
    model, *_ = build_extended_model(cfg, 4, load_pretrained=False, head=HEAD)
    model.eval()
    path = tmp_path / "trained.pt"
    torch.save({"state_dict": model.state_dict(), "config": {"train": {"id_head": HEAD}}}, path)
    loaded, dim, *_ = load_trained_encoder(cfg, path)
    loaded.eval()
    x = torch.randn(8, 3, 12, 12)
    assert dim == 32 and torch.equal(model(x), loaded(x))
    assert torch.allclose(loaded(x)[0], loaded(x[:1])[0], atol=1e-6)


def test_osnet_ain_feature_map_path_has_no_original_classifier_and_gem_is_bounded():
    model = OSNetAppearance({"dim": 16, "pooling": "gem"}, load_pretrained=False).eval()
    assert isinstance(model.network.classifier, nn.Identity) and isinstance(model.network.fc, nn.Identity)
    with torch.no_grad():
        model.pool.p.fill_(-1.0)
        output = model(torch.randn(2, 3, 64, 64))
    assert output.shape == (2, 16) and torch.isfinite(output).all()
