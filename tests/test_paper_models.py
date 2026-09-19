from functools import partial

import cv2
import numpy as np
import pandas as pd
import pytest
import torch
from timm.models.vision_transformer import VisionTransformer

from reid.paper_models import AFEM, CLIPSENet, GeM, GFAMUnit, build_paper_model, shifted_window_mask, tinyclip_state_for_timm
from reid.train import CropDataset
from reid.train_paper import build_paper_optimizer, build_paper_scheduler, paper_loss, warmup_multistep


def test_tinyclip_weights_map_onto_timm_vit_with_resampled_positions():
    width, depth, patch = 16, 2, 8
    p = "vision_model."
    state = {p + "embeddings.class_embedding": torch.randn(width),
             p + "embeddings.patch_embedding.weight": torch.randn(width, 3, patch, patch),
             p + "embeddings.position_embedding.weight": torch.randn(1 + 2 * 2, width),  # trained at 16 px
             p + "embeddings.position_ids": torch.arange(5)[None],
             "visual_projection.weight": torch.randn(4, width),
             "text_model.embeddings.token_embedding.weight": torch.randn(3, 2)}
    for name in ("pre_layrnorm", "post_layernorm"):
        state[f"{p}{name}.weight"], state[f"{p}{name}.bias"] = torch.randn(width), torch.randn(width)
    for i in range(depth):
        src = f"{p}encoder.layers.{i}."
        for name, shape in (("self_attn.q_proj", (width, width)), ("self_attn.k_proj", (width, width)),
                            ("self_attn.v_proj", (width, width)), ("self_attn.out_proj", (width, width)),
                            ("mlp.fc1", (4 * width, width)), ("mlp.fc2", (width, 4 * width)),
                            ("layer_norm1", (width,)), ("layer_norm2", (width,))):
            state[f"{src}{name}.weight"] = torch.randn(shape)
            state[f"{src}{name}.bias"] = torch.randn(shape[0])
    vit = VisionTransformer(img_size=24, patch_size=patch, embed_dim=width, depth=depth, num_heads=2, mlp_ratio=4,
                            pre_norm=True, num_classes=0, global_pool="token",
                            norm_layer=partial(torch.nn.LayerNorm, eps=1e-5))
    vit.load_state_dict(tinyclip_state_for_timm(state, (3, 3)))  # strict: every timm parameter is mapped
    qkv = vit.blocks[1].attn.qkv.weight
    assert torch.equal(qkv[:width], state[p + "encoder.layers.1.self_attn.q_proj.weight"])
    assert torch.equal(qkv[2 * width:], state[p + "encoder.layers.1.self_attn.v_proj.weight"])
    assert vit.pos_embed.shape == (1, 1 + 9, width)
    assert torch.equal(vit.pos_embed[0, 0], state[p + "embeddings.position_embedding.weight"][0])  # class position kept
    assert vit(torch.randn(2, 3, 24, 24)).shape == (2, width)


def test_legacy_afem_gates_the_kept_vector_with_the_weighted_groups():
    afem = AFEM(in_dim=6, dim=4, groups=3).eval()
    ts = torch.randn(5, 6)
    f = afem.linear(ts).view(5, 4, 4)
    torch.nn.init.zeros_(afem.weights)
    assert torch.allclose(afem(ts), f[:, 0])
    torch.nn.init.ones_(afem.weights)
    assert torch.allclose(afem(ts), f[:, 0] * (1 + f[:, 1:].sum(1)))


def test_paper_afem_adds_the_weighted_groups_to_the_retained_vector():
    afem = AFEM(in_dim=6, dim=4, groups=3, mode="paper_sum").eval()
    ts = torch.randn(5, 6)
    f = afem.linear(ts).view(5, 4, 4)
    torch.nn.init.zeros_(afem.weights)
    assert torch.allclose(afem(ts), f[:, 3])
    torch.nn.init.ones_(afem.weights)
    assert torch.allclose(afem(ts), f[:, 3] + f[:, :3].sum(1))


def test_afem_rejects_unknown_formula():
    with pytest.raises(ValueError, match="unknown AFEM mode"):
        AFEM(in_dim=6, dim=4, groups=3, mode="guess")


def test_compact_afem_scales_channel_groups_without_full_vector_copies():
    afem = AFEM(in_dim=6, dim=12, groups=3, mode="compact_group_scale").eval()
    ts = torch.randn(5, 6)
    f = afem.linear(ts)
    assert afem.linear[0].out_features == 12 and afem.weights.shape == (3, 1)
    torch.nn.init.zeros_(afem.weights)
    assert torch.allclose(afem(ts), f)
    torch.nn.init.ones_(afem.weights)
    assert torch.allclose(afem(ts), 2 * f)
    with pytest.raises(ValueError, match="must be divisible"):
        AFEM(in_dim=6, dim=10, groups=3, mode="compact_group_scale")


def test_clip_senet_semantic_bn_is_an_explicit_option():
    cfg = {"input_hw": [32, 32], "embedding_dim": 16, "afem_groups": 2, "semantic_bn": True,
           "semantic": {"patch": 32, "width": 20, "depth": 1, "heads": 2, "mlp": 40, "dim": 8}}
    model = CLIPSENet(cfg, n_classes=3, load_pretrained=False)
    assert isinstance(model.semantic_norm, torch.nn.BatchNorm1d) and model.semantic_norm.num_features == 8
    optimizer = build_paper_optimizer(model, {"weight_decay": 5e-4,
                                               "lr_groups": {"cnn": 3.5e-5, "clip": 1e-5, "new": 5e-4}})
    assert {group["branch"]: group["lr"] for group in optimizer.param_groups} == {
        "cnn": 3.5e-5, "clip": 1e-5, "new": 5e-4}
    grouped = [p for group in optimizer.param_groups for p in group["params"]]
    assert len(grouped) == len({id(p) for p in grouped}) == len(list(model.parameters()))


def test_shifted_window_mask_separates_regions_wrapped_by_the_shift():
    mask = shifted_window_mask(14, 14, 7, 3, "cpu")
    assert mask.shape == (4, 49, 49)
    assert mask[0].all()  # the top-left window holds one region only
    assert not mask[3].all() and mask[3].diagonal().all()  # the bottom-right one mixes four wrapped regions


def test_gfam_units_keep_the_map_shape():
    x = torch.randn(2, 14, 14, 16)
    for shift in (0, 3):
        y = GFAMUnit(16, heads=2, window=7, shift=shift, onn_order=3)(x, x)
        assert y.shape == x.shape and torch.isfinite(y).all()


def test_log_space_gem_matches_power_mean_and_is_finite_on_signed_stripes():
    positive = torch.rand(2, 4, 5, 3) + 0.01
    gem = GeM(p=3.0)
    expected = positive.pow(3).mean((2, 3)).pow(1 / 3)
    assert torch.allclose(gem(positive), expected, atol=1e-6, rtol=1e-6)

    signed = torch.full((2, 4, 5, 3), -20.0, requires_grad=True)
    stable = GeM(p=12.0, min_p=1.0, input_activation="softplus")
    pooled = stable(signed)
    pooled.sum().backward()
    assert torch.isfinite(pooled).all()
    assert torch.isfinite(signed.grad).all()
    assert torch.isfinite(stable.p.grad)


def test_gem_rejects_unknown_input_activation():
    with pytest.raises(ValueError, match="unknown GeM input activation"):
        GeM(input_activation="signed_power")


def test_signed_gem_preserves_mean_at_p_one_and_large_p_gradients_are_finite():
    x = torch.randn(2, 4, 5, 3, requires_grad=True)
    average = GeM(p=1.0, min_p=1.0, input_activation="signed")
    assert torch.allclose(average(x), x.mean((2, 3)), atol=2e-6, rtol=1e-6)

    stable = GeM(p=12.0, min_p=1.0, input_activation="signed")
    pooled = stable(x)
    pooled.square().sum().backward()
    assert torch.isfinite(pooled).all() and torch.isfinite(x.grad).all() and torch.isfinite(stable.p.grad)


def test_convnext_glpe_follows_the_described_layout():
    cfg = {"arch": "convnext_glpe", "gfam": {"heads": 8, "window": 7, "onn_order": 3}, "stage3_bn": True,
           "gem_input_activation": "softplus"}
    model, dim, _, _ = build_paper_model(cfg, n_classes=5, load_pretrained=False)
    assert len(model.units) == 2 and [u.attn.shift for u in model.units] == [0, 3]  # one block: W-GFAM, SW-GFAM
    assert all(u.attn.window == 7 for u in model.units) and all(len(u.attn.q.powers) == 3 for u in model.units)
    gates = []
    for unit in model.units:
        unit.attn.gate.register_forward_hook(lambda module, args, out: gates.append(args[0]))
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        block_input = model.stage_norm.eval()(model.features(x)).permute(0, 2, 3, 1)
        model.stage_norm.train()
        embeddings, logits = model.train()(x)
    assert block_input.shape == (2, 14, 14, 768)
    assert len(embeddings) == len(logits) == 6 and logits[0].shape == (2, 5)
    assert all(torch.allclose(e.norm(dim=1), torch.ones(2)) for e in embeddings)
    assert torch.equal(gates[0], gates[1])  # both gates see the OT-GFAM input, not the W-GFAM output
    with torch.no_grad():
        descriptor = model.eval()(x)
    assert descriptor.shape == (2, 6 * 768) == (2, dim)
    assert torch.allclose(descriptor.view(2, 6, 768).norm(dim=2), torch.ones(2, 6))  # six normalised parts
    assert all(pool.input_activation == "softplus" for pool in model.pools)

    optimizer = build_paper_optimizer(
        model, {"weight_decay": 5e-4, "lr_groups": {"backbone": 1e-5, "new": 3.5e-4}})
    assert {group["branch"]: group["lr"] for group in optimizer.param_groups} == {
        "backbone": 1e-5, "new": 3.5e-4}
    grouped = [p for group in optimizer.param_groups for p in group["params"]]
    assert len(grouped) == len({id(p) for p in grouped}) == len(list(model.parameters()))


def test_gem_exponent_is_kept_at_or_above_its_floor():
    x = torch.rand(2, 4, 3, 3)
    gem = GeM(min_p=1.0)
    gem.p.data.fill_(-0.5)  # where unbounded training can push it
    assert torch.allclose(gem(x), x.mean((2, 3)), atol=1e-5)  # clamped to 1 = average pooling
    gem.p.data.fill_(3.0)
    assert torch.allclose(gem(x), x.pow(3).mean((2, 3)).pow(1 / 3), atol=1e-5)
    unbounded = GeM(p=0.0)
    unbounded(x).sum().backward()
    assert not torch.isfinite(unbounded.p.grad)  # the unbounded version breaks at p = 0 (EXP-061)


def test_warmup_then_step_decay():
    factor = warmup_multistep(10, [40, 70])
    assert [round(factor(e), 4) for e in (0, 5, 10, 39, 40, 69, 70)] == [0.1, 0.55, 1.0, 1.0, 0.1, 0.1, 0.01]


def test_small_dataset_recipe_uses_24_epoch_cosine_without_warmup():
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.Adam([parameter], lr=5e-4)
    scheduler = build_paper_scheduler(optimizer, {"scheduler": "cosine", "epochs": 24})
    assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
    assert scheduler.T_max == 24 and scheduler.get_last_lr() == [5e-4]


def test_cosine_run_can_stop_before_the_fixed_scheduler_horizon():
    parameter = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.Adam([parameter], lr=5e-4)
    scheduler = build_paper_scheduler(optimizer, {"scheduler": "cosine", "epochs": 80, "scheduler_t_max": 124})
    assert scheduler.T_max == 124
    with pytest.raises(ValueError, match="cannot be shorter"):
        build_paper_scheduler(optimizer, {"scheduler": "cosine", "epochs": 80, "scheduler_t_max": 79})


def test_paper_loss_sums_ce_and_metric_loss_over_embeddings():
    labels = torch.tensor([0, 0, 1, 1])
    embeddings = [torch.randn(4, 8), torch.randn(4, 8)]
    logits = [torch.randn(4, 2), torch.randn(4, 2)]
    ce = torch.nn.CrossEntropyLoss()
    one, _ = paper_loss(embeddings[:1], logits[:1], labels, {"ce": 1.0, "triplet": 0.5, "margin": 0.3}, ce)
    two, branches = paper_loss(embeddings, logits, labels, {"ce": 1.0, "triplet": 0.5, "margin": 0.3}, ce)
    assert two > one > 0 and len(branches) == 2
    assert torch.isclose(two, branches[0] + branches[1]) and torch.isclose(one, branches[0])  # summed, not averaged
    with pytest.raises(KeyError):
        paper_loss(embeddings, logits, labels, {"ce": 1.0, "supcon": 1.0}, ce)  # SupCon needs its temperature


def test_color_jitter_changes_colours_and_is_off_by_default(tmp_path):
    cv2.imwrite(str(tmp_path / "a.jpg"), np.full((32, 32, 3), 120, np.uint8))
    df = pd.DataFrame({"image_id": ["a"], "label": [0]})
    base = {"flip_p": 0, "pad": 0, "erasing_p": 0}
    plain = CropDataset(df, tmp_path, (16, 16), (0.5,) * 3, (0.5,) * 3, base)
    assert plain.jitter is None
    jittered = CropDataset(df, tmp_path, (16, 16), (0.5,) * 3, (0.5,) * 3,
                           {**base, "color_jitter": {"p": 1.0, "brightness": 0.5}})
    torch.manual_seed(0)
    assert not torch.equal(plain[0][0], jittered[0][0])
