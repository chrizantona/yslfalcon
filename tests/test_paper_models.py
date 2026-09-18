from functools import partial

import cv2
import numpy as np
import pandas as pd
import pytest
import torch
from timm.models.vision_transformer import VisionTransformer

from reid.paper_models import AFEM, GFAMUnit, shifted_window_mask, tinyclip_state_for_timm
from reid.train import CropDataset
from reid.train_paper import paper_loss, warmup_multistep


def test_tinyclip_weights_map_onto_timm_vit_with_resampled_positions():
    width, depth, patch = 16, 2, 8
    p = "vision_model."
    state = {p + "embeddings.class_embedding": torch.randn(width),
             p + "embeddings.patch_embedding.weight": torch.randn(width, 3, patch, patch),
             p + "embeddings.position_embedding.weight": torch.randn(1 + 2 * 2, width),  # trained at 16 px
             p + "embeddings.position_ids": torch.arange(5)[None],
             "visual_projection.weight": torch.randn(4, width), "text_model.embeddings.token_embedding.weight": torch.randn(3, 2)}
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
                            pre_norm=True, num_classes=0, global_pool="token", norm_layer=partial(torch.nn.LayerNorm, eps=1e-5))
    vit.load_state_dict(tinyclip_state_for_timm(state, (3, 3)))  # strict: every timm parameter is mapped
    qkv = vit.blocks[1].attn.qkv.weight
    assert torch.equal(qkv[:width], state[p + "encoder.layers.1.self_attn.q_proj.weight"])
    assert torch.equal(qkv[2 * width:], state[p + "encoder.layers.1.self_attn.v_proj.weight"])
    assert vit.pos_embed.shape == (1, 1 + 9, width)
    assert torch.equal(vit.pos_embed[0, 0], state[p + "embeddings.position_embedding.weight"][0])  # class position kept
    assert vit(torch.randn(2, 3, 24, 24)).shape == (2, width)


def test_afem_gates_the_kept_vector_with_the_weighted_groups():
    afem = AFEM(in_dim=6, dim=4, groups=3).eval()
    ts = torch.randn(5, 6)
    f = afem.linear(ts).view(5, 4, 4)
    torch.nn.init.zeros_(afem.weights)
    assert torch.allclose(afem(ts), f[:, 0])
    torch.nn.init.ones_(afem.weights)
    assert torch.allclose(afem(ts), f[:, 0] * (1 + f[:, 1:].sum(1)))


def test_shifted_window_mask_separates_regions_wrapped_by_the_shift():
    mask = shifted_window_mask(14, 14, 7, 3, "cpu")
    assert mask.shape == (4, 49, 49)
    assert mask[0].all()  # the top-left window holds one region only
    assert not mask[3].all() and mask[3].diagonal().all()  # the bottom-right one mixes four wrapped regions


def test_gfam_units_keep_the_map_shape():
    x = torch.randn(2, 14, 14, 16)
    for shift in (0, 3):
        y = GFAMUnit(16, heads=2, window=7, shift=shift, onn_order=3)(x)
        assert y.shape == x.shape and torch.isfinite(y).all()


def test_warmup_then_step_decay():
    factor = warmup_multistep(10, [40, 70])
    assert [round(factor(e), 4) for e in (0, 5, 10, 39, 40, 69, 70)] == [0.1, 0.55, 1.0, 1.0, 0.1, 0.1, 0.01]


def test_paper_loss_sums_ce_and_metric_loss_over_embeddings():
    labels = torch.tensor([0, 0, 1, 1])
    embeddings = [torch.randn(4, 8), torch.randn(4, 8)]
    logits = [torch.randn(4, 2), torch.randn(4, 2)]
    ce = torch.nn.CrossEntropyLoss()
    one = paper_loss(embeddings[:1], logits[:1], labels, {"ce": 1.0, "triplet": 0.5, "margin": 0.3}, ce)
    two = paper_loss(embeddings, logits, labels, {"ce": 1.0, "triplet": 0.5, "margin": 0.3}, ce)
    assert two > one > 0
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
