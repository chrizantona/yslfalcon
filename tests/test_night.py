import copy
import hashlib
from types import SimpleNamespace

import pytest
import torch

from reid.benchmark import InferencePrecision
from reid.models import build_backbone, checkpoint_name
from reid.night_gate import decide


def test_local_clip_preserves_tag_normalisation_and_verifies_hash(tmp_path, monkeypatch):
    path = tmp_path / "weights.safetensors"
    path.write_bytes(b"test weights")
    cfg = {"backbone": "clip", "name": "ViT-L-14", "pretrained": "laion2b_s32b_b82k",
           "weights": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    calls = []
    visual = SimpleNamespace(output_dim=768, preprocess_cfg={"mean": (0.5,) * 3, "std": (0.5,) * 3})
    monkeypatch.setattr("reid.models.open_clip.get_pretrained_cfg", lambda *a: {"mean": (0.5,) * 3, "std": (0.5,) * 3})
    monkeypatch.setattr("reid.models.open_clip.create_model_and_transforms",
                        lambda *a, **kw: (calls.append(kw) or SimpleNamespace(visual=visual), None, None))
    _, dim, mean, std = build_backbone(cfg, tmp_path)
    assert dim == 768 and mean == std == (0.5,) * 3
    assert calls[0]["pretrained"] == str(path) and calls[0]["image_mean"] == (0.5,) * 3
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        build_backbone(cfg, tmp_path)
    assert len(calls) == 1
    # Complete trained checkpoints initialise the architecture without requiring public files at inference.
    build_backbone(cfg, tmp_path, load_pretrained=False)
    assert calls[-1]["pretrained"] is None and calls[-1]["image_mean"] == (0.5,) * 3


def test_conditional_chain_requires_gain_over_both_seed_spreads():
    gate = {"parent": "EXP-003", "parent_mean": 0.710, "parent_std": 0.004, "exp016_mean": 0.773}
    assert not decide({"retrieval": {"mAP10": {"mean": 0.713, "std": 0.002}}}, gate)["continue"]
    assert not decide({"retrieval": {"mAP10": {"mean": 0.720, "std": 0.015}}}, gate)["continue"]
    result = decide({"retrieval": {"mAP10": {"mean": 0.730, "std": 0.005}}}, gate)
    assert result["continue"] and result["delta_vs_exp016"] == pytest.approx(-0.043)


def test_checkpoint_names_cannot_escape_checkpoint_directory():
    assert checkpoint_name({"checkpoint_name": "exp020_seed{seed}.pt"}, 2) == "exp020_seed2.pt"
    with pytest.raises(ValueError):
        checkpoint_name({"checkpoint_name": "../EXP-020.pt"})


def test_precision_wrapper_preserves_full_encoder_and_rejects_cpu_fp16():
    encoder = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.BatchNorm1d(3)).eval()
    wrapped = InferencePrecision(encoder, "fp32", "cpu").eval()
    x = torch.randn(4, 2)
    assert torch.equal(encoder(x), wrapped(x))
    with pytest.raises(ValueError, match="requires CUDA"):
        InferencePrecision(encoder, "fp16", "cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fp16 weights and CUDA graphs need a GPU")
def test_precision_wrapper_speedups_keep_autocast_embeddings_bit_identical():
    torch.manual_seed(0)
    encoder = torch.nn.Sequential(
        torch.nn.Conv2d(3, 16, 4, stride=4), torch.nn.GELU(), torch.nn.Flatten(), torch.nn.LayerNorm(16 * 8 * 8),
        torch.nn.Linear(16 * 8 * 8, 32), torch.nn.BatchNorm1d(32)).eval().cuda()
    reference = copy.deepcopy(encoder)
    wrapped = InferencePrecision(encoder, "fp16", "cuda").eval()
    assert encoder[0].weight.dtype == torch.float16 and encoder[3].weight.dtype == torch.float32
    for batch in (1, 5, 1):  # the second batch-1 call replays the captured graph
        x = torch.randn(batch, 3, 32, 32, device="cuda")
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            expected = reference(x)
        assert torch.equal(wrapped(x), expected)
    assert list(wrapped.graphs) == [((1, 3, 32, 32), torch.float32)]
