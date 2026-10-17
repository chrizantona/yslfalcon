"""Legacy model imports and checkpoint compatibility in research and serving trees."""
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize("tree", ["src", "service/runtime"])
def test_legacy_model_contract_in_each_tree(tree):
    root = Path(__file__).resolve().parents[1]
    environment = {
        **os.environ,
        "PYTHONPATH": str(root / tree),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    command = (
        f"import runpy; runpy.run_path({str(Path(__file__).resolve())!r})"
        f"['_check_legacy_model_contract']({tree!r})"
    )
    result = subprocess.run(
        [sys.executable, "-c", command], cwd=root, env=environment,
        text=True, capture_output=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _check_legacy_model_contract(tree):
    """Use clean interpreters so the two reid packages cannot mask one another."""
    import io
    import pickle
    from types import SimpleNamespace
    from unittest.mock import patch

    import torch
    from timm.layers import AttentionPoolLatent
    from torch import nn

    from reid import models

    torch.set_num_threads(1)
    for name in (
        "build_backbone", "MultiQueryPool", "apply_drop_path", "checkpoint_name",
        "interpolate_position_embedding", "load_backbone", "MarginClassifier", "ReIDModel",
    ):
        definition = getattr(models, name)
        assert definition.__module__ == "reid.models"
        assert pickle.loads(pickle.dumps(definition)) is definition
        # This is the GLOBAL reference stored by full-model pickles before the split.
        assert pickle.loads(f"creid.models\n{name}\n.".encode()) is definition

    model = models.ReIDModel(nn.Linear(4, 3), 3, 2).eval()
    assert list(model.state_dict()) == [
        "backbone.weight", "backbone.bias", "bnneck.weight", "bnneck.bias",
        "bnneck.running_mean", "bnneck.running_var", "bnneck.num_batches_tracked",
        "classifier.weight",
    ]
    assert not model.bnneck.bias.requires_grad
    x = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    expected = model(x)
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    restored = models.ReIDModel(nn.Linear(4, 3), 3, 2).eval()
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    assert torch.equal(restored(x), expected)
    model.train()
    features, logits = model(x)
    assert features.shape == (5, 3) and logits.shape == (5, 2)

    head = {"type": "cosface", "scale": 30.0, "margin": 0.35, "sub_centers": 2}
    model = models.ReIDModel(nn.Linear(4, 3), 3, 2, head).eval()
    buffer = io.BytesIO()
    torch.save(model, buffer)
    buffer.seek(0)
    restored = torch.load(buffer, weights_only=False)
    assert isinstance(restored, models.ReIDModel)
    assert isinstance(restored.classifier, models.MarginClassifier)
    assert torch.equal(restored(x), model(x))
    restored.train()
    _, logits = restored(x, torch.tensor([0, 1, 0, 1, 0]))
    assert logits.shape == (5, 2) and torch.isfinite(logits).all()

    pool = models.MultiQueryPool(AttentionPoolLatent(16, num_heads=4), 3, 2.0)
    assert "query_offsets" in pool.state_dict()
    assert not any(key.startswith("pool.") for key in pool.state_dict())
    assert torch.equal(pool.query_offsets[0], torch.zeros(16))
    assert pool(torch.randn(2, 5, 16)).shape == (2, 16)
    pool.overlap.backward()
    assert pool.query_offsets.grad is not None

    # Existing tests and callers patch this dependency through reid.models.
    visual = SimpleNamespace(output_dim=16, preprocess_cfg={"mean": (.5,) * 3, "std": (.25,) * 3})
    with patch.object(models.open_clip, "get_pretrained_cfg", return_value={"mean": (.5,) * 3}), \
            patch.object(models.open_clip, "create_model_and_transforms",
                         return_value=(SimpleNamespace(visual=visual), None, None)) as create:
        result = models.build_backbone(
            {"backbone": "clip", "name": "test", "pretrained": "test"}, load_pretrained=False)
    assert result == (visual, 16, (.5,) * 3, (.25,) * 3)
    assert create.call_args.kwargs == {"pretrained": None, "image_mean": (.5,) * 3}

    assert models.checkpoint_name({"checkpoint_name": "seed{seed}.pt"}, 7) == "seed7.pt"
    position = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    interpolated = models.interpolate_position_embedding(position, torch.empty(10, 4))
    assert interpolated.shape == (10, 4)
    assert torch.equal(interpolated[0], position[0])

    if tree == "service/runtime":
        assert "reid.teachers" not in sys.modules
        assert not hasattr(models, "load_fastreid")
        with pytest.raises(ValueError, match="unknown backbone fastreid_r50ibn"):
            models.build_backbone({"backbone": "fastreid_r50ibn"})
    else:
        assert callable(models.load_fastreid)
