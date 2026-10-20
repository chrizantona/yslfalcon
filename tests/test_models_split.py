"""reid.models is split into reid._model_* modules; imports, pickles and checkpoints keep the reid.models path."""
import io
import pickle
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from timm.layers import AttentionPoolLatent
from torch import nn

from reid import models

NAMES = ("build_backbone", "MultiQueryPool", "apply_drop_path", "checkpoint_name",
         "interpolate_position_embedding", "load_backbone", "MarginClassifier", "ReIDModel")


@pytest.mark.parametrize("name", NAMES)
def test_every_definition_keeps_its_reid_models_path(name):
    definition = getattr(models, name)
    assert definition.__module__ == "reid.models"
    assert pickle.loads(pickle.dumps(definition)) is definition
    # the GLOBAL reference that full-model pickles made before the split store
    assert pickle.loads(f"creid.models\n{name}\n.".encode()) is definition


def test_state_dict_keys_and_full_model_pickles_survive_the_split():
    torch.set_num_threads(1)
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


def test_pool_backbone_and_checkpoint_helpers_behave_as_before():
    pool = models.MultiQueryPool(AttentionPoolLatent(16, num_heads=4), 3, 2.0)
    assert "query_offsets" in pool.state_dict()
    assert not any(key.startswith("pool.") for key in pool.state_dict())
    assert torch.equal(pool.query_offsets[0], torch.zeros(16))
    assert pool(torch.randn(2, 5, 16)).shape == (2, 16)
    pool.overlap.backward()
    assert pool.query_offsets.grad is not None

    # callers patch this dependency through reid.models
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

    assert "reid.teachers" not in sys.modules
    assert not hasattr(models, "load_fastreid")
    with pytest.raises(ValueError, match="unknown backbone fastreid_r50ibn"):
        models.build_backbone({"backbone": "fastreid_r50ibn"})
