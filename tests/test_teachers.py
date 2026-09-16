import numpy as np
import open_clip
import torch

from reid.ensemble import mean_embedding
from reid.models import ReIDModel, build_backbone
from reid.soup import average_state_dicts
from reid.teachers import FastReIDResNet50IBN, load_clipreid, load_fastreid


def test_fastreid_checkpoint_loads_with_its_names_gem_power_and_pixel_stats(tmp_path):
    torch.manual_seed(0)
    model = FastReIDResNet50IBN(gem_p=3.5).eval()
    model.bottleneck.running_mean.uniform_(0, 1)
    state = {k if k.startswith("backbone.") else "heads.bottleneck.0." + k[len("bottleneck."):]: v
             for k, v in model.state_dict().items()}
    state.update({"heads.classifier.weight": torch.zeros(10, 2048), "heads.pool_layer.p": torch.tensor([3.5]),
                  "pixel_mean": torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1),
                  "pixel_std": torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1)})
    torch.save({"model": state}, tmp_path / "fastreid.pth")

    loaded, dim, mean, std = load_fastreid(tmp_path / "fastreid.pth")
    x = torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        assert dim == 2048 and loaded.gem_p == 3.5 and np.allclose(mean, (0.485, 0.456, 0.406), atol=1e-3)
        assert torch.allclose(loaded.eval()(x), model(x), atol=1e-5)


def test_clipreid_feature_is_cls_and_its_projection_like_open_clip(tmp_path):
    visual = open_clip.create_model("ViT-B-16-quickgelu", pretrained=None, force_image_size=(256, 256)).visual.eval()
    state = {"image_encoder." + k: v for k, v in visual.state_dict().items()}
    state["text_encoder.dummy"] = torch.zeros(1)
    torch.save(state, tmp_path / "clipreid.pth")

    encoder, dim, _, _ = load_clipreid(tmp_path / "clipreid.pth", (256, 256))
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        feature = encoder.eval()(x)
        assert feature.shape == (1, dim) == (1, 1280)
        assert torch.allclose(feature[:, 768:], visual(x), atol=1e-4)  # projection part = open_clip image embedding


def test_mean_embedding_normalises_members_and_result():
    a, b = np.array([[2.0, 0.0]]), np.array([[0.0, 5.0]])
    assert np.allclose(mean_embedding([a, b]), [[1 / np.sqrt(2), 1 / np.sqrt(2)]])


def test_soup_averages_floats_and_keeps_integer_buffers():
    states = [{"w": torch.tensor([1.0, 3.0]), "n": torch.tensor(5)}, {"w": torch.tensor([3.0, 5.0]), "n": torch.tensor(7)}]
    soup = average_state_dicts(states)
    assert torch.equal(soup["w"], torch.tensor([2.0, 4.0])) and soup["n"].item() == 5


def test_fastreid_backbone_for_training_drops_its_bn_and_fits_reidmodel(tmp_path):
    torch.manual_seed(0)
    source = FastReIDResNet50IBN().eval()
    state = {k if k.startswith("backbone.") else "heads.bottleneck.0." + k[len("bottleneck."):]: v
             for k, v in source.state_dict().items()}
    state.update({"pixel_mean": torch.full((1, 3, 1, 1), 127.5), "pixel_std": torch.full((1, 3, 1, 1), 127.5)})
    (tmp_path / "fast-reid").mkdir()
    torch.save({"model": state}, tmp_path / "fast-reid" / "w.pth")

    backbone, dim, mean, _ = build_backbone({"backbone": "fastreid_r50ibn", "weights": "fast-reid/w.pth"}, tmp_path)
    model = ReIDModel(backbone, dim, n_classes=7).train()
    x, y = torch.randn(4, 3, 64, 64), torch.tensor([0, 0, 1, 1])
    feat, logits = model(x, y)
    assert feat.shape == (4, 2048) and logits.shape == (4, 7) and np.allclose(mean, (0.5, 0.5, 0.5))
    assert not any(k.startswith("backbone.bottleneck") for k in model.state_dict())  # only our BNNeck remains
