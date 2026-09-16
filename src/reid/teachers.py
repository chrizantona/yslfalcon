"""Public vehicle ReID checkpoints used as reference models and teachers.

- FastReID BoT ResNet50-IBN (github.com/JDAI-CV/fast-reid, Apache-2.0): last stride 1, GeM pooling, test feature =
  BN(GeM(layer4)), input 256x256, pixel mean/std stored in the checkpoint on the 0..255 scale.
- CLIP-ReID ViT-B/16 (github.com/Syliz517/CLIP-ReID, MIT): OpenAI CLIP visual tower (QuickGELU) at 256x256, test
  feature = concat(ln_post(cls), ln_post(cls) @ proj) with NECK_FEAT "before", mean = std = 0.5.

build_teacher(model_cfg, weights_dir) -> (encoder, feature dim, mean, std) where encoder(images) returns that feature.
"""
from pathlib import Path

import open_clip
import torch
import torch.nn.functional as F
import torchvision
from torch import nn


class IBN(nn.Module):
    """IBN-Net (a): instance norm on the first half of the channels, batch norm on the rest."""

    def __init__(self, planes):
        super().__init__()
        self.half = planes // 2
        self.IN = nn.InstanceNorm2d(self.half, affine=True)
        self.BN = nn.BatchNorm2d(planes - self.half)

    def forward(self, x):
        first, rest = torch.split(x, [self.half, x.shape[1] - self.half], 1)
        return torch.cat([self.IN(first.contiguous()), self.BN(rest.contiguous())], 1)


class FastReIDResNet50IBN(nn.Module):
    """FastReID backbone + GeM + BNNeck with the same parameter names, so its state dict loads as is."""

    def __init__(self, gem_p=3.0):
        super().__init__()
        net = torchvision.models.resnet50()
        for layer in (net.layer1, net.layer2, net.layer3):  # FastReID puts IBN in layers 1-3 only
            for block in layer:
                block.bn1 = IBN(block.bn1.num_features)
        net.layer4[0].conv2.stride = (1, 1)
        net.layer4[0].downsample[0].stride = (1, 1)
        net.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        del net.avgpool, net.fc
        self.backbone, self.bottleneck, self.gem_p = net, nn.BatchNorm2d(2048), gem_p

    def forward(self, x):
        b = self.backbone
        x = b.maxpool(b.relu(b.bn1(b.conv1(x))))
        x = b.layer4(b.layer3(b.layer2(b.layer1(x))))
        pooled = F.adaptive_avg_pool2d(x.float().clamp(min=1e-6).pow(self.gem_p), 1).pow(1 / self.gem_p)
        return self.bottleneck(pooled)[..., 0, 0]


def load_fastreid(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)["model"]
    model = FastReIDResNet50IBN(float(ckpt["heads.pool_layer.p"]) if "heads.pool_layer.p" in ckpt else 3.0)
    state = {k: v for k, v in ckpt.items() if k.startswith("backbone.")}
    state.update({"bottleneck." + k[len("heads.bottleneck.0."):]: v for k, v in ckpt.items()
                  if k.startswith("heads.bottleneck.0.")})
    model.load_state_dict(state)
    mean, std = (ckpt[k].flatten() / 255 for k in ("pixel_mean", "pixel_std"))
    return model, 2048, tuple(mean.tolist()), tuple(std.tolist())


class ClipReIDViT(nn.Module):
    def __init__(self, visual):
        super().__init__()
        self.visual = visual

    def forward(self, x):
        v = self.visual
        x = v.conv1(x).flatten(2).transpose(1, 2)
        x = torch.cat([v.class_embedding.to(x.dtype).expand(x.shape[0], 1, -1), x], 1) + v.positional_embedding.to(x.dtype)
        x = v.transformer(v.ln_pre(x))
        cls = v.ln_post(x[:, 0])
        return torch.cat([cls, cls @ v.proj], 1)


def load_clipreid(path, input_hw):
    visual = open_clip.create_model("ViT-B-16-quickgelu", pretrained=None, force_image_size=tuple(input_hw)).visual
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    visual.load_state_dict({k[len("image_encoder."):]: v for k, v in ckpt.items() if k.startswith("image_encoder.")})
    return ClipReIDViT(visual), 768 + 512, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)


def build_teacher(model_cfg, weights_dir):
    path = Path(weights_dir) / model_cfg["weights"]
    if model_cfg["backbone"] == "fastreid_r50ibn":
        return load_fastreid(path)
    if model_cfg["backbone"] == "clipreid_vitb16":
        return load_clipreid(path, model_cfg["input_hw"])
    raise ValueError(f"unknown teacher {model_cfg['backbone']}")
