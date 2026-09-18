"""ResNeXt101-IBN-a (32x4d) from IBN-Net: github.com/XingangPan/IBN-Net, commit d1673389b36c1180cf9bc35ea8260d84046da915,
ibnnet/modules.py and ibnnet/resnext_ibn.py. Parameter names are kept, so the released ImageNet weights
resnext101_ibn_a-6ace051d.pth load strictly (without fc). Changes: forward stops at layer4 and returns the feature map;
the random init and the other depths are removed.

MIT License

Copyright (c) 2017 Wei Yang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import math

import torch
import torch.nn as nn


class IBN(nn.Module):
    """Instance norm on the first half of the channels, batch norm on the rest."""

    def __init__(self, planes, ratio=0.5):
        super().__init__()
        self.half = int(planes * ratio)
        self.IN = nn.InstanceNorm2d(self.half, affine=True)
        self.BN = nn.BatchNorm2d(planes - self.half)

    def forward(self, x):
        first, second = torch.split(x, self.half, 1)
        return torch.cat((self.IN(first.contiguous()), self.BN(second.contiguous())), 1)


class Bottleneck(nn.Module):
    """ResNeXt bottleneck, type C; IBN replaces the first BN in layer1-3."""

    def __init__(self, inplanes, planes, stride=1, downsample=None, ibn=False, base_width=4, cardinality=32):
        super().__init__()
        width = int(math.floor(planes * (base_width / 64))) * cardinality
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False)
        self.bn1 = IBN(width) if ibn else nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + (x if self.downsample is None else self.downsample(x)))


class ResNeXt101IBNa(nn.Module):
    """Stem and layer1-4 of ResNeXt101-IBN-a; returns the stride-32 map with 2048 channels."""

    def __init__(self):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 3, 1, ibn=True)
        self.layer2 = self._make_layer(128, 4, 2, ibn=True)
        self.layer3 = self._make_layer(256, 23, 2, ibn=True)
        self.layer4 = self._make_layer(512, 3, 2, ibn=False)

    def _make_layer(self, planes, blocks, stride, ibn):
        downsample = None
        if stride != 1 or self.inplanes != planes * 4:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * 4, kernel_size=1, stride=stride, bias=False),
                                       nn.BatchNorm2d(planes * 4))
        layers = [Bottleneck(self.inplanes, planes, stride, downsample, ibn)]
        self.inplanes = planes * 4
        layers += [Bottleneck(self.inplanes, planes, ibn=ibn) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool1(self.relu(self.bn1(self.conv1(x))))
        return self.layer4(self.layer3(self.layer2(self.layer1(x))))
