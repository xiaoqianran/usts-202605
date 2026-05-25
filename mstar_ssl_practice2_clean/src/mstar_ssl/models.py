from __future__ import annotations

import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + residual)


class SmallResNet(nn.Module):
    """Compact ResNet for 1-channel MSTAR SAR target images."""

    def __init__(self, num_classes: int, widths: tuple[int, ...] = (32, 64, 128, 256)):
        super().__init__()
        self.in_channels = widths[0]
        self.stem = nn.Sequential(
            nn.Conv2d(1, widths[0], 3, padding=1, bias=False),
            nn.BatchNorm2d(widths[0]),
            nn.ReLU(inplace=True),
        )
        self.layer1 = self._make_layer(widths[0], blocks=2, stride=1)
        self.layer2 = self._make_layer(widths[1], blocks=2, stride=2)
        self.layer3 = self._make_layer(widths[2], blocks=2, stride=2)
        self.layer4 = self._make_layer(widths[3], blocks=2, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(widths[3], num_classes)
        self._init_weights()

    def _make_layer(self, channels: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [BasicBlock(self.in_channels, channels, stride)]
        self.in_channels = channels
        for _ in range(blocks - 1):
            layers.append(BasicBlock(self.in_channels, channels))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


class SmallCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x).flatten(1))


def build_model(name: str, num_classes: int) -> nn.Module:
    if name == "smallresnet":
        return SmallResNet(num_classes)
    if name == "smallcnn":
        return SmallCNN(num_classes)
    raise ValueError(f"Unknown model: {name}")


def describe_model(num_classes: int = 10) -> dict[str, object]:
    model = SmallResNet(num_classes=num_classes)
    params = sum(p.numel() for p in model.parameters())
    return {
        "name": "SmallResNet",
        "input": "1 x 128 x 128 grayscale SAR image",
        "stages": [
            "stem: 3x3 conv, 32 channels",
            "layer1: 2 residual blocks, 32 channels",
            "layer2: 2 residual blocks, 64 channels, stride 2",
            "layer3: 2 residual blocks, 128 channels, stride 2",
            "layer4: 2 residual blocks, 256 channels, stride 2",
            "global average pooling + linear classifier",
        ],
        "output": f"{num_classes} target-class logits",
        "parameters": params,
    }

