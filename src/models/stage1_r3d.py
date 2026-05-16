from __future__ import annotations

import torch
import torch.nn as nn


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride=(1, 2, 2)):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class _SharedEyeEncoder(nn.Module):
    def __init__(self, base_channels: int = 16, layers=(1, 1, 1, 1)):
        super().__init__()
        channels = [base_channels, base_channels * 2, base_channels * 4, base_channels * 8]
        blocks = [_ConvBlock(3, channels[0], stride=(1, 2, 2))]
        for idx in range(1, len(channels)):
            blocks.append(_ConvBlock(channels[idx - 1], channels[idx]))
            for _ in range(max(0, layers[idx] - 1)):
                blocks.append(_ConvBlock(channels[idx], channels[idx], stride=(1, 1, 1)))
        self.net = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.out_channels = channels[-1]

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.net(x)
        return self.pool(x).flatten(1)


class Stage1FlowClassifier(nn.Module):
    """Shared-eye 3D CNN for FloPNet SCC identification."""

    def __init__(self, num_classes: int = 6, base_channels: int = 16, layers=(2, 2, 2, 2)):
        super().__init__()
        self.encoder = _SharedEyeEncoder(base_channels=base_channels, layers=layers)
        self.classifier = nn.Linear(self.encoder.out_channels * 2, num_classes)

    def forward(self, left_flow, right_flow):
        left = self.encoder(left_flow)
        right = self.encoder(right_flow)
        return self.classifier(torch.cat([left, right], dim=1))


def stage1_r3d(**kwargs):
    return Stage1FlowClassifier(**kwargs)
