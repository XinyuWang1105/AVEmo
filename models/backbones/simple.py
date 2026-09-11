from __future__ import annotations

import torch
import torch.nn as nn

from .base import TemporalBackbone


class ConvNormAct2d(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, stride=(1, 1)):
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )


class ConvNormAct3d(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, stride=(1, 1, 1)):
        super().__init__(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm3d(out_channels),
            nn.GELU(),
        )


class SimpleAudioBackbone(TemporalBackbone):
    """Small temporal CNN used only for smoke tests and ablations."""

    def __init__(self, out_dim: int = 256, freeze: bool = False):
        super().__init__()
        self.out_dim = int(out_dim)
        self.net = nn.Sequential(
            ConvNormAct2d(1, 32, stride=(2, 2)),
            ConvNormAct2d(32, 64, stride=(2, 2)),
            ConvNormAct2d(64, self.out_dim, stride=(2, 2)),
        )
        self.set_frozen(freeze)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.ndim != 3:
            raise ValueError(f"audio must be [B,T,F], got {tuple(audio.shape)}")
        x = self.net(audio.unsqueeze(1))  # [B,C,T',F']
        return x.mean(dim=-1).transpose(1, 2).contiguous()  # [B,T',C]


class SimpleVideoBackbone(TemporalBackbone):
    """Small temporal 3D CNN used only for smoke tests and ablations."""

    def __init__(self, out_dim: int = 256, freeze: bool = False):
        super().__init__()
        self.out_dim = int(out_dim)
        self.net = nn.Sequential(
            ConvNormAct3d(3, 32, stride=(1, 2, 2)),
            ConvNormAct3d(32, 64, stride=(2, 2, 2)),
            ConvNormAct3d(64, self.out_dim, stride=(2, 2, 2)),
        )
        self.set_frozen(freeze)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        if video.ndim != 5:
            raise ValueError(f"video must be [B,3,T,H,W], got {tuple(video.shape)}")
        x = self.net(video)  # [B,C,T',H',W']
        return x.mean(dim=(-1, -2)).transpose(1, 2).contiguous()  # [B,T',C]
