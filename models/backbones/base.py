from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class TemporalBackbone(nn.Module, ABC):
    """Unified interface for audio/video backbones.

    Every backbone returns temporal tokens shaped [B, T, C].
    The backbone must expose ``out_dim`` so downstream modules can
    construct projection layers without knowing its internal architecture.
    """

    out_dim: int

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def set_frozen(self, frozen: bool) -> None:
        self.frozen = bool(frozen)
        for parameter in self.parameters():
            parameter.requires_grad = not self.frozen

    def train(self, mode: bool = True):
        # Frozen pretrained backbones stay in eval mode to avoid stochastic
        # layers changing their behavior while saving activation memory.
        return super().train(False if getattr(self, "frozen", False) else mode)
