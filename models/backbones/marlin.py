from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from .base import TemporalBackbone


class SpatialAttentionPool(nn.Module):
    """Learn a spatial weighting while retaining MARLIN temporal tokens."""

    def __init__(self, feature_dim: int):
        super().__init__()
        self.score = nn.Linear(feature_dim, 1)
        # Start exactly from the former mean-pooling behavior.
        nn.init.zeros_(self.score.weight)
        nn.init.zeros_(self.score.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4:
            raise ValueError(
                "SpatialAttentionPool expects [B,T,S,C], "
                f"got {tuple(features.shape)}"
            )
        weights = torch.softmax(self.score(features).squeeze(-1), dim=2)
        return (features * weights.unsqueeze(-1)).sum(dim=2).contiguous()


class MARLINBackbone(TemporalBackbone):
    """MARLIN facial-video encoder returning complete final-layer patch tokens.

    The official MARLIN ViT-B receives [B, 3, 16, 224, 224] clips and
    produces 8 x 14 x 14 patch tokens. All 1,568 tokens are retained, as in
    HAVIC's fine-tuning path; spatial aggregation is left to downstream code.
    """

    MODEL_DIMS = {
        "marlin_vit_small_ytf": 384,
        "marlin_vit_base_ytf": 768,
        "marlin_vit_large_ytf": 1024,
    }

    def __init__(
        self,
        checkpoint: str | None,
        model_name: str = "marlin_vit_base_ytf",
        freeze: bool = False,
        allow_online: bool = False,
        tubelet_size: int = 2,
    ):
        super().__init__()
        try:
            from marlin_pytorch import Marlin
        except ImportError as exc:
            raise ImportError(
                "MARLINBackbone requires marlin-pytorch. Install it with: "
                "pip install marlin-pytorch"
            ) from exc

        if model_name not in self.MODEL_DIMS:
            raise ValueError(
                f"Unsupported MARLIN model {model_name!r}. "
                f"Available: {list(self.MODEL_DIMS)}"
            )
        self.out_dim = self.MODEL_DIMS[model_name]
        self.tubelet_size = int(tubelet_size)

        if checkpoint:
            path = Path(checkpoint).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"MARLIN checkpoint not found: {path}")
            self.model = Marlin.from_file(model_name, str(path))
            print(f"[INFO] loaded MARLIN encoder: {path}")
        elif allow_online:
            self.model = Marlin.from_online(model_name)
            print(f"[INFO] loaded MARLIN encoder online: {model_name}")
        else:
            raise ValueError(
                "MARLIN checkpoint is required. Set video_backbone.checkpoint "
                "or explicitly set allow_online=True."
            )

        self.set_frozen(freeze)

    def set_frozen(self, frozen: bool) -> None:
        self.frozen = bool(frozen)
        if hasattr(self, "model"):
            self.model.requires_grad_(not self.frozen)

    def train(self, mode: bool = True):
        nn.Module.train(self, mode)
        if getattr(self, "frozen", False):
            self.model.train(False)
        return self

    def _forward_impl(self, video: torch.Tensor) -> torch.Tensor:
        features = self.model.extract_features(video, keep_seq=True)
        return self._flatten_features(features, video)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        self._validate_input(video)
        if getattr(self, "frozen", False):
            with torch.no_grad():
                features = self.model.extract_features(video, keep_seq=True)
            return self._flatten_features(features, video)
        return self._forward_impl(video)

    @staticmethod
    def _validate_input(video: torch.Tensor) -> None:
        if video.ndim != 5:
            raise ValueError(f"video must be [B,3,T,H,W], got {tuple(video.shape)}")
        if video.shape[1] != 3:
            raise ValueError(f"video channel dimension must be 3, got {video.shape[1]}")
        if tuple(video.shape[-2:]) != (224, 224):
            raise ValueError(f"MARLIN expects 224x224 frames, got {tuple(video.shape[-2:])}")
        if video.min().detach().item() < -1e-4 or video.max().detach().item() > 1.0001:
            raise ValueError(
                "MARLIN expects video values in [0,1]. Do not apply ImageNet "
                "normalization in data.py before this backbone."
            )

    def _flatten_features(
        self,
        features: torch.Tensor | tuple | list | dict,
        video: torch.Tensor,
    ) -> torch.Tensor:
        if isinstance(features, (tuple, list)):
            features = features[0]
        if isinstance(features, dict):
            for key in ("last_hidden_state", "features", "tokens"):
                if key in features:
                    features = features[key]
                    break
        if not torch.is_tensor(features):
            raise TypeError("MARLIN extract_features returned an unsupported object")
        if features.ndim == 5:
            return features.flatten(1, 3).contiguous()
        if features.ndim == 4:
            return features.flatten(1, 2).contiguous()
        if features.ndim != 3:
            raise RuntimeError(
                "MARLIN must return sequence tokens [B,N,C], "
                f"got {tuple(features.shape)}"
            )
        temporal_tokens = video.shape[2] // self.tubelet_size
        if temporal_tokens < 1 or features.shape[1] % temporal_tokens != 0:
            raise RuntimeError(
                f"Cannot reshape MARLIN tokens {tuple(features.shape)} into "
                f"{temporal_tokens} temporal groups"
            )
        return features.contiguous()


__all__ = ["MARLINBackbone", "SpatialAttentionPool"]
