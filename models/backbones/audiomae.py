from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from .base import TemporalBackbone
from .utils import load_checkpoint_file


def _get_1d_sincos(embed_dim: int, positions: np.ndarray) -> np.ndarray:
    if embed_dim % 2 != 0:
        raise ValueError("1D sin-cos embedding dimension must be even")
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / (10000**omega)
    out = np.einsum("m,d->md", positions.reshape(-1), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def _get_2d_sincos(embed_dim: int, grid_size: Tuple[int, int], cls_token: bool) -> np.ndarray:
    if embed_dim % 2 != 0:
        raise ValueError("2D sin-cos embedding dimension must be even")
    grid_t = np.arange(grid_size[0], dtype=np.float32)
    grid_f = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_f, grid_t)  # frequency first, then time
    grid = np.stack(grid, axis=0).reshape(2, -1)
    emb_t = _get_1d_sincos(embed_dim // 2, grid[1])
    emb_f = _get_1d_sincos(embed_dim // 2, grid[0])
    pos = np.concatenate([emb_t, emb_f], axis=1)
    if cls_token:
        pos = np.concatenate([np.zeros((1, embed_dim), dtype=np.float32), pos], axis=0)
    return pos.astype(np.float32)


class AudioPatchEmbed(nn.Module):
    def __init__(
        self,
        input_size: Tuple[int, int] = (1024, 128),
        patch_size: Tuple[int, int] = (16, 16),
        in_channels: int = 1,
        embed_dim: int = 768,
    ):
        super().__init__()
        self.input_size = tuple(input_size)
        self.patch_size = tuple(patch_size)
        if any(size % patch != 0 for size, patch in zip(self.input_size, self.patch_size)):
            raise ValueError(
                f"input_size={self.input_size} must be divisible by patch_size={self.patch_size}"
            )
        self.patch_hw = (
            self.input_size[0] // self.patch_size[0],
            self.input_size[1] // self.patch_size[1],
        )
        self.num_patches = self.patch_hw[0] * self.patch_hw[1]
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2).contiguous()


class AudioMAEBackbone(TemporalBackbone):
    """AudioMAE ViT-B encoder returning the complete final-layer patch grid.

    Input:
        [B, 1024, 128] normalized log-Mel filterbanks.
    Output:
        [B, 512, 768], retaining all 64 x 8 time-frequency patch tokens.

    The module names intentionally follow the official AudioMAE encoder
    (patch_embed, cls_token, pos_embed, blocks, norm), allowing official
    AudioSet-2M checkpoints to be loaded with decoder keys ignored.
    """

    def __init__(
        self,
        checkpoint: str | None,
        input_size=(1024, 128),
        patch_size=(16, 16),
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        freeze: bool = False,
        require_pretrained: bool = True,
    ):
        super().__init__()
        try:
            from timm.models.vision_transformer import Block
        except ImportError as exc:
            raise ImportError(
                "AudioMAEBackbone requires timm. Install it with: pip install timm"
            ) from exc

        self.out_dim = int(embed_dim)
        self.patch_embed = AudioPatchEmbed(
            input_size=tuple(input_size),
            patch_size=tuple(patch_size),
            in_channels=1,
            embed_dim=self.out_dim,
        )
        self.time_patches, self.freq_patches = self.patch_embed.patch_hw
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.out_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.patch_embed.num_patches + 1, self.out_dim),
            requires_grad=False,
        )
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=self.out_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                )
                for _ in range(depth)
            ]
        )
        self.norm = norm_layer(self.out_dim)
        self._initialize_weights()

        self.pretrained_loaded = False
        if checkpoint:
            self._load_pretrained(checkpoint)
        elif require_pretrained:
            raise ValueError(
                "AudioMAE checkpoint is required. Set audio_backbone.checkpoint "
                "to the official AudioSet-2M pretrained .pth file."
            )

        self.set_frozen(freeze)

    def set_frozen(self, frozen: bool) -> None:
        """Keep the analytical sin-cos positions fixed when fine-tuning."""

        super().set_frozen(frozen)
        self.pos_embed.requires_grad_(False)

    def _initialize_weights(self) -> None:
        position = _get_2d_sincos(
            self.out_dim,
            (self.time_patches, self.freq_patches),
            cls_token=True,
        )
        self.pos_embed.data.copy_(torch.from_numpy(position).unsqueeze(0))
        nn.init.xavier_uniform_(self.patch_embed.proj.weight.flatten(1))
        if self.patch_embed.proj.bias is not None:
            nn.init.zeros_(self.patch_embed.proj.bias)
        nn.init.normal_(self.cls_token, std=0.02)
        self.apply(self._init_module)

    @staticmethod
    def _init_module(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _load_pretrained(self, checkpoint: str) -> None:
        state = load_checkpoint_file(checkpoint)

        # Some transferred checkpoints store the encoder under a prefix.
        candidate_prefixes = (
            "audio_encoder.",
            "audio_backbone.",
            "encoder.",
        )
        for prefix in candidate_prefixes:
            matching = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
            if matching and any(key.startswith("blocks.") for key in matching):
                state = matching
                break

        # Decoder/pretraining-only parameters are intentionally discarded.
        allowed_roots = ("patch_embed.", "cls_token", "pos_embed", "blocks.", "norm.")
        encoder_state = {
            key: value
            for key, value in state.items()
            if key.startswith(allowed_roots)
        }
        if not encoder_state:
            raise RuntimeError(
                f"No AudioMAE encoder parameters found in checkpoint: {checkpoint}"
            )

        incompatible = self.load_state_dict(encoder_state, strict=False)
        important_missing = [
            key
            for key in incompatible.missing_keys
            if key.startswith(("patch_embed.", "blocks.", "norm."))
        ]
        if important_missing:
            preview = ", ".join(important_missing[:8])
            raise RuntimeError(
                "AudioMAE checkpoint is incompatible with this ViT-B encoder. "
                f"Missing important parameters such as: {preview}"
            )
        self.pretrained_loaded = True
        print(
            f"[INFO] loaded AudioMAE encoder: {Path(checkpoint)} "
            f"({len(encoder_state)} tensors)"
        )

    def _forward_impl(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.ndim != 3:
            raise ValueError(f"audio must be [B,T,F], got {tuple(audio.shape)}")
        expected = self.patch_embed.input_size
        if tuple(audio.shape[1:]) != expected:
            raise ValueError(
                f"AudioMAE expects [B,{expected[0]},{expected[1]}], "
                f"got {tuple(audio.shape)}"
            )

        x = self.patch_embed(audio.unsqueeze(1))
        x = x + self.pos_embed[:, 1:, :].to(dtype=x.dtype)
        cls = (self.cls_token + self.pos_embed[:, :1, :]).to(dtype=x.dtype)
        x = torch.cat([cls.expand(x.shape[0], -1, -1), x], dim=1)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        # Match HAVIC's fine-tuning encoder path: remove only the global CLS
        # token and preserve every final-layer time-frequency patch token.
        return x[:, 1:, :].contiguous()

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        if getattr(self, "frozen", False):
            with torch.no_grad():
                return self._forward_impl(audio)
        return self._forward_impl(audio)
