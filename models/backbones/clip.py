from __future__ import annotations

import logging
from pathlib import Path

import torch
import torch.nn as nn

from .base import TemporalBackbone


logger = logging.getLogger(__name__)


class CLIPVideoBackbone(TemporalBackbone):
    """HuggingFace CLIP visual encoder adapted as a video backbone.

    This follows the CLIP usage in CDL-Det: load ``CLIPModel`` from a local
    HuggingFace path or model name, keep the visual encoder frozen by default,
    and optionally enable LayerNorm tuning or q/v LoRA tuning.
    """

    CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
    CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        model_path: str | None = None,
        local_files_only: bool = True,
        freeze: bool | None = None,
        freeze_clip: bool = True,
        use_ln_tuning: bool = False,
        ln_tuning: bool | None = None,
        use_lora: bool = False,
        lora_r: int = 8,
        lora_alpha: float = 16,
        lora_dropout: float = 0.05,
        token_mode: str = "cls",
        normalize: bool = True,
        image_size: int = 224,
    ):
        super().__init__()
        try:
            from transformers import CLIPModel
        except ImportError as exc:
            raise ImportError(
                "CLIPVideoBackbone requires HuggingFace transformers. "
                "Install it with: pip install transformers"
            ) from exc

        model_name_or_path = self._resolve_model_name_or_path(model_path or model_name)
        clip_model = CLIPModel.from_pretrained(
            model_name_or_path,
            local_files_only=bool(local_files_only),
        )
        self.model_name_or_path = model_name_or_path
        self.vision_model = clip_model.vision_model
        self.out_dim = int(clip_model.config.vision_config.hidden_size)
        self.image_size = int(image_size)
        self.token_mode = str(token_mode).lower()
        if self.token_mode not in {"cls", "pooler", "patch"}:
            raise ValueError("token_mode must be one of: cls, pooler, patch")

        self.normalize = bool(normalize)
        mean = torch.tensor(self.CLIP_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(self.CLIP_STD, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("clip_mean", mean)
        self.register_buffer("clip_std", std)

        self.freeze_clip = bool(freeze_clip if freeze is None else freeze)
        self.use_ln_tuning = bool(use_ln_tuning if ln_tuning is None else ln_tuning)
        self.use_lora = bool(use_lora)

        if self.freeze_clip:
            self.vision_model.requires_grad_(False)

        layer_norm_count = self._enable_layer_norm_tuning()
        lora_module_count = self._enable_lora_tuning(
            rank=int(lora_r),
            alpha=float(lora_alpha),
            dropout=float(lora_dropout),
        )

        trainable = sum(p.numel() for p in self.vision_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.vision_model.parameters())
        self.frozen = trainable == 0
        logger.info(
            "Loaded CLIP visual backbone from %s | freeze=%s | LN=%s (%d) | "
            "LoRA=%s (%d) | token_mode=%s | trainable=%d/%d (%.4f%%)",
            model_name_or_path,
            self.freeze_clip,
            self.use_ln_tuning,
            layer_norm_count,
            self.use_lora,
            lora_module_count,
            self.token_mode,
            trainable,
            total,
            100.0 * trainable / max(total, 1),
        )
        print(
            "[INFO] loaded CLIP visual backbone: "
            f"{model_name_or_path} token_mode={self.token_mode} "
            f"trainable={trainable}/{total}"
        )

    @staticmethod
    def _resolve_model_name_or_path(model_name_or_path: str) -> str:
        path = Path(model_name_or_path)
        if not path.exists() or (path / "config.json").is_file():
            return model_name_or_path

        snapshots = path / "snapshots"
        refs_main = path / "refs" / "main"
        if refs_main.is_file():
            revision = refs_main.read_text(encoding="utf-8").strip()
            snapshot = snapshots / revision
            if (snapshot / "config.json").is_file():
                return str(snapshot)

        if snapshots.is_dir():
            candidates = [
                item
                for item in snapshots.iterdir()
                if item.is_dir() and (item / "config.json").is_file()
            ]
            if candidates:
                return str(sorted(candidates, key=lambda item: item.stat().st_mtime)[-1])

        return model_name_or_path

    def _enable_layer_norm_tuning(self) -> int:
        if not self.use_ln_tuning:
            return 0
        count = 0
        for module in self.vision_model.modules():
            if isinstance(module, nn.LayerNorm):
                count += 1
                for parameter in module.parameters():
                    parameter.requires_grad_(True)
        if count == 0:
            raise RuntimeError("LN tuning requested, but CLIP contains no nn.LayerNorm")
        return count

    def _enable_lora_tuning(self, rank: int, alpha: float, dropout: float) -> int:
        if not self.use_lora:
            return 0
        try:
            import loralib as lora
        except ImportError as exc:
            raise ImportError(
                "CLIP LoRA tuning requires loralib. Install it with: pip install loralib"
            ) from exc

        if not hasattr(self.vision_model, "encoder"):
            raise RuntimeError("CLIP vision model has no encoder for q/v LoRA tuning")
        count = 0
        for encoder_layer in self.vision_model.encoder.layers:
            attention = encoder_layer.self_attn
            for projection_name in ("q_proj", "v_proj"):
                projection = getattr(attention, projection_name)
                if isinstance(projection, lora.Linear):
                    continue
                replacement = lora.Linear(
                    projection.in_features,
                    projection.out_features,
                    r=rank,
                    lora_alpha=alpha,
                    lora_dropout=dropout,
                    bias=projection.bias is not None,
                    merge_weights=False,
                ).to(device=projection.weight.device, dtype=projection.weight.dtype)
                with torch.no_grad():
                    replacement.weight.copy_(projection.weight)
                    if projection.bias is not None:
                        replacement.bias.copy_(projection.bias)
                replacement.weight.requires_grad_(not self.freeze_clip)
                if replacement.bias is not None:
                    replacement.bias.requires_grad_(not self.freeze_clip)
                setattr(attention, projection_name, replacement)
                count += 1
        return count

    def set_frozen(self, frozen: bool) -> None:
        self.frozen = bool(frozen)
        self.vision_model.requires_grad_(not self.frozen)

    def train(self, mode: bool = True):
        nn.Module.train(self, mode)
        if getattr(self, "frozen", False):
            self.vision_model.train(False)
        return self

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        self._validate_input(video)
        if getattr(self, "frozen", False):
            with torch.no_grad():
                return self._forward_impl(video)
        return self._forward_impl(video)

    def _validate_input(self, video: torch.Tensor) -> None:
        if video.ndim != 5:
            raise ValueError(f"video must be [B,3,T,H,W], got {tuple(video.shape)}")
        if video.shape[1] != 3:
            raise ValueError(f"video channel dimension must be 3, got {video.shape[1]}")
        if tuple(video.shape[-2:]) != (self.image_size, self.image_size):
            raise ValueError(
                f"CLIP expects {self.image_size}x{self.image_size} frames, "
                f"got {tuple(video.shape[-2:])}"
            )
        if video.min().detach().item() < -1e-4 or video.max().detach().item() > 1.0001:
            raise ValueError("CLIPVideoBackbone expects video values in [0,1]")

    def _forward_impl(self, video: torch.Tensor) -> torch.Tensor:
        batch_size, _, frames, height, width = video.shape
        pixel_values = video.permute(0, 2, 1, 3, 4).reshape(
            batch_size * frames,
            3,
            height,
            width,
        )
        if self.normalize:
            mean = self.clip_mean.to(dtype=pixel_values.dtype)
            std = self.clip_std.to(dtype=pixel_values.dtype)
            pixel_values = (pixel_values - mean) / std

        output = self.vision_model(
            pixel_values=pixel_values,
            output_hidden_states=False,
            return_dict=True,
        )
        tokens = self._extract_tokens(output)
        if self.token_mode == "patch":
            return tokens.reshape(batch_size, frames * tokens.shape[1], -1).contiguous()
        return tokens.reshape(batch_size, frames, -1).contiguous()

    def _extract_tokens(self, output) -> torch.Tensor:
        if self.token_mode == "pooler" and hasattr(output, "pooler_output"):
            return output.pooler_output

        last_hidden_state = output.last_hidden_state
        if self.token_mode == "patch":
            return last_hidden_state[:, 1:, :]

        cls_token = last_hidden_state[:, 0, :]
        post_layernorm = getattr(self.vision_model, "post_layernorm", None)
        if post_layernorm is not None:
            cls_token = post_layernorm(cls_token)
        return cls_token


__all__ = ["CLIPVideoBackbone"]
