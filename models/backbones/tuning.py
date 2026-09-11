from __future__ import annotations

import math
import re
from collections.abc import Iterable

import torch
import torch.nn as nn
from torch.nn.utils import parametrize


class LowRankAdditiveWeight(nn.Module):
    """LoRA-style additive update applied directly to a Linear weight.

    Weight parametrization is intentional: MARLIN's attention implementation
    reads ``qkv.weight`` through ``F.linear`` instead of calling ``qkv(x)``.
    Parametrizing the weight therefore works for both AudioMAE and MARLIN.
    """

    def __init__(
        self,
        out_features: int,
        in_features: int,
        rank: int,
        alpha: float,
    ) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError("LoRA rank must be positive")
        if alpha <= 0:
            raise ValueError("LoRA alpha must be positive")
        self.rank = int(rank)
        self.scale = float(alpha) / self.rank
        self.lora_A = nn.Parameter(torch.empty(self.rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, self.rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        update = self.lora_B @ self.lora_A
        return weight + update.to(dtype=weight.dtype) * self.scale


def _block_index(name: str) -> int | None:
    match = re.search(r"(?:^|\.)(?:blocks|layers)\.(\d+)(?:\.|$)", name)
    return int(match.group(1)) if match else None


def _last_block_indices(module: nn.Module, last_n: int) -> set[int]:
    if last_n < 1:
        raise ValueError("last_n must be positive")
    indices = {
        index
        for name, _ in module.named_modules()
        if (index := _block_index(name)) is not None
    }
    if not indices:
        raise RuntimeError(
            f"Could not find transformer blocks in {type(module).__name__}"
        )
    return set(sorted(indices)[-last_n:])


def _matches_target(name: str, targets: Iterable[str]) -> bool:
    return any(name == target or name.endswith(f".{target}") for target in targets)


def apply_backbone_tuning(
    backbone: nn.Module,
    config: dict | None,
    modality: str,
) -> dict[str, int | str]:
    """Apply full/frozen/last_n/LoRA tuning after all pretrained loads.

    A missing config preserves the legacy ``freeze`` option supplied to the
    backbone constructor.  Partial and LoRA strategies keep ``frozen=False``
    so the backbone forward pass is not wrapped in ``torch.no_grad``; precise
    trainability is controlled with ``requires_grad`` instead.
    """

    if config is None:
        return {
            "strategy": "legacy",
            "trainable": sum(p.numel() for p in backbone.parameters() if p.requires_grad),
            "adapters": 0,
        }

    cfg = dict(config)
    strategy = str(cfg.pop("strategy", "full")).lower()
    if strategy not in {"full", "frozen", "last_n", "lora"}:
        raise ValueError(
            f"Unknown {modality} backbone tuning strategy {strategy!r}; "
            "expected full, frozen, last_n, or lora"
        )

    if strategy == "full":
        if cfg:
            raise ValueError(f"Unused {modality} full-tuning options: {sorted(cfg)}")
        backbone.set_frozen(False)
        return {
            "strategy": strategy,
            "trainable": sum(p.numel() for p in backbone.parameters() if p.requires_grad),
            "adapters": 0,
        }

    if strategy == "frozen":
        if cfg:
            raise ValueError(f"Unused {modality} frozen-tuning options: {sorted(cfg)}")
        backbone.set_frozen(True)
        return {
            "strategy": strategy,
            "trainable": sum(p.numel() for p in backbone.parameters() if p.requires_grad),
            "adapters": 0,
        }

    backbone.set_frozen(False)
    backbone.requires_grad_(False)

    if strategy == "last_n":
        last_n = int(cfg.pop("last_n", 2))
        if cfg:
            raise ValueError(f"Unused {modality} last_n options: {sorted(cfg)}")
        selected = _last_block_indices(backbone, last_n)
        for name, parameter in backbone.named_parameters():
            index = _block_index(name)
            lower_name = name.lower()
            if (
                index in selected
                or (index is None and "norm" in lower_name)
            ):
                parameter.requires_grad_(True)
        trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
        if trainable == 0:
            raise RuntimeError(f"{modality} last_n strategy selected no parameters")
        return {"strategy": strategy, "trainable": trainable, "adapters": 0}

    rank = int(cfg.pop("rank", 8))
    alpha = float(cfg.pop("alpha", rank * 2))
    targets = tuple(cfg.pop("target_modules", ("attn.qkv", "attn.proj")))
    last_n_value = cfg.pop("last_n", None)
    if cfg:
        raise ValueError(f"Unused {modality} LoRA options: {sorted(cfg)}")
    if not targets:
        raise ValueError(f"{modality} LoRA target_modules cannot be empty")
    selected = (
        _last_block_indices(backbone, int(last_n_value))
        if last_n_value is not None
        else None
    )

    matched = []
    for name, module in list(backbone.named_modules()):
        if not isinstance(module, nn.Linear) or not _matches_target(name, targets):
            continue
        index = _block_index(name)
        if selected is not None and index not in selected:
            continue
        if parametrize.is_parametrized(module, "weight"):
            raise RuntimeError(f"Linear layer {name!r} is already parametrized")
        parametrize.register_parametrization(
            module,
            "weight",
            LowRankAdditiveWeight(
                module.out_features,
                module.in_features,
                rank,
                alpha,
            ),
        )
        module.parametrizations.weight.original.requires_grad_(False)
        if module.bias is not None:
            module.bias.requires_grad_(False)
        matched.append(name)

    if not matched:
        available = [
            name for name, module in backbone.named_modules()
            if isinstance(module, nn.Linear)
        ]
        preview = ", ".join(available[:12])
        raise RuntimeError(
            f"{modality} LoRA matched no Linear layers for targets {targets}. "
            f"Available examples: {preview}"
        )

    adapters = sum(
        p.numel()
        for module in backbone.modules()
        if isinstance(module, LowRankAdditiveWeight)
        for p in module.parameters()
    )
    trainable = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    return {
        "strategy": strategy,
        "trainable": trainable,
        "adapters": adapters,
        "matched": len(matched),
    }


def adapter_parameter_ids(module: nn.Module) -> set[int]:
    return {
        id(parameter)
        for child in module.modules()
        if isinstance(child, LowRankAdditiveWeight)
        for parameter in child.parameters(recurse=False)
        if parameter.requires_grad
    }


__all__ = [
    "LowRankAdditiveWeight",
    "adapter_parameter_ids",
    "apply_backbone_tuning",
]
