from __future__ import annotations

from copy import deepcopy

from .avemo import AVEmo
from .avemo_ablation import AVEmo as AVEmo0

MODEL_REGISTRY = {
    "avemo": AVEmo,
    "avemo-ablation":AVEmo0
}


def build_model(name: str, params: dict):
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model {name!r}. Available: {list(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[name](**deepcopy(params))


__all__ = [
    "build_model",
    "AVEmo",
    "AVEmo0",
]
