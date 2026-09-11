from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Mapping, MutableMapping, Tuple

import torch


def unwrap_checkpoint(payload):
    """Return a state dict from common checkpoint containers."""
    if isinstance(payload, Mapping):
        for key in ("model", "state_dict", "module", "encoder"):
            value = payload.get(key)
            if isinstance(value, Mapping):
                return dict(value)
        if payload and all(isinstance(key, str) for key in payload.keys()):
            return dict(payload)
    raise TypeError("Checkpoint does not contain a recognizable state dict")


def strip_prefix_if_present(state_dict: MutableMapping[str, torch.Tensor], prefix: str):
    if not state_dict:
        return dict(state_dict)
    if all(key.startswith(prefix) for key in state_dict):
        return {key[len(prefix):]: value for key, value in state_dict.items()}
    return dict(state_dict)


def normalize_state_dict(state_dict: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    state = dict(state_dict)
    for prefix in ("module.", "model."):
        state = strip_prefix_if_present(state, prefix)
    return state


def load_checkpoint_file(path: str | Path) -> Dict[str, torch.Tensor]:
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu")
    return normalize_state_dict(unwrap_checkpoint(payload))


def filter_state_dict_by_prefixes(
    state_dict: Mapping[str, torch.Tensor],
    ignored_prefixes: Iterable[str],
) -> Dict[str, torch.Tensor]:
    prefixes = tuple(ignored_prefixes)
    return {
        key: value
        for key, value in state_dict.items()
        if not key.startswith(prefixes)
    }
