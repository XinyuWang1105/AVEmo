from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

from .base import TemporalBackbone


AUDIO_BACKBONES = {}
VIDEO_BACKBONES = {}


def register_audio_backbone(name):
    def decorator(cls):
        AUDIO_BACKBONES[name] = cls
        return cls
    return decorator


def register_video_backbone(name):
    def decorator(cls):
        VIDEO_BACKBONES[name] = cls
        return cls
    return decorator


def _register_builtin_backbones() -> None:
    if AUDIO_BACKBONES and VIDEO_BACKBONES:
        return

    from .simple import SimpleAudioBackbone, SimpleVideoBackbone
    from .audiomae import AudioMAEBackbone
    from .clip import CLIPVideoBackbone
    from .marlin import MARLINBackbone

    AUDIO_BACKBONES.update(
        {
            "simple_audio": SimpleAudioBackbone,
            "audiomae_vitb": AudioMAEBackbone,
        }
    )
    VIDEO_BACKBONES.update(
        {
            "simple_video": SimpleVideoBackbone,
            "marlin_vitb": MARLINBackbone,
            "clip": CLIPVideoBackbone,
            "clip14": CLIPVideoBackbone,
            "clip_l14": CLIPVideoBackbone,
            "clip_vit_l14": CLIPVideoBackbone,
        }
    )


def build_audio_backbone(config: Dict[str, Any]) -> TemporalBackbone:
    _register_builtin_backbones()
    cfg = deepcopy(config)
    name = cfg.pop("name")
    if name not in AUDIO_BACKBONES:
        raise ValueError(
            f"Unknown audio backbone {name!r}. Available: {list(AUDIO_BACKBONES)}"
        )
    return AUDIO_BACKBONES[name](**cfg)


def build_video_backbone(config: Dict[str, Any]) -> TemporalBackbone:
    _register_builtin_backbones()
    cfg = deepcopy(config)
    name = cfg.pop("name")
    if name not in VIDEO_BACKBONES:
        raise ValueError(
            f"Unknown video backbone {name!r}. Available: {list(VIDEO_BACKBONES)}"
        )
    return VIDEO_BACKBONES[name](**cfg)


__all__ = [
    "TemporalBackbone",
    "build_audio_backbone",
    "build_video_backbone",
    "register_audio_backbone",
    "register_video_backbone",
]
