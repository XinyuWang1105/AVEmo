from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbones import build_audio_backbone, build_video_backbone
from .backbones.tuning import adapter_parameter_ids, apply_backbone_tuning


TRANSFERRED_PREFIXES = (
    "audio_backbone.",
    "video_backbone.",
    "audio_projector.",
    "video_projector.",
)

CONSISTENCY_PRETRAIN_PREFIXES = TRANSFERRED_PREFIXES + (
    "emotion_extractor.",
    "fusion.",
)

FINETUNE_BASE_PREFIXES = TRANSFERRED_PREFIXES


def _resolve_pretrained_prefixes(prefixes=None) -> tuple[str, ...]:
    if prefixes is None:
        return TRANSFERRED_PREFIXES
    if isinstance(prefixes, str):
        key = prefixes.lower()
        aliases = {
            "stage1": TRANSFERRED_PREFIXES,
            "transfer": TRANSFERRED_PREFIXES,
            "consistency": CONSISTENCY_PRETRAIN_PREFIXES,
            "av_consistency": CONSISTENCY_PRETRAIN_PREFIXES,
        }
        if key in aliases:
            return aliases[key]
        prefixes = [item.strip() for item in prefixes.split(",") if item.strip()]
    result = []
    for prefix in prefixes:
        prefix = str(prefix)
        result.append(prefix if prefix.endswith(".") else f"{prefix}.")
    if not result:
        raise ValueError("pretrained_prefixes must not be empty")
    return tuple(result)


def _matches_prefix(name: str, prefixes: tuple[str, ...]) -> bool:
    return any(name.startswith(prefix) for prefix in prefixes)


def _temporal_delta(tokens: torch.Tensor) -> torch.Tensor:
    first = torch.zeros_like(tokens[:, :1])
    return torch.cat([first, tokens[:, 1:] - tokens[:, :-1]], dim=1)


def _check_sequence(tokens: torch.Tensor, name: str) -> None:
    if tokens.ndim != 3:
        raise ValueError(f"{name} must be [B,T,C], got {tuple(tokens.shape)}")


class TemporalProjector(nn.Module):
    """Group backbone tokens into aligned temporal segments, keeping C=768."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_segments: int,
        kernel_size: int = 3,
    ):
        super().__init__()
        self.num_segments = int(num_segments)
        if self.num_segments < 2:
            raise ValueError("num_segments must be at least 2")
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")

        self.channel_projection = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
        )
        self.temporal_projection = nn.Sequential(
            nn.Conv1d(
                hidden_dim,
                hidden_dim,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=hidden_dim,
                bias=False,
            ),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1, bias=False),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        _check_sequence(tokens, "tokens")
        tokens = self.channel_projection(tokens)
        tokens = self._align_time(tokens)
        local = self.temporal_projection(tokens.transpose(1, 2)).transpose(1, 2)
        return self.output_norm(tokens + local).contiguous()

    def _align_time(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.shape[1] == self.num_segments:
            return tokens

        batch, length, channels = tokens.shape
        if length % self.num_segments == 0:
            group_size = length // self.num_segments
            return tokens.reshape(batch, self.num_segments, group_size, channels).mean(dim=2)

        raise ValueError(
            f"Cannot align {length} tokens into {self.num_segments} segments "
            "without uneven temporal pooling"
        )


class FeedForward(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.net(tokens)


class ResidualCrossAttention(nn.Module):
    """Cross-modal attention with a 768-dimensional residual path."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        _check_sequence(query, "query")
        _check_sequence(context, "context")
        context = self.context_norm(context)
        attended, _ = self.attention(
            query=self.query_norm(query),
            key=context,
            value=context,
            need_weights=False,
        )
        return self.output_norm(query + self.dropout(attended))


class CrossModalAffineModulation(nn.Module):
    """Generate alpha/beta and modulate the dominant emotion sequence."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.to_alpha_beta = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 2),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

        linear = self.to_alpha_beta[-1]
        nn.init.normal_(linear.weight, std=1e-3)
        nn.init.zeros_(linear.bias)

    def forward(
        self,
        dominant: torch.Tensor,
        guidance: torch.Tensor,
    ) -> torch.Tensor:
        alpha, beta = self.to_alpha_beta(guidance).chunk(2, dim=-1)
        modulated = (1.0 + torch.tanh(alpha)) * dominant + beta
        return self.output_norm(modulated)


class MambaStateSpaceLayer(nn.Module):
    """Official Mamba SSM used by the visual micro-expression branch."""

    def __init__(
        self,
        hidden_dim: int,
        d_state: int,
        d_conv: int,
        expand: int,
        variant: str,
    ):
        super().__init__()
        try:
            from mamba_ssm import Mamba
        except ImportError as exc:
            raise ImportError(
                "AVEmo requires the official mamba-ssm package for its SSM "
                "branch. Install mamba-ssm before constructing this model."
            ) from exc

        variant = str(variant).lower()
        if variant == "mamba2":
            try:
                from mamba_ssm import Mamba2
            except ImportError as exc:
                raise ImportError(
                    "ssm_variant='mamba2' was requested, but this mamba-ssm "
                    "installation does not expose Mamba2."
                ) from exc
            block_cls = Mamba2
        elif variant == "mamba":
            block_cls = Mamba
        else:
            raise ValueError("ssm_variant must be 'mamba' or 'mamba2'")

        self.input_norm = nn.LayerNorm(hidden_dim)
        self.ssm = block_cls(
            d_model=hidden_dim,
            d_state=int(d_state),
            d_conv=int(d_conv),
            expand=int(expand),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.ssm(self.input_norm(tokens))


class MicroExpressionSSMUnit(nn.Module):
    """Local facial variation followed by official SSM temporal modeling."""

    def __init__(
        self,
        hidden_dim: int,
        dropout: float,
        kernel_size: int,
        ssm_backend: str,
        ssm_variant: str,
        ssm_d_state: int,
        ssm_d_conv: int,
        ssm_expand: int,
    ):
        super().__init__()
        if str(ssm_backend).lower() != "mamba":
            raise ValueError("AVEmo only supports ssm_backend='mamba'")
        if kernel_size % 2 == 0:
            raise ValueError("ssm_kernel_size must be odd")

        self.input_norm = nn.LayerNorm(hidden_dim)
        self.local_change = nn.Sequential(
            nn.Conv1d(
                hidden_dim,
                hidden_dim,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=hidden_dim,
            ),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
        )
        self.state_space = MambaStateSpaceLayer(
            hidden_dim=hidden_dim,
            d_state=ssm_d_state,
            d_conv=ssm_d_conv,
            expand=ssm_expand,
            variant=ssm_variant,
        )
        self.dropout = nn.Dropout(dropout)
        self.feed_forward = FeedForward(hidden_dim, dropout)
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        residual = tokens
        tokens = self.input_norm(tokens)
        tokens = self.local_change(tokens.transpose(1, 2)).transpose(1, 2)
        tokens = residual + self.dropout(self.state_space(tokens))
        tokens = tokens + self.feed_forward(tokens)
        return self.output_norm(tokens)


def _resolve_slstm_at(value, num_blocks: int) -> list[int]:
    if value is None:
        return [max(0, int(num_blocks) - 1)]
    if isinstance(value, str):
        lowered = value.lower()
        if lowered == "all":
            return list(range(int(num_blocks)))
        if lowered in {"none", ""}:
            return []
        raise ValueError("xlstm_slstm_at must be a sequence, 'all', or 'none'")
    return [int(index) for index in value]


class OfficialXLSTMSequenceLayer(nn.Module):
    """Official xLSTM block stack used by the audio context branch."""

    def __init__(
        self,
        hidden_dim: int,
        context_length: int,
        num_heads: int,
        num_blocks: int,
        dropout: float,
        slstm_at,
        slstm_backend: str,
        conv1d_kernel_size: int,
        qkv_proj_blocksize: int,
    ):
        super().__init__()
        try:
            from xlstm import (
                FeedForwardConfig,
                mLSTMBlockConfig,
                mLSTMLayerConfig,
                sLSTMBlockConfig,
                sLSTMLayerConfig,
                xLSTMBlockStack,
                xLSTMBlockStackConfig,
            )
        except ImportError as exc:
            raise ImportError(
                "AVEmo requires the official xlstm package for its audio "
                "context branch. Install xlstm before constructing this model."
            ) from exc

        self.input_norm = nn.LayerNorm(hidden_dim)
        config = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    conv1d_kernel_size=int(conv1d_kernel_size),
                    qkv_proj_blocksize=int(qkv_proj_blocksize),
                    num_heads=int(num_heads),
                )
            ),
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    backend=str(slstm_backend),
                    num_heads=int(num_heads),
                    conv1d_kernel_size=int(conv1d_kernel_size),
                    bias_init="powerlaw_blockdependent",
                ),
                feedforward=FeedForwardConfig(
                    proj_factor=1.3,
                    act_fn="gelu",
                ),
            ),
            context_length=int(context_length),
            num_blocks=int(num_blocks),
            embedding_dim=int(hidden_dim),
            slstm_at=_resolve_slstm_at(slstm_at, int(num_blocks)),
            dropout=float(dropout),
        )
        self.xlstm = xLSTMBlockStack(config)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.xlstm(self.input_norm(tokens))


class ContextEmotionXLSTMUnit(nn.Module):
    """xLSTM context, memory attention, and gated emotional memory."""

    def __init__(
        self,
        hidden_dim: int,
        dropout: float,
        context_length: int,
        num_heads: int,
        xlstm_backend: str,
        xlstm_blocks: int,
        xlstm_slstm_at,
        xlstm_slstm_backend: str,
        xlstm_conv1d_kernel_size: int,
        xlstm_qkv_proj_blocksize: int,
    ):
        super().__init__()
        if str(xlstm_backend).lower() != "official":
            raise ValueError("AVEmo only supports xlstm_backend='official'")

        self.context_encoder = OfficialXLSTMSequenceLayer(
            hidden_dim=hidden_dim,
            context_length=context_length,
            num_heads=num_heads,
            num_blocks=xlstm_blocks,
            dropout=dropout,
            slstm_at=xlstm_slstm_at,
            slstm_backend=xlstm_slstm_backend,
            conv1d_kernel_size=xlstm_conv1d_kernel_size,
            qkv_proj_blocksize=xlstm_qkv_proj_blocksize,
        )
        self.memory_attention = nn.Linear(hidden_dim * 2, hidden_dim)
        self.retain_gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.update_gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.candidate_gate = nn.Linear(hidden_dim * 2, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        residual = tokens
        context = self.context_encoder(tokens)
        enhanced = self._memory_attention(context)
        memory = self._gated_memory(enhanced)
        memory = self.output_projection(memory)
        return self.output_norm(residual + self.dropout(memory))

    def _memory_attention(self, context: torch.Tensor) -> torch.Tensor:
        previous = torch.cat([context[:, :1], context[:, :-1]], dim=1)
        attention = torch.sigmoid(
            self.memory_attention(torch.cat([previous, context], dim=-1))
        )
        return attention * context

    def _gated_memory(self, enhanced: torch.Tensor) -> torch.Tensor:
        memory = torch.zeros(
            enhanced.shape[0],
            enhanced.shape[2],
            device=enhanced.device,
            dtype=enhanced.dtype,
        )
        outputs = []
        for current in enhanced.unbind(dim=1):
            gate_input = torch.cat([current, memory], dim=-1)
            retain = torch.sigmoid(self.retain_gate(gate_input))
            update = torch.sigmoid(self.update_gate(gate_input))
            candidate = torch.tanh(self.candidate_gate(gate_input))
            memory = retain * memory + update * candidate
            outputs.append(memory)
        return torch.stack(outputs, dim=1)


class AVEmoInteractionLayer(nn.Module):
    """One video-dominant and audio-dominant emotion interaction layer."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        context_length: int,
        ssm_kernel_size: int,
        ssm_backend: str,
        ssm_variant: str,
        ssm_d_state: int,
        ssm_d_conv: int,
        ssm_expand: int,
        xlstm_backend: str,
        xlstm_blocks: int,
        xlstm_slstm_at,
        xlstm_slstm_backend: str,
        xlstm_conv1d_kernel_size: int,
        xlstm_qkv_proj_blocksize: int,
        use_me: bool = True,
        use_ac: bool = True,
        use_ddi: bool = True,
    ):
        super().__init__()
        self.use_me = bool(use_me)
        self.use_ac = bool(use_ac)
        self.use_ddi = bool(use_ddi)

        self.video_unit = (
            MicroExpressionSSMUnit(
                hidden_dim=hidden_dim,
                dropout=dropout,
                kernel_size=ssm_kernel_size,
                ssm_backend=ssm_backend,
                ssm_variant=ssm_variant,
                ssm_d_state=ssm_d_state,
                ssm_d_conv=ssm_d_conv,
                ssm_expand=ssm_expand,
            )
            if self.use_me
            else nn.Identity()
        )
        self.audio_unit = (
            ContextEmotionXLSTMUnit(
                hidden_dim=hidden_dim,
                dropout=dropout,
                context_length=context_length,
                num_heads=num_heads,
                xlstm_backend=xlstm_backend,
                xlstm_blocks=xlstm_blocks,
                xlstm_slstm_at=xlstm_slstm_at,
                xlstm_slstm_backend=xlstm_slstm_backend,
                xlstm_conv1d_kernel_size=xlstm_conv1d_kernel_size,
                xlstm_qkv_proj_blocksize=xlstm_qkv_proj_blocksize,
            )
            if self.use_ac
            else nn.Identity()
        )

        if self.use_ddi:
            self.video_queries_audio = ResidualCrossAttention(
                hidden_dim,
                num_heads,
                dropout,
            )
            self.audio_queries_video = ResidualCrossAttention(
                hidden_dim,
                num_heads,
                dropout,
            )
            self.video_modulation = CrossModalAffineModulation(hidden_dim)
            self.audio_modulation = CrossModalAffineModulation(hidden_dim)

    def forward(
        self,
        audio: torch.Tensor,
        video: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        video_emotion = self.video_unit(video)
        audio_emotion = self.audio_unit(audio)

        if not self.use_ddi:
            return audio_emotion, video_emotion

        audio_guidance = self.video_queries_audio(video_emotion, audio_emotion)
        video_guidance = self.audio_queries_video(audio_emotion, video_emotion)
        video_next = self.video_modulation(video_emotion, audio_guidance)
        audio_next = self.audio_modulation(audio_emotion, video_guidance)
        return audio_next, video_next


class MultimodalEmotionExtractor(nn.Module):
    """Stack AVEmo interaction layers without changing feature width."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
        context_length: int,
        num_layers: int,
        ssm_kernel_size: int,
        ssm_backend: str,
        ssm_variant: str,
        ssm_d_state: int,
        ssm_d_conv: int,
        ssm_expand: int,
        xlstm_backend: str,
        xlstm_blocks: int,
        xlstm_slstm_at,
        xlstm_slstm_backend: str,
        xlstm_conv1d_kernel_size: int,
        xlstm_qkv_proj_blocksize: int,
        use_me: bool = True,
        use_ac: bool = True,
        use_ddi: bool = True,
    ):
        super().__init__()
        if int(num_layers) < 1:
            raise ValueError("num_layers must be at least 1")

        self.use_me = bool(use_me)
        self.use_ac = bool(use_ac)
        self.use_ddi = bool(use_ddi)

        self.layers = nn.ModuleList(
            [
                AVEmoInteractionLayer(
                    hidden_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    context_length=context_length,
                    ssm_kernel_size=ssm_kernel_size,
                    ssm_backend=ssm_backend,
                    ssm_variant=ssm_variant,
                    ssm_d_state=ssm_d_state,
                    ssm_d_conv=ssm_d_conv,
                    ssm_expand=ssm_expand,
                    xlstm_backend=xlstm_backend,
                    xlstm_blocks=xlstm_blocks,
                    xlstm_slstm_at=xlstm_slstm_at,
                    xlstm_slstm_backend=xlstm_slstm_backend,
                    xlstm_conv1d_kernel_size=xlstm_conv1d_kernel_size,
                    xlstm_qkv_proj_blocksize=xlstm_qkv_proj_blocksize,
                    use_me=self.use_me,
                    use_ac=self.use_ac,
                    use_ddi=self.use_ddi,
                )
                for _ in range(int(num_layers))
            ]
        )

    def forward(
        self,
        audio: torch.Tensor,
        video: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for layer in self.layers:
            audio, video = layer(audio, video)
        return audio, video


class ChangeEnhancedProjection(nn.Module):
    """Project concatenated emotion state and temporal change to C=768."""

    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.projection = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.projection(torch.cat([tokens, _temporal_delta(tokens)], dim=-1))


class AttentionBottleneckFusion(nn.Module):
    """Align audio/video emotion state and change through temporal bottlenecks."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        num_segments: int,
        dropout: float,
    ):
        super().__init__()
        if int(num_segments) < 2:
            raise ValueError("num_segments must be at least 2")
        self.time_anchor = nn.Parameter(torch.zeros(1, int(num_segments), hidden_dim))
        self.audio_change = ChangeEnhancedProjection(hidden_dim, dropout)
        self.video_change = ChangeEnhancedProjection(hidden_dim, dropout)
        self.audio_bottleneck = ResidualCrossAttention(hidden_dim, num_heads, dropout)
        self.video_bottleneck = ResidualCrossAttention(hidden_dim, num_heads, dropout)
        self.audio_to_video = ResidualCrossAttention(hidden_dim, num_heads, dropout)
        self.video_to_audio = ResidualCrossAttention(hidden_dim, num_heads, dropout)
        self.audio_visual_projection = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        nn.init.normal_(self.time_anchor, std=0.02)

    def forward(
        self,
        audio_emotion: torch.Tensor,
        video_emotion: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        anchor = self.time_anchor.to(
            device=audio_emotion.device,
            dtype=audio_emotion.dtype,
        ).expand(audio_emotion.shape[0], -1, -1)

        audio_state = self.audio_bottleneck(anchor, self.audio_change(audio_emotion))
        video_state = self.video_bottleneck(anchor, self.video_change(video_emotion))
        audio_aligned = self.audio_to_video(audio_state, video_state)
        video_aligned = self.video_to_audio(video_state, audio_state)
        fused = self.audio_visual_projection(
            torch.cat([audio_aligned, video_aligned], dim=-1)
        )

        return {
            "av_feature": fused.mean(dim=1),
            "audio_bottleneck": audio_state,
            "video_bottleneck": video_state,
        }


class EmotionPredictionHead(nn.Module):
    """A 768-dimensional feature normalization followed by binary prediction."""

    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.feature_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, 1)

    def encode(self, feature: torch.Tensor) -> torch.Tensor:
        return self.feature_norm(feature)

    def forward(self, feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature = self.encode(feature)
        logit = self.classifier(self.dropout(feature)).squeeze(-1)
        return feature, logit


class AVEmo(nn.Module):
    """Audio-Visual Emotion Consistency detector."""

    def __init__(
        self,
        audio_backbone: dict,
        video_backbone: dict,
        hidden_dim: int = 768,
        num_segments: int = 8,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
        ssm_kernel_size: int = 3,
        ssm_backend: str = "mamba",
        ssm_variant: str = "mamba",
        ssm_d_state: int = 16,
        ssm_d_conv: int = 4,
        ssm_expand: int = 2,
        xlstm_backend: str = "official",
        xlstm_blocks: int = 2,
        xlstm_slstm_at=None,
        xlstm_slstm_backend: str = "vanilla",
        xlstm_conv1d_kernel_size: int = 4,
        xlstm_qkv_proj_blocksize: int = 4,
        pretrained_path: str | None = None,
        require_pretrained: bool = False,
        pretrained_prefixes=None,
        backbone_tuning: dict | None = None,
        binary_only: bool = False,
        use_me: bool = True,
        use_ac: bool = True,
        use_ddi: bool = True,
    ):
        super().__init__()
        if int(hidden_dim) != 768:
            raise ValueError("AVEmo keeps the paper feature width fixed at 768")
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.audio_backbone = build_audio_backbone(audio_backbone)
        self.video_backbone = build_video_backbone(video_backbone)
        self.audio_projector = TemporalProjector(
            self.audio_backbone.out_dim,
            hidden_dim,
            num_segments,
        )
        self.video_projector = TemporalProjector(
            self.video_backbone.out_dim,
            hidden_dim,
            num_segments,
        )
        self.emotion_extractor = MultimodalEmotionExtractor(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            context_length=num_segments,
            num_layers=num_layers,
            ssm_kernel_size=ssm_kernel_size,
            ssm_backend=ssm_backend,
            ssm_variant=ssm_variant,
            ssm_d_state=ssm_d_state,
            ssm_d_conv=ssm_d_conv,
            ssm_expand=ssm_expand,
            xlstm_backend=xlstm_backend,
            xlstm_blocks=xlstm_blocks,
            xlstm_slstm_at=xlstm_slstm_at,
            xlstm_slstm_backend=xlstm_slstm_backend,
            xlstm_conv1d_kernel_size=xlstm_conv1d_kernel_size,
            xlstm_qkv_proj_blocksize=xlstm_qkv_proj_blocksize,
            use_me=use_me,
            use_ac=use_ac,
            use_ddi=use_ddi,
        )
        self.fusion = AttentionBottleneckFusion(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_segments=num_segments,
            dropout=dropout,
        )
        self.audio_head = EmotionPredictionHead(hidden_dim, dropout)
        self.video_head = EmotionPredictionHead(hidden_dim, dropout)
        self.fusion_head = EmotionPredictionHead(hidden_dim, dropout)
        self.binary_only = bool(binary_only)
        self.ablation = {
            "use_me": bool(use_me),
            "use_ac": bool(use_ac),
            "use_ddi": bool(use_ddi),
        }
        if self.binary_only:
            for head in (self.audio_head, self.video_head):
                head.classifier.requires_grad_(False)

        self.stage1_pretrained_loaded = False
        self.pretrained_module_prefixes = ()
        if pretrained_path:
            self.load_pretrained_modules(
                pretrained_path,
                required=require_pretrained,
                prefixes=pretrained_prefixes,
            )
        elif require_pretrained:
            raise ValueError("require_pretrained=True, but pretrained_path is empty")

        self.backbone_tuning = self._apply_backbone_tuning(backbone_tuning)

    def _apply_backbone_tuning(self, config: dict | None) -> dict[str, dict]:
        tuning_cfg = dict(config or {})
        unknown = set(tuning_cfg).difference({"audio", "video"})
        if unknown:
            raise ValueError(f"Unknown backbone_tuning options: {sorted(unknown)}")

        reports = {
            "audio": apply_backbone_tuning(
                self.audio_backbone,
                tuning_cfg.get("audio"),
                "audio",
            ),
            "video": apply_backbone_tuning(
                self.video_backbone,
                tuning_cfg.get("video"),
                "video",
            ),
        }
        for modality, report in reports.items():
            print(
                f"[INFO] {modality} backbone tuning: "
                + ", ".join(f"{key}={value}" for key, value in report.items())
            )
        return reports

    def encode_backbone_tokens(
        self,
        audio: torch.Tensor,
        video: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.audio_backbone(audio), self.video_backbone(video)

    def project_sequences(
        self,
        audio_tokens: torch.Tensor,
        video_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.audio_projector(audio_tokens), self.video_projector(video_tokens)

    def encode_sequences(
        self,
        audio: torch.Tensor,
        video: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        audio_tokens, video_tokens = self.encode_backbone_tokens(audio, video)
        return self.project_sequences(audio_tokens, video_tokens)

    def encode_emotions(
        self,
        audio: torch.Tensor,
        video: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        audio_sequence, video_sequence = self.encode_sequences(audio, video)
        audio_emotion, video_emotion = self.emotion_extractor(
            audio_sequence,
            video_sequence,
        )
        return audio_sequence, video_sequence, audio_emotion, video_emotion

    def forward(
        self,
        audio: torch.Tensor,
        video: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        audio_sequence, video_sequence = self.encode_sequences(audio, video)
        audio_emotion, video_emotion = self.emotion_extractor(
            audio_sequence,
            video_sequence,
        )
        fusion = self.fusion(audio_emotion, video_emotion)

        _, fusion_logit = self.fusion_head(fusion["av_feature"])
        content_score = F.l1_loss(
            audio_emotion,
            video_emotion,
            reduction="none",
        ).mean(dim=(1, 2))
        temporal_sync_score = F.mse_loss(
            fusion["audio_bottleneck"],
            fusion["video_bottleneck"],
            reduction="none",
        ).mean(dim=(1, 2))
        output = {
            "logit": fusion_logit,
            "content_score": content_score,
            "temporal_sync_score": temporal_sync_score,
            # Backward-compatible aliases for existing evaluation scripts.
            "state_score": content_score,
            "dynamic_score": temporal_sync_score,
        }
        if not self.binary_only:
            _, output["audio_logit"] = self.audio_head(audio_emotion.mean(dim=1))
            _, output["video_logit"] = self.video_head(video_emotion.mean(dim=1))
        return output

    def pretrained_state_dict(self, prefixes=None) -> dict[str, torch.Tensor]:
        prefixes = _resolve_pretrained_prefixes(prefixes)
        return {
            name: tensor.detach().cpu()
            for name, tensor in self.state_dict().items()
            if _matches_prefix(name, prefixes)
        }

    def load_pretrained_modules(
        self,
        checkpoint_path: str,
        required: bool = True,
        prefixes=None,
    ) -> None:
        path = Path(checkpoint_path)
        if not path.is_file():
            if required:
                raise FileNotFoundError(f"AVEmo pretrained checkpoint not found: {path}")
            print(f"[WARN] AVEmo pretrained checkpoint not found: {path}")
            return

        checkpoint = torch.load(path, map_location="cpu")
        source_state = checkpoint.get("model", checkpoint)
        target_state = self.state_dict()
        prefixes = _resolve_pretrained_prefixes(prefixes)
        compatible = {}
        skipped = []
        for raw_name, tensor in source_state.items():
            name = raw_name
            for prefix in ("module.", "avemo.", "ecn."):
                if name.startswith(prefix):
                    name = name[len(prefix):]

            if not _matches_prefix(name, prefixes):
                continue
            if name not in target_state:
                skipped.append((name, "missing"))
                continue
            if tensor.shape != target_state[name].shape:
                skipped.append(
                    (
                        name,
                        f"shape {tuple(tensor.shape)} != {tuple(target_state[name].shape)}",
                    )
                )
                continue
            compatible[name] = tensor

        expected = {
            name
            for name in target_state
            if _matches_prefix(name, prefixes)
        }
        missing = sorted(expected.difference(compatible))
        if not compatible:
            raise RuntimeError(f"No compatible AVEmo pretrained parameters found in {path}")
        if required and missing:
            preview = ", ".join(missing[:5])
            suffix = " ..." if len(missing) > 5 else ""
            raise RuntimeError(
                "AVEmo pretrained checkpoint is incomplete: "
                f"missing {len(missing)}/{len(expected)} transferred tensors "
                f"({preview}{suffix}) in {path}"
            )

        self.load_state_dict(compatible, strict=False)
        self.stage1_pretrained_loaded = not missing
        self.pretrained_module_prefixes = prefixes
        loaded_counts = {
            prefix.rstrip("."): sum(
                name.startswith(prefix)
                for name in compatible
            )
            for prefix in prefixes
        }
        counts_text = ", ".join(
            f"{name}={count}"
            for name, count in loaded_counts.items()
        )
        print(
            f"[INFO] loaded AVEmo pretraining from {path} "
            f"({len(compatible)} tensors; {counts_text}; skipped={len(skipped)})"
        )

    def parameter_groups(
        self,
        backbone_lr: float,
        module_lr: float,
        adapter_lr: float | None = None,
    ) -> list[dict]:
        adapter_ids = adapter_parameter_ids(self)
        pretrained_parameters = []
        adapter_parameters = []
        new_module_parameters = []
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            if id(parameter) in adapter_ids:
                adapter_parameters.append(parameter)
            elif name.startswith(("audio_backbone.", "video_backbone.")):
                pretrained_parameters.append(parameter)
            elif self.pretrained_module_prefixes and _matches_prefix(
                name,
                self.pretrained_module_prefixes,
            ):
                pretrained_parameters.append(parameter)
            else:
                new_module_parameters.append(parameter)

        groups = []
        if pretrained_parameters:
            groups.append(
                {
                    "params": pretrained_parameters,
                    "lr": float(backbone_lr),
                    "group_name": "pretrained_modules",
                }
            )
        if adapter_parameters:
            groups.append(
                {
                    "params": adapter_parameters,
                    "lr": float(adapter_lr if adapter_lr is not None else module_lr),
                    "weight_decay": 0.0,
                    "group_name": "backbone_adapters",
                }
            )
        if new_module_parameters:
            groups.append(
                {
                    "params": new_module_parameters,
                    "lr": float(module_lr),
                    "group_name": "new_modules",
                }
            )
        if not groups:
            raise RuntimeError("The model has no trainable parameters")

        grouped_ids = [id(p) for group in groups for p in group["params"]]
        expected_ids = {
            id(parameter)
            for parameter in self.parameters()
            if parameter.requires_grad
        }
        if len(grouped_ids) != len(set(grouped_ids)):
            raise RuntimeError("A parameter appears in multiple optimizer groups")
        if set(grouped_ids) != expected_ids:
            raise RuntimeError("Optimizer groups do not cover all trainable parameters")
        return groups


__all__ = [
    "AVEmo",
    "TRANSFERRED_PREFIXES",
    "FINETUNE_BASE_PREFIXES",
]
