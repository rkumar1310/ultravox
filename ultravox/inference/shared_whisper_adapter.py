"""Gate and adapt shared Ultravox audio states for Whisper transcription."""

from __future__ import annotations

import dataclasses
import pathlib

import torch
import torch.nn as nn


class ShortStateAdapter(nn.Module):
    """Cross-attend fixed Whisper positions to a variable short audio state."""

    def __init__(
        self, hidden_size: int = 1280, adapter_size: int = 128, frames: int = 1500
    ) -> None:
        super().__init__()
        self.frames = frames
        self.input_norm = nn.LayerNorm(hidden_size)
        self.key_value = nn.Linear(hidden_size, adapter_size * 2, bias=False)
        self.queries = nn.Parameter(torch.empty(frames, adapter_size))
        self.attention = nn.MultiheadAttention(
            adapter_size, num_heads=4, batch_first=True
        )
        self.output = nn.Sequential(
            nn.LayerNorm(adapter_size),
            nn.Linear(adapter_size, hidden_size, bias=False),
        )
        self.position_bias = nn.Parameter(torch.zeros(frames, hidden_size))
        nn.init.normal_(self.queries, std=0.02)

    def forward(
        self,
        source: torch.Tensor,
        source_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        projected = self.key_value(self.input_norm(source))
        keys, values = projected.chunk(2, dim=-1)
        queries = self.queries.unsqueeze(0).expand(source.shape[0], -1, -1)
        attended, _ = self.attention(
            queries,
            keys,
            values,
            key_padding_mask=None if source_mask is None else ~source_mask.bool(),
            need_weights=False,
        )
        output = self.output(attended) + self.position_bias
        residual_frames = min(source.shape[1], self.frames)
        residual = source[:, :residual_frames]
        if source_mask is not None:
            residual = residual * source_mask[:, :residual_frames, None]
        output[:, :residual_frames] += residual
        return output


class SpeechPresenceHead(nn.Module):
    """Classify a variable-length shared encoder state as speech or no speech."""

    def __init__(self, hidden_size: int = 1280, projection_size: int = 192) -> None:
        super().__init__()
        self.frame_projection = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, projection_size),
            nn.GELU(),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(projection_size * 2),
            nn.Linear(projection_size * 2, projection_size),
            nn.GELU(),
            nn.Linear(projection_size, 1),
        )

    def forward(
        self,
        source: torch.Tensor,
        source_mask: torch.Tensor,
    ) -> torch.Tensor:
        if source.ndim != 3:
            raise ValueError("source must have shape [batch, frames, hidden]")
        if source_mask.shape != source.shape[:2]:
            raise ValueError("source_mask must match source batch and frame axes")
        if not bool(source_mask.any(dim=1).all()):
            raise ValueError("every sample must contain at least one valid frame")

        projected = self.frame_projection(source.float())
        mask = source_mask.bool()
        weights = mask.unsqueeze(-1).to(projected.dtype)
        mean = (projected * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
        maximum = projected.masked_fill(~mask.unsqueeze(-1), -torch.inf).amax(dim=1)
        pooled = torch.cat([mean, maximum], dim=-1)
        return self.classifier(pooled).squeeze(-1)


@dataclasses.dataclass(frozen=True)
class GatedWhisperAdapterOutput:
    """The gate decision plus adapted states for accepted batch rows only."""

    speech_probabilities: torch.Tensor
    should_transcribe: torch.Tensor
    active_indices: torch.Tensor
    whisper_encoder_states: torch.Tensor | None


class GatedWhisperAdapter(nn.Module):
    """Suppress no-speech rows before running the ASR adapter and decoder."""

    def __init__(
        self,
        adapter: ShortStateAdapter,
        speech_head: nn.Module,
        *,
        threshold: float = 0.5,
    ) -> None:
        super().__init__()
        if not 0.0 < threshold < 1.0:
            raise ValueError("threshold must be strictly between zero and one")
        self.adapter = adapter
        self.speech_head = speech_head
        self.threshold = threshold

    @classmethod
    def from_checkpoints(
        cls,
        *,
        adapter_checkpoint: pathlib.Path,
        speech_head_checkpoint: pathlib.Path,
        device: torch.device | str,
        threshold: float = 0.5,
    ) -> GatedWhisperAdapter:
        adapter = ShortStateAdapter()
        adapter.load_state_dict(
            torch.load(adapter_checkpoint, map_location="cpu", weights_only=True),
            strict=True,
        )
        speech_head = SpeechPresenceHead()
        speech_head.load_state_dict(
            torch.load(speech_head_checkpoint, map_location="cpu", weights_only=True),
            strict=True,
        )
        return cls(adapter, speech_head, threshold=threshold).to(device).eval()

    def forward(
        self,
        source: torch.Tensor,
        source_mask: torch.Tensor | None = None,
    ) -> GatedWhisperAdapterOutput:
        if source.ndim != 3:
            raise ValueError("source must have shape [batch, frames, hidden]")
        if source_mask is None:
            source_mask = torch.ones(
                source.shape[:2], device=source.device, dtype=torch.bool
            )
        elif source_mask.shape != source.shape[:2]:
            raise ValueError("source_mask must match source batch and frame axes")
        else:
            source_mask = source_mask.to(device=source.device, dtype=torch.bool)

        probabilities = self.speech_head(source, source_mask).sigmoid()
        should_transcribe = probabilities >= self.threshold
        active_indices = should_transcribe.nonzero(as_tuple=False).flatten()
        whisper_states: torch.Tensor | None = None
        if active_indices.numel():
            adapter_dtype = next(self.adapter.parameters()).dtype
            active_source = source.index_select(0, active_indices).to(adapter_dtype)
            active_mask = source_mask.index_select(0, active_indices)
            whisper_states = self.adapter(active_source, active_mask)
        return GatedWhisperAdapterOutput(
            speech_probabilities=probabilities,
            should_transcribe=should_transcribe,
            active_indices=active_indices,
            whisper_encoder_states=whisper_states,
        )
