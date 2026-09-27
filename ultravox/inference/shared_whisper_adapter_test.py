from __future__ import annotations

import torch

from ultravox.inference.shared_whisper_adapter import GatedWhisperAdapter
from ultravox.inference.shared_whisper_adapter import ShortStateAdapter


class _FixedHead(torch.nn.Module):
    def __init__(self, logits: list[float]) -> None:
        super().__init__()
        self.register_buffer("logits", torch.tensor(logits))

    def forward(self, source: torch.Tensor, source_mask: torch.Tensor) -> torch.Tensor:
        del source_mask
        return self.logits[: source.shape[0]]


def test_gate_only_adapts_rows_classified_as_speech() -> None:
    adapter = ShortStateAdapter(hidden_size=8, adapter_size=4, frames=3)
    pipeline = GatedWhisperAdapter(
        adapter,
        _FixedHead([-10.0, 10.0]),
        threshold=0.5,
    )
    source = torch.randn(2, 4, 8)
    mask = torch.ones(2, 4, dtype=torch.bool)

    output = pipeline(source, mask)

    assert output.should_transcribe.tolist() == [False, True]
    assert output.active_indices.tolist() == [1]
    assert output.whisper_encoder_states is not None
    assert output.whisper_encoder_states.shape == (1, 3, 8)


def test_gate_skips_adapter_when_every_row_is_no_speech() -> None:
    pipeline = GatedWhisperAdapter(
        ShortStateAdapter(hidden_size=8, adapter_size=4, frames=3),
        _FixedHead([-10.0, -10.0]),
        threshold=0.5,
    )

    output = pipeline(torch.randn(2, 4, 8))

    assert output.should_transcribe.tolist() == [False, False]
    assert output.active_indices.numel() == 0
    assert output.whisper_encoder_states is None
