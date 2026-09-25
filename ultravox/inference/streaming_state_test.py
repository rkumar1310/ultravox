from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
from transformers import WhisperConfig

from ultravox.inference.streaming_state import IncrementalWhisperEncoder
from ultravox.inference.streaming_state import StreamingClassifierHead
from ultravox.inference.streaming_state import StreamingUltravoxSession
from ultravox.inference.streaming_state import _install_block_causal_audio_mask
from ultravox.model.ultravox_model import ModifiedWhisperEncoder


class FakeCache:
    def __init__(self, length=0):
        self.length = length

    def get_seq_length(self):
        return self.length


class FakeDecoder(nn.Module):
    def forward(self, inputs_embeds, past_key_values=None, **_):
        previous = 0 if past_key_values is None else past_key_values.length
        return SimpleNamespace(
            last_hidden_state=inputs_embeds.cumsum(dim=1) + previous,
            past_key_values=FakeCache(previous + inputs_embeds.shape[1]),
        )


class FakeAudioTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(max_source_positions=32)
        self.audio_streaming_mask = None

    @property
    def dtype(self):
        return self.anchor.dtype

    def forward(self, audio_values, audio_len=None):
        del audio_len
        # Mimic Whisper's stride-2 encoder output.
        frames = audio_values[:, :2, ::2].transpose(1, 2)
        return SimpleNamespace(last_hidden_state=frames)


class FakeProjector(nn.Module):
    def forward(self, encoder_output):
        batch, frames, channels = encoder_output.shape
        assert frames % 2 == 0
        return encoder_output.reshape(batch, frames // 2, channels * 2)


class FakeIncrementalAudioEncoder:
    def __init__(self):
        self.processed = 0

    def push(self, full_audio_values, *, audio_len, final):
        del audio_len
        frames = full_audio_values[:, :2, ::2].transpose(1, 2)
        stable = frames.shape[1] if final else ((frames.shape[1] - 1) // 2) * 2
        if stable <= self.processed:
            return None
        output = frames[:, self.processed : stable]
        self.processed = stable
        return output


class FakeUltravox(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(stack_factor=2)
        self.audio_tower = FakeAudioTower()
        self.multi_modal_projector = FakeProjector()
        self.embeddings = nn.Embedding(32, 4)
        self.output = nn.Linear(4, 32, bias=False)
        self.decoder = FakeDecoder()

    def get_input_embeddings(self):
        return self.embeddings

    def get_output_embeddings(self):
        return self.output

    def get_decoder(self):
        return self.decoder


def test_text_audio_text_share_one_append_only_cache():
    model = FakeUltravox()
    classifier = StreamingClassifierHead(hidden_size=4, num_labels=3)
    session = StreamingUltravoxSession(
        model,
        classifier=classifier,
        encoder_stride=2,
        audio_block_encoder_frames=2,
        incremental_audio_encoder=FakeIncrementalAudioEncoder(),
    )

    prefix = session.push_text([1, 2, 3])
    assert prefix.cache_tokens == 3

    # 5 feature frames -> (5 - 1) / 2 = one stable 2-frame encoder block.
    audio = torch.arange(10, dtype=torch.float32).reshape(1, 2, 5)
    first_audio = session.push_audio_features(audio)
    assert first_audio is not None
    assert first_audio.appended_tokens == 1
    assert first_audio.cache_tokens == 4
    assert first_audio.classifier_logits.shape == (1, 3)

    # 9 feature frames -> two stable blocks total, so only one more is appended.
    audio = torch.arange(18, dtype=torch.float32).reshape(1, 2, 9)
    second_audio = session.push_audio_features(audio)
    assert second_audio is not None
    assert second_audio.appended_tokens == 1
    assert second_audio.cache_tokens == 5

    suffix = session.push_text([4, 5])
    assert suffix.cache_tokens == 7
    assert session.text_tokens == 5
    assert session.committed_audio_tokens == 2


def test_incomplete_audio_block_does_not_mutate_llama_state():
    session = StreamingUltravoxSession(
        FakeUltravox(),
        encoder_stride=2,
        audio_block_encoder_frames=2,
        incremental_audio_encoder=FakeIncrementalAudioEncoder(),
    )
    session.push_text([1])
    audio = torch.ones((1, 2, 4))
    assert session.push_audio_features(audio) is None
    assert session.cache_tokens == 1


def test_final_audio_flushes_partial_group_and_allows_text_suffix():
    session = StreamingUltravoxSession(
        FakeUltravox(),
        encoder_stride=2,
        audio_block_encoder_frames=2,
        incremental_audio_encoder=FakeIncrementalAudioEncoder(),
    )
    # Fake projector requires complete groups, so use a complete encoder group.
    audio = torch.ones((1, 2, 4))
    step = session.push_audio_features(audio, final=True)
    assert step is not None
    assert step.appended_tokens == 1

    suffix = session.push_text([1])
    assert suffix.cache_tokens == 2

    try:
        session.push_audio_features(audio)
    except RuntimeError as exc:
        assert "finalized" in str(exc)
    else:
        raise AssertionError("finalized stream accepted more audio")


def test_label_scores_only_materialize_requested_labels():
    session = StreamingUltravoxSession(
        FakeUltravox(),
        audio_block_encoder_frames=2,
        incremental_audio_encoder=FakeIncrementalAudioEncoder(),
    )
    session.push_text([1, 2])
    scores = session.score_label_tokens({"continue": 7, "decide": 8})
    assert set(scores) == {"continue", "decide"}
    assert all(isinstance(value, float) for value in scores.values())


def test_pcm_is_accumulated_before_feature_extraction():
    class FakeProcessor:
        def __init__(self):
            self.lengths = []

        def __call__(self, *, audio, **_):
            self.lengths.append(len(audio))
            frames = max(1, len(audio) // 160)
            return {
                "audio_values": torch.ones((1, 2, frames)),
                "audio_len": torch.tensor([frames]),
            }

    processor = FakeProcessor()
    session = StreamingUltravoxSession(
        FakeUltravox(),
        encoder_stride=2,
        audio_block_encoder_frames=2,
        incremental_audio_encoder=FakeIncrementalAudioEncoder(),
    )
    session.push_audio_waveform(np.zeros(800, dtype=np.float32), processor)
    session.push_audio_waveform(np.zeros(800, dtype=np.float32), processor)
    assert processor.lengths == [800, 1600]


def test_incremental_whisper_matches_full_block_causal_encoder():
    torch.manual_seed(0)
    config = WhisperConfig(
        num_mel_bins=4,
        d_model=8,
        encoder_layers=2,
        encoder_attention_heads=2,
        encoder_ffn_dim=16,
        max_source_positions=16,
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
    )
    tower = ModifiedWhisperEncoder(config).eval()
    incremental = IncrementalWhisperEncoder(tower, block_encoder_frames=2)
    audio = torch.randn(1, 4, 9)

    first = incremental.push(audio[:, :, :5], audio_len=torch.tensor([5]), final=False)
    second = incremental.push(audio, audio_len=torch.tensor([9]), final=True)
    streamed = torch.cat((first, second), dim=1)

    _install_block_causal_audio_mask(tower, block_encoder_frames=2)
    full = tower(audio, audio_len=torch.tensor([9])).last_hidden_state
    torch.testing.assert_close(streamed, full, atol=1e-6, rtol=1e-6)
