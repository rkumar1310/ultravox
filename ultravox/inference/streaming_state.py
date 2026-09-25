"""Experimental incremental state for Ultravox v0.5.

This module deliberately separates two ideas that are often both called
"streaming":

* the audio encoder produces stable, causal blocks; and
* the projected audio embeddings are appended to the language-model KV cache.

The released v0.5 checkpoint was not trained with its causal audio mask enabled,
so this is an architectural prototype, not a claim of preserved model quality.
The implementation caches Whisper attention keys/values per layer and appends
only newly stable embeddings to Llama. The small convolutional frontend is
replayed over the observed feature prefix so block-edge values remain exact.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Dict, Mapping, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache
from transformers.cache_utils import EncoderDecoderCache

Cache = object


@dataclasses.dataclass(frozen=True)
class StreamingStep:
    """Observable result after appending one modality to the session."""

    modality: str
    appended_tokens: int
    cache_tokens: int
    audio_seconds_seen: float
    hidden_state: torch.Tensor
    classifier_logits: Optional[torch.Tensor] = None
    encoder_frames: int = 0


class StreamingClassifierHead(nn.Module):
    """A small trainable classifier over Ultravox's latest Llama hidden state.

    This head is intentionally independent of the language-model vocabulary
    projection.  It makes an intermediate decision cheap, but its weights must
    be trained on the desired labels before its output is meaningful.
    """

    def __init__(self, hidden_size: int, num_labels: int):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.projection = nn.Linear(hidden_size, num_labels)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return self.projection(self.norm(hidden_state))


class IncrementalWhisperEncoder:
    """Run only new causal Whisper blocks while caching attention keys/values."""

    def __init__(
        self,
        audio_tower: nn.Module,
        *,
        block_encoder_frames: int,
        encoder_stride: int = 2,
    ):
        self.audio_tower = audio_tower
        self.block_encoder_frames = block_encoder_frames
        self.encoder_stride = encoder_stride
        self.processed_encoder_frames = 0
        self.cache = EncoderDecoderCache(DynamicCache(), DynamicCache())
        for layer_index, layer in enumerate(audio_tower.layers):
            # Whisper's encoder normally has no cache, so Transformers leaves
            # these unset. DynamicCache needs a stable index per encoder layer.
            layer.self_attn.layer_idx = layer_index

    @torch.inference_mode()
    def push(
        self,
        full_audio_values: torch.Tensor,
        *,
        audio_len: torch.Tensor,
        final: bool,
    ) -> Optional[torch.Tensor]:
        """Return newly completed encoder frames, or ``None`` if none are stable."""

        tower = self.audio_tower
        hidden_states = F.gelu(tower.conv1(full_audio_values))
        hidden_states = F.gelu(tower.conv2(hidden_states)).permute(0, 2, 1)
        encoder_frames = int(hidden_states.shape[1])

        if final:
            stable_frames = encoder_frames
        else:
            # The two kernel-3 convolutions need one mel frame of look-ahead for
            # the final encoder frame in a block to stop changing.
            stable_frames = max(0, (int(audio_len.item()) - 1) // self.encoder_stride)
            stable_frames = min(encoder_frames, stable_frames)
            stable_frames = (
                stable_frames // self.block_encoder_frames
            ) * self.block_encoder_frames

        if stable_frames <= self.processed_encoder_frames:
            return None

        emitted = []
        while self.processed_encoder_frames < stable_frames:
            start = self.processed_encoder_frames
            end = min(start + self.block_encoder_frames, stable_frames)
            block = hidden_states[:, start:end]
            block = block + tower.embed_positions.weight[start:end]
            block = F.dropout(block, p=tower.dropout, training=tower.training)

            for layer in tower.layers:
                residual = block
                normalized = layer.self_attn_layer_norm(block)
                attended, _, _ = layer.self_attn(
                    hidden_states=normalized,
                    past_key_value=self.cache,
                    attention_mask=None,
                    layer_head_mask=None,
                    output_attentions=False,
                )
                attended = F.dropout(attended, p=layer.dropout, training=layer.training)
                block = residual + attended

                residual = block
                block = layer.final_layer_norm(block)
                block = layer.activation_fn(layer.fc1(block))
                block = F.dropout(
                    block, p=layer.activation_dropout, training=layer.training
                )
                block = layer.fc2(block)
                block = F.dropout(block, p=layer.dropout, training=layer.training)
                block = residual + block

                if block.dtype == torch.float16 and (
                    torch.isinf(block).any() or torch.isnan(block).any()
                ):
                    clamp = torch.finfo(block.dtype).max - 1000
                    block = torch.clamp(block, min=-clamp, max=clamp)

            emitted.append(tower.layer_norm(block))
            self.processed_encoder_frames = end

        return torch.cat(emitted, dim=1)


class StreamingUltravoxSession:
    """Maintain one append-only multimodal Llama state.

    Text tokens and stable projected audio embeddings are appended in arrival
    order.  Model weights stay resident and unchanged; ``past_key_values`` is
    the per-session state that grows.

    Audio correctness requires a causal audio encoder. Ultravox v0.5 ships with
    causal masking disabled, so this class executes the encoder block by block
    and caches each layer's attention keys/values. A block is aligned with the
    projector stack, producing one Llama audio token per 160 ms block for the
    released Whisper configuration.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        classifier: Optional[nn.Module] = None,
        sampling_rate: int = 16_000,
        mel_hop_samples: int = 160,
        encoder_stride: int = 2,
        audio_block_encoder_frames: Optional[int] = None,
        incremental_audio_encoder: Optional[IncrementalWhisperEncoder] = None,
    ):
        self.model = model.eval()
        self.classifier = classifier.eval() if classifier is not None else None
        self.sampling_rate = sampling_rate
        self.mel_hop_samples = mel_hop_samples
        self.encoder_stride = encoder_stride

        self.stack_factor = int(model.config.stack_factor)
        self.audio_block_encoder_frames = int(
            audio_block_encoder_frames or self.stack_factor
        )
        if self.audio_block_encoder_frames % self.stack_factor != 0:
            raise ValueError(
                "audio_block_encoder_frames must be a multiple of the projector "
                "stack factor"
            )

        self.audio_tower = _unwrap_module(model.audio_tower)
        self.incremental_audio_encoder = incremental_audio_encoder or (
            IncrementalWhisperEncoder(
                self.audio_tower,
                block_encoder_frames=self.audio_block_encoder_frames,
                encoder_stride=self.encoder_stride,
            )
        )

        self.decoder = model.get_decoder()
        self.device = next(model.parameters()).device
        self.embedding_dtype = model.get_input_embeddings().weight.dtype

        self.past_key_values: Optional[Cache] = None
        self.last_hidden_state: Optional[torch.Tensor] = None
        self._waveform = np.empty((0,), dtype=np.float32)
        self._committed_audio_tokens = 0
        self._text_tokens = 0
        self._audio_finalized = False

    @property
    def cache_tokens(self) -> int:
        return _cache_length(self.past_key_values)

    @property
    def committed_audio_tokens(self) -> int:
        return self._committed_audio_tokens

    @property
    def text_tokens(self) -> int:
        return self._text_tokens

    @property
    def audio_seconds_seen(self) -> float:
        return len(self._waveform) / self.sampling_rate

    @torch.inference_mode()
    def push_text(self, input_ids: Union[torch.Tensor, Sequence[int]]) -> StreamingStep:
        """Append already-tokenized text to the same Llama KV cache."""

        ids = torch.as_tensor(input_ids, dtype=torch.long, device=self.device)
        if ids.ndim == 1:
            ids = ids.unsqueeze(0)
        if ids.ndim != 2 or ids.shape[0] != 1:
            raise ValueError("streaming prototype currently supports batch size 1")
        if ids.shape[1] == 0:
            raise ValueError("push_text requires at least one token")

        embeddings = self.model.get_input_embeddings()(ids)
        self._text_tokens += ids.shape[1]
        return self._append_embeddings("text", embeddings)

    @torch.inference_mode()
    def push_audio_waveform(
        self,
        pcm: Union[np.ndarray, torch.Tensor, Sequence[float]],
        processor,
        *,
        final: bool = False,
    ) -> Optional[StreamingStep]:
        """Append PCM and commit any newly stable audio blocks.

        The feature extractor and small convolutional frontend are replayed over
        the accumulated waveform, while Whisper transformer blocks and Llama are
        incremental. Only audio blocks that cannot change on the next call enter
        the Llama cache.
        """

        self._ensure_audio_open()
        chunk = _to_mono_float32(pcm)
        if len(chunk):
            self._waveform = np.concatenate((self._waveform, chunk))
        if len(self._waveform) == 0:
            return None

        processed = processor(
            audio=self._waveform,
            sampling_rate=self.sampling_rate,
            return_tensors="pt",
        )
        return self.push_audio_features(
            processed["audio_values"],
            audio_len=processed.get("audio_len"),
            final=final,
        )

    @torch.inference_mode()
    def push_audio_features(
        self,
        full_audio_values: torch.Tensor,
        *,
        audio_len: Optional[torch.Tensor] = None,
        final: bool = False,
    ) -> Optional[StreamingStep]:
        """Append stable embeddings from a full prefix of audio features.

        ``full_audio_values`` is the complete feature prefix observed so far,
        not merely the newest feature chunk.  The explicit contract prevents
        accidental drift at feature-extractor and convolution boundaries.
        """

        self._ensure_audio_open()
        audio_values = full_audio_values.to(
            device=self.device, dtype=self.audio_tower.dtype
        )
        if audio_values.ndim != 3 or audio_values.shape[0] != 1:
            raise ValueError("full_audio_values must have shape [1, mel_bins, frames]")

        feature_frames = int(audio_values.shape[-1])
        if audio_len is None:
            audio_len = torch.tensor([feature_frames], device=self.device)
        else:
            audio_len = torch.as_tensor(audio_len, device=self.device).reshape(1)

        new_encoder_output = self.incremental_audio_encoder.push(
            audio_values, audio_len=audio_len, final=final
        )
        if new_encoder_output is None:
            if final:
                self._audio_finalized = True
            return None

        new_audio_embeds = self.model.multi_modal_projector(
            new_encoder_output.to(self.embedding_dtype)
        )
        # Ultravox v0.5's StackAudioFrames intentionally appends one padding
        # group because the normal model path later slices by audio_token_len.
        # The streaming path bypasses that merge, so remove the sentinel here.
        expected_audio_tokens = math.ceil(
            new_encoder_output.shape[1] / self.stack_factor
        )
        new_audio_embeds = new_audio_embeds[:, :expected_audio_tokens]
        if new_audio_embeds.shape[1] == 0:
            if final:
                self._audio_finalized = True
            return None

        self._committed_audio_tokens += int(new_audio_embeds.shape[1])
        step = dataclasses.replace(
            self._append_embeddings("audio", new_audio_embeds),
            encoder_frames=int(new_encoder_output.shape[1]),
        )
        if final:
            self._audio_finalized = True
        return step

    @torch.inference_mode()
    def classify(self) -> torch.Tensor:
        """Run the small classifier without invoking the LM vocabulary head."""

        if self.classifier is None:
            raise RuntimeError("no classifier was supplied")
        if self.last_hidden_state is None:
            raise RuntimeError("the session has no state to classify")
        return self.classifier(self.last_hidden_state)

    @torch.inference_mode()
    def score_label_tokens(
        self, label_token_ids: Mapping[str, int]
    ) -> Dict[str, float]:
        """Score single-token labels without materializing full-vocabulary logits.

        This is useful for a zero-training smoke test.  A trained classifier head
        is preferable for production because label words can be tokenization- and
        prompt-sensitive.
        """

        if self.last_hidden_state is None:
            raise RuntimeError("the session has no state to score")
        if not label_token_ids:
            raise ValueError("at least one label token is required")

        output = self.model.get_output_embeddings()
        ids = torch.tensor(
            list(label_token_ids.values()), dtype=torch.long, device=self.device
        )
        selected_weight = output.weight.index_select(0, ids)
        logits = self.last_hidden_state.to(selected_weight.dtype) @ selected_weight.T
        if getattr(output, "bias", None) is not None:
            logits = logits + output.bias.index_select(0, ids)
        values = logits.squeeze(0).float().cpu().tolist()
        return dict(zip(label_token_ids.keys(), values))

    def _append_embeddings(
        self, modality: str, embeddings: torch.Tensor
    ) -> StreamingStep:
        output = self.decoder(
            inputs_embeds=embeddings,
            past_key_values=self.past_key_values,
            use_cache=True,
            return_dict=True,
        )
        self.past_key_values = output.past_key_values
        self.last_hidden_state = output.last_hidden_state[:, -1, :]
        classifier_logits = (
            self.classifier(self.last_hidden_state)
            if self.classifier is not None
            else None
        )
        return StreamingStep(
            modality=modality,
            appended_tokens=int(embeddings.shape[1]),
            cache_tokens=self.cache_tokens,
            audio_seconds_seen=self.audio_seconds_seen,
            hidden_state=self.last_hidden_state,
            classifier_logits=classifier_logits,
        )

    def _ensure_audio_open(self) -> None:
        if self._audio_finalized:
            raise RuntimeError("audio stream was finalized")


def _install_block_causal_audio_mask(
    audio_tower: nn.Module, block_encoder_frames: int
) -> None:
    """Install a causal block mask in encoder-frame units.

    Ultravox v0.5's helper allocates in pre-convolution frame units even though
    the mask is consumed after Whisper's stride-2 convolution.  Building the
    mask directly at ``max_source_positions`` makes the latency and alignment
    explicit and permits a block equal to the projector's stack factor.
    """

    if block_encoder_frames <= 0:
        raise ValueError("block_encoder_frames must be positive")
    max_encoder_frames = int(audio_tower.config.max_source_positions)
    blocks = math.ceil(max_encoder_frames / block_encoder_frames)
    device = next(audio_tower.parameters()).device
    causal_blocks = torch.tril(
        torch.ones(blocks, blocks, dtype=torch.bool, device=device), diagonal=0
    )
    allowed = causal_blocks.repeat_interleave(
        block_encoder_frames, dim=0
    ).repeat_interleave(block_encoder_frames, dim=1)
    allowed = allowed[:max_encoder_frames, :max_encoder_frames]
    dtype = audio_tower.dtype
    mask = torch.zeros_like(allowed, dtype=dtype)
    mask.masked_fill_(~allowed, torch.finfo(dtype).min)
    mask = mask[None, None]

    if "audio_streaming_mask" in audio_tower._buffers:
        audio_tower._buffers["audio_streaming_mask"] = mask
    else:
        if hasattr(audio_tower, "audio_streaming_mask"):
            delattr(audio_tower, "audio_streaming_mask")
        audio_tower.register_buffer("audio_streaming_mask", mask, persistent=False)


def _unwrap_module(module: nn.Module) -> nn.Module:
    get_base_model = getattr(module, "get_base_model", None)
    return get_base_model() if callable(get_base_model) else module


def _cache_length(cache: Optional[Cache]) -> int:
    if cache is None:
        return 0
    get_seq_length = getattr(cache, "get_seq_length", None)
    if callable(get_seq_length):
        return int(get_seq_length())
    if isinstance(cache, tuple) and cache:
        return int(cache[0][0].shape[-2])
    raise TypeError(f"unsupported cache type: {type(cache)!r}")


def _to_mono_float32(
    pcm: Union[np.ndarray, torch.Tensor, Sequence[float]],
) -> np.ndarray:
    if isinstance(pcm, torch.Tensor):
        pcm = pcm.detach().cpu().numpy()
    value = np.asarray(pcm)
    if value.ndim == 2:
        value = value.mean(axis=0)
    if value.ndim != 1:
        raise ValueError("PCM must be mono [samples] or channel-first [C, samples]")
    if np.issubdtype(value.dtype, np.integer):
        scale = max(abs(np.iinfo(value.dtype).min), np.iinfo(value.dtype).max)
        value = value.astype(np.float32) / float(scale)
    return value.astype(np.float32, copy=False)
