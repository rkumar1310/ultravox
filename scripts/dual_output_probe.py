"""Decode an Ultravox response and a Whisper transcript from one audio encoding."""

from __future__ import annotations

import argparse
import gc
import json
import pathlib
import time
import wave

import numpy as np
import torch
import transformers
from transformers.modeling_outputs import BaseModelOutput

from ultravox.model.ultravox_config import UltravoxConfig
from ultravox.model.ultravox_model import UltravoxModel
from ultravox.model.ultravox_processing import UltravoxProcessor
from ultravox.inference.shared_whisper_adapter import GatedWhisperAdapter

DEFAULT_ULTRAVOX_MODEL = pathlib.Path(".model-cache/ultravox-v05")
DEFAULT_TEXT_MODEL = "unsloth/Llama-3.2-1B-Instruct"
DEFAULT_WHISPER_MODEL = "openai/whisper-large-v3-turbo"
DEFAULT_ADAPTER_CHECKPOINT = pathlib.Path(
    "checkpoints/shared-whisper-adapter/adapter-asr-robust-20000.pt"
)
DEFAULT_SPEECH_HEAD_CHECKPOINT = pathlib.Path(
    "checkpoints/shared-whisper-adapter/speech-head-best-16000.pt"
)
EXPECTED_TRANSCRIPT = (
    "My project codename is blue bicycle seven. Please schedule the launch "
    "for Tuesday at nine in the morning."
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=pathlib.Path)
    parser.add_argument(
        "--ultravox-model", type=pathlib.Path, default=DEFAULT_ULTRAVOX_MODEL
    )
    parser.add_argument("--text-model", default=DEFAULT_TEXT_MODEL)
    parser.add_argument("--whisper-model", default=DEFAULT_WHISPER_MODEL)
    parser.add_argument(
        "--adapter-checkpoint",
        type=pathlib.Path,
        default=DEFAULT_ADAPTER_CHECKPOINT,
    )
    parser.add_argument(
        "--speech-head-checkpoint",
        type=pathlib.Path,
        default=DEFAULT_SPEECH_HEAD_CHECKPOINT,
    )
    parser.add_argument("--speech-threshold", type=float, default=0.5)
    parser.add_argument("--max-response-tokens", type=int, default=24)
    parser.add_argument("--max-transcript-tokens", type=int, default=64)
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float16 if device.type == "mps" else torch.float32
    print(f"device={device} dtype={dtype}", flush=True)

    whisper_processor = transformers.AutoProcessor.from_pretrained(args.whisper_model)
    whisper_model = _load_whisper_decoder_model(args.whisper_model, device, dtype)
    transcription_adapter = GatedWhisperAdapter.from_checkpoints(
        adapter_checkpoint=args.adapter_checkpoint,
        speech_head_checkpoint=args.speech_head_checkpoint,
        device=device,
        threshold=args.speech_threshold,
    )
    model, tokenizer, processor = _load_ultravox(
        args.ultravox_model, args.text_model, device, dtype
    )

    audio = _read_pcm16_wav(args.audio)
    system_prompt = (
        "Act only as an intent classifier. If the recording asks to schedule "
        "something, reply with exactly SCHEDULE_EVENT; otherwise reply with exactly "
        "OTHER. Do not transcribe or answer the recording."
    )
    text = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "<|audio|>"},
        ],
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = processor(
        text=text,
        audio=audio,
        sampling_rate=16_000,
        return_tensors="pt",
    ).to(device)

    _sync(device)
    encode_started = time.perf_counter()
    with torch.inference_mode():
        shared_encoder_state = model.audio_tower(
            inputs["audio_values"].to(dtype),
            audio_len=inputs.get("audio_len"),
        ).last_hidden_state
        gated_transcription = transcription_adapter(shared_encoder_state)
    _sync(device)
    encoder_ms = (time.perf_counter() - encode_started) * 1_000

    response_inputs = _inject_shared_audio(
        model=model,
        input_ids=inputs["input_ids"],
        encoder_state=shared_encoder_state,
        audio_start=int(inputs["audio_token_start_idx"][0]),
        audio_length=int(inputs["audio_token_len"][0]),
    )

    _sync(device)
    decode_started = time.perf_counter()
    response_ids = _decode_llama(
        model.language_model,
        response_inputs,
        inputs["attention_mask"],
        stop_ids={
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        },
        max_new_tokens=args.max_response_tokens,
    )
    transcript_ids: list[int] = []
    if gated_transcription.whisper_encoder_states is not None:
        transcript_ids = _decode_whisper_official(
            model=whisper_model,
            encoder_state=gated_transcription.whisper_encoder_states,
            max_new_tokens=args.max_transcript_tokens,
        )
    _sync(device)
    decode_ms = (time.perf_counter() - decode_started) * 1_000

    response = tokenizer.decode(response_ids, skip_special_tokens=True).strip()
    transcript = (
        whisper_processor.batch_decode([transcript_ids], skip_special_tokens=True)[
            0
        ].strip()
        if transcript_ids
        else ""
    )
    speech_probability = float(gated_transcription.speech_probabilities[0].cpu())
    result = {
        "expectedTranscript": EXPECTED_TRANSCRIPT,
        "biasedPrompt": system_prompt,
        "llamaResponse": response,
        "whisperTranscript": transcript,
        "speechProbability": round(speech_probability, 6),
        "speechThreshold": args.speech_threshold,
        "transcriptionSuppressed": not bool(
            gated_transcription.should_transcribe[0].item()
        ),
        "adapterCheckpoint": str(args.adapter_checkpoint),
        "speechHeadCheckpoint": str(args.speech_head_checkpoint),
        "sharedEncoderPasses": 1,
        "encoderMs": round(encoder_ms, 1),
        "combinedDecodeMs": round(decode_ms, 1),
    }
    print(json.dumps(result, indent=2), flush=True)


def _load_ultravox(
    model_path: pathlib.Path,
    text_model: str,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[
    UltravoxModel,
    transformers.PreTrainedTokenizerBase,
    UltravoxProcessor,
]:
    raw_config = json.loads((model_path / "config.json").read_text())
    raw_config["text_model_id"] = text_model
    raw_config["torch_dtype"] = str(dtype).removeprefix("torch.")
    config = UltravoxConfig(**raw_config)
    print("loading Ultravox encoder, projector, and Llama...", flush=True)
    model = UltravoxModel.from_pretrained(
        model_path,
        config=config,
        torch_dtype=dtype,
    ).to(device)
    model.eval()

    audio_processor = transformers.AutoProcessor.from_pretrained(
        config.audio_config._name_or_path
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(text_model)
    tokenizer.pad_token = tokenizer.eos_token
    processor = UltravoxProcessor(
        audio_processor=audio_processor,
        tokenizer=tokenizer,
        stack_factor=config.stack_factor,
    )
    return model, tokenizer, processor


def _load_whisper_decoder_model(
    model_id: str, device: torch.device, dtype: torch.dtype
) -> transformers.WhisperForConditionalGeneration:
    print("loading Whisper's official decoder/generation implementation...", flush=True)
    model = transformers.WhisperForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.generation_config = transformers.GenerationConfig.from_pretrained(model_id)
    # The encoder output is supplied by Ultravox. Removing Whisper's own encoder
    # proves that the official generation path cannot silently encode a second time.
    model.model.encoder = _UnavailableWhisperEncoder().to(dtype=dtype)
    gc.collect()
    model.to(device).eval()
    return model


@torch.inference_mode()
def _inject_shared_audio(
    *,
    model: UltravoxModel,
    input_ids: torch.Tensor,
    encoder_state: torch.Tensor,
    audio_start: int,
    audio_length: int,
) -> torch.Tensor:
    embeddings = model.get_input_embeddings()(input_ids)
    audio_embeddings = model.multi_modal_projector(encoder_state.to(embeddings.dtype))
    audio_length = min(audio_length, audio_embeddings.shape[1])
    embeddings[:, audio_start : audio_start + audio_length] = audio_embeddings[
        :, :audio_length
    ]
    return embeddings


@torch.inference_mode()
def _decode_llama(
    model: torch.nn.Module,
    inputs_embeds: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    stop_ids: set[int],
    max_new_tokens: int,
) -> list[int]:
    output = model(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        use_cache=True,
        return_dict=True,
    )
    past_key_values = output.past_key_values
    next_token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
    generated: list[int] = []
    for _ in range(max_new_tokens):
        token_id = int(next_token.item())
        if token_id in stop_ids:
            break
        generated.append(token_id)
        attention_mask = torch.cat(
            [attention_mask, torch.ones_like(next_token, dtype=attention_mask.dtype)],
            dim=1,
        )
        output = model(
            input_ids=next_token,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        past_key_values = output.past_key_values
        next_token = output.logits[:, -1].argmax(dim=-1, keepdim=True)
    return generated


@torch.inference_mode()
def _decode_whisper_official(
    *,
    model: transformers.WhisperForConditionalGeneration,
    encoder_state: torch.Tensor,
    max_new_tokens: int,
) -> list[int]:
    generated = model.generate(
        encoder_outputs=BaseModelOutput(
            last_hidden_state=encoder_state.to(model.model.decoder.dtype)
        ),
        language="english",
        task="transcribe",
        return_timestamps=False,
        condition_on_prev_tokens=False,
        do_sample=False,
        max_new_tokens=max_new_tokens,
    )
    return generated[0].tolist()


class _UnavailableWhisperEncoder(torch.nn.Module):
    """Expose stride metadata while failing if generation tries to re-encode."""

    main_input_name = "input_features"

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv1d(1, 1, kernel_size=1, stride=1)
        self.conv2 = torch.nn.Conv1d(1, 1, kernel_size=1, stride=2)

    def forward(self, *args, **kwargs):
        raise RuntimeError("Whisper attempted a forbidden second encoder pass")


def _read_pcm16_wav(path: pathlib.Path) -> np.ndarray:
    with wave.open(str(path), "rb") as handle:
        if handle.getframerate() != 16_000:
            raise ValueError("WAV must be 16 kHz")
        if handle.getsampwidth() != 2:
            raise ValueError("WAV must contain signed 16-bit PCM")
        channels = handle.getnchannels()
        pcm = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2")
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1)
    return pcm.astype(np.float32) / 32768.0


def _sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


if __name__ == "__main__":
    main()
