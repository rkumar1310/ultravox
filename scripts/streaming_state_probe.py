"""Run the Ultravox v0.5 streaming-state prototype on local PCM WAV audio."""

from __future__ import annotations

import argparse
import json
import pathlib
import time
import wave

import numpy as np
import torch
import transformers
from huggingface_hub import hf_hub_download

from ultravox.inference.streaming_state import StreamingClassifierHead
from ultravox.inference.streaming_state import StreamingUltravoxSession
from ultravox.model.ultravox_config import UltravoxConfig
from ultravox.model.ultravox_model import UltravoxModel
from ultravox.model.ultravox_processing import UltravoxProcessor

DEFAULT_MODEL = "fixie-ai/ultravox-v0_5-llama-3_2-1b"
DEFAULT_TEXT_MODEL = "unsloth/Llama-3.2-1B-Instruct"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=pathlib.Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--text-model",
        default=DEFAULT_TEXT_MODEL,
        help="Public mirror used when Meta's gated repository is unavailable.",
    )
    parser.add_argument("--chunk-ms", type=int, default=160)
    parser.add_argument("--device", choices=("cpu", "mps"), default="mps")
    parser.add_argument(
        "--suffix-text",
        default=" Context update: this is a priority customer.",
        help="Text appended to the same KV cache after the audio stream.",
    )
    args = parser.parse_args()

    if args.chunk_ms % 160:
        raise SystemExit("--chunk-ms must be a multiple of 160")

    device = torch.device(
        "mps" if args.device == "mps" and torch.backends.mps.is_available() else "cpu"
    )
    dtype = torch.float16 if device.type == "mps" else torch.float32

    local_model = pathlib.Path(args.model)
    config_path = (
        str(local_model / "config.json")
        if local_model.is_dir()
        else hf_hub_download(args.model, "config.json")
    )
    raw_config = json.loads(pathlib.Path(config_path).read_text())
    raw_config["text_model_id"] = args.text_model
    raw_config["torch_dtype"] = str(dtype).removeprefix("torch.")
    config = UltravoxConfig(**raw_config)

    started = time.perf_counter()
    model = UltravoxModel.from_pretrained(
        args.model,
        config=config,
        torch_dtype=dtype,
    ).to(device)
    model.eval()

    audio_processor = transformers.AutoProcessor.from_pretrained(
        config.audio_config._name_or_path
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.text_model)
    tokenizer.pad_token = tokenizer.eos_token
    processor = UltravoxProcessor(
        audio_processor=audio_processor,
        tokenizer=tokenizer,
        stack_factor=config.stack_factor,
    )
    print(f"model_ready_s={time.perf_counter() - started:.3f} device={device}")

    classifier = StreamingClassifierHead(
        config.text_config.hidden_size, num_labels=2
    ).to(device=device, dtype=dtype)
    session = StreamingUltravoxSession(model, classifier=classifier)

    instruction = (
        "You are listening to live speech. Decide whether enough information has "
        "arrived to act. Think of the next token as yes or no. Speech follows: "
    )
    prefix_ids = tokenizer(instruction, add_special_tokens=True).input_ids
    prefix = session.push_text(prefix_ids)
    print(f"text appended={prefix.appended_tokens} cache={prefix.cache_tokens}")

    pcm = read_pcm16_wav(args.audio)
    chunk_samples = 16_000 * args.chunk_ms // 1_000
    label_ids = {
        label: _single_token_id(tokenizer, " " + label) for label in ("no", "yes")
    }
    for offset in range(0, len(pcm), chunk_samples):
        final = offset + chunk_samples >= len(pcm)
        tick = time.perf_counter()
        step = session.push_audio_waveform(
            pcm[offset : offset + chunk_samples], processor, final=final
        )
        elapsed_ms = (time.perf_counter() - tick) * 1_000
        if step is None:
            print(
                f"audio_seen_s={session.audio_seconds_seen:.3f} stable_tokens=0 "
                f"elapsed_ms={elapsed_ms:.1f}"
            )
            continue
        scores = session.score_label_tokens(label_ids)
        hidden_norm = step.hidden_state.float().norm().item()
        print(
            f"audio_seen_s={step.audio_seconds_seen:.3f} "
            f"encoder_frames={step.encoder_frames} appended={step.appended_tokens} "
            f"cache={step.cache_tokens} "
            f"hidden_norm={hidden_norm:.3f} label_scores={scores} "
            f"elapsed_ms={elapsed_ms:.1f}"
        )

    suffix_ids = tokenizer(args.suffix_text, add_special_tokens=False).input_ids
    suffix = session.push_text(suffix_ids)
    print(
        f"text_after_audio appended={suffix.appended_tokens} "
        f"cache={suffix.cache_tokens} hidden_norm="
        f"{suffix.hidden_state.float().norm().item():.3f}"
    )


def read_pcm16_wav(path: pathlib.Path) -> np.ndarray:
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


def _single_token_id(tokenizer, text: str) -> int:
    ids = tokenizer(text, add_special_tokens=False).input_ids
    if len(ids) != 1:
        raise ValueError(f"label {text!r} is not a single token: {ids}")
    return ids[0]


if __name__ == "__main__":
    main()
