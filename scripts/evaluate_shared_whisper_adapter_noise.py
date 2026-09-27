"""Measure adapter hallucinations on silence and noise-dominated speech."""

from __future__ import annotations

import argparse
import json
import math
import pathlib

import numpy as np
import torch
import transformers
from jiwer import wer
from transformers.modeling_outputs import BaseModelOutput

from scripts.dual_output_probe import _load_whisper_decoder_model
from scripts.dual_output_probe import _read_pcm16_wav
from scripts.train_shared_whisper_adapter import MODEL_ID
from scripts.train_shared_whisper_adapter import EXPECTED_HOLDOUT
from scripts.train_shared_whisper_adapter_local import _load_audio_encoder
from scripts.train_shared_whisper_adapter_librispeech import _normalize_for_wer
from ultravox.inference.shared_whisper_adapter import ShortStateAdapter

DEFAULT_AUDIO = pathlib.Path(".model-cache/dual-output.wav")
DEFAULT_CHECKPOINT = pathlib.Path(
    "checkpoints/shared-whisper-adapter/adapter-asr-robust-17000.pt"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=pathlib.Path, default=DEFAULT_AUDIO)
    parser.add_argument("--checkpoint", type=pathlib.Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--seed", type=int, default=731)
    args = parser.parse_args()

    if not args.audio.exists():
        parser.error(f"audio does not exist: {args.audio}")
    if not args.checkpoint.exists():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")

    if torch.backends.mps.is_available():
        device = torch.device("mps")
        encoder_dtype = torch.float16
        decoder_dtype = torch.float32
    else:
        device = torch.device("cpu")
        encoder_dtype = torch.float32
        decoder_dtype = torch.float32

    speech = _read_pcm16_wav(args.audio)
    cases = _build_cases(speech, seed=args.seed)
    processor = transformers.AutoProcessor.from_pretrained(MODEL_ID)
    encoder = _load_audio_encoder(device, encoder_dtype)
    adapter = ShortStateAdapter().to(device=device, dtype=torch.float32).eval()
    adapter.load_state_dict(
        torch.load(args.checkpoint, map_location=device, weights_only=True)
    )
    decoder = _load_whisper_decoder_model(MODEL_ID, device, decoder_dtype)

    audio_arrays = [case["audio"] for case in cases]
    features = processor(
        audio_arrays,
        sampling_rate=16_000,
        padding="longest",
        max_length=max(len(audio) for audio in audio_arrays),
        return_attention_mask=True,
        return_tensors="pt",
    )
    audio_lengths = features.attention_mask.sum(dim=-1).to(device=device)
    with torch.inference_mode():
        short = encoder(
            features.input_features.to(device=device, dtype=encoder_dtype),
            audio_len=audio_lengths,
        ).last_hidden_state
        encoded_lengths = encoder._get_feat_extract_output_lengths(audio_lengths)
        source_mask = (
            torch.arange(short.shape[1], device=device)[None, :]
            < encoded_lengths[:, None]
        )
        adapted = adapter(short.float(), source_mask).to(decoder_dtype)
        generated = decoder.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=adapted),
            language="english",
            task="transcribe",
            return_timestamps=False,
            condition_on_prev_tokens=False,
            do_sample=False,
            max_new_tokens=128,
        )
    transcripts = [
        text.strip()
        for text in processor.batch_decode(generated, skip_special_tokens=True)
    ]

    results: list[dict[str, object]] = []
    for case, transcript in zip(cases, transcripts, strict=True):
        reference = str(case["reference"])
        normalized_hypothesis = _normalize_for_wer(transcript)
        result = {
            "name": case["name"],
            "kind": case["kind"],
            "snrDb": case["snrDb"],
            "reference": reference,
            "transcript": transcript,
            "hallucinatedWords": (
                len(normalized_hypothesis.split()) if not reference else None
            ),
            "wer": (
                wer(
                    _normalize_for_wer(reference),
                    normalized_hypothesis,
                )
                if reference
                else None
            ),
        }
        results.append(result)

    noise_only = [result for result in results if result["kind"] == "noise-only"]
    heavy_speech = [
        result for result in results if result["kind"] == "noise-dominated-speech"
    ]
    report = {
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "mixDefinition": (
            "speech and noise are independently RMS-normalized, then mixed at "
            "0.20 speech + 0.80 noise (-12.04 dB SNR)"
        ),
        "noiseOnlyFalseTranscriptionRate": sum(
            bool(result["transcript"]) for result in noise_only
        )
        / len(noise_only),
        "noiseOnlyHallucinatedWords": sum(
            int(result["hallucinatedWords"]) for result in noise_only
        ),
        "heavyNoiseMeanWer": float(
            np.mean([float(result["wer"]) for result in heavy_speech])
        ),
        "results": results,
    }
    print(json.dumps(report, indent=2))


def _build_cases(speech: np.ndarray, *, seed: int) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    speech_rms = _rms(speech)
    noises = {
        "white": rng.normal(size=len(speech)).astype(np.float32),
        "pink": _pink_noise(len(speech), rng),
        "hum": _hum_noise(len(speech), rng),
    }
    cases: list[dict[str, object]] = [
        _case("clean-speech", "speech", speech, EXPECTED_HOLDOUT, None),
        _case(
            "digital-silence",
            "noise-only",
            np.zeros_like(speech),
            "",
            None,
        ),
    ]
    for name, noise in noises.items():
        unit_noise = noise / _rms(noise)
        cases.append(
            _case(
                f"quiet-{name}-noise-only",
                "noise-only",
                unit_noise * speech_rms * 0.01,
                "",
                None,
            )
        )
        cases.append(
            _case(
                f"speech-20-noise-80-{name}",
                "noise-dominated-speech",
                _safe_peak_normalize(0.20 * (speech / speech_rms) + 0.80 * unit_noise),
                EXPECTED_HOLDOUT,
                20.0 * math.log10(0.20 / 0.80),
            )
        )
    return cases


def _case(
    name: str,
    kind: str,
    audio: np.ndarray,
    reference: str,
    snr_db: float | None,
) -> dict[str, object]:
    return {
        "name": name,
        "kind": kind,
        "audio": np.asarray(audio, dtype=np.float32),
        "reference": reference,
        "snrDb": None if snr_db is None else round(snr_db, 2),
    }


def _pink_noise(length: int, rng: np.random.Generator) -> np.ndarray:
    frequencies = np.fft.rfftfreq(length)
    scale = np.ones_like(frequencies)
    scale[1:] = 1.0 / np.sqrt(frequencies[1:])
    spectrum = (
        rng.normal(size=len(frequencies)) + 1j * rng.normal(size=len(frequencies))
    ) * scale
    spectrum[0] = 0
    return np.fft.irfft(spectrum, n=length).astype(np.float32)


def _hum_noise(length: int, rng: np.random.Generator) -> np.ndarray:
    time = np.arange(length, dtype=np.float32) / 16_000
    hum = sum(
        (1.0 / harmonic) * np.sin(2 * np.pi * 60 * harmonic * time)
        for harmonic in range(1, 6)
    )
    hum += 0.12 * rng.normal(size=length)
    return np.asarray(hum, dtype=np.float32)


def _rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(samples), dtype=np.float64) + 1e-12))


def _safe_peak_normalize(samples: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(samples)))
    if peak <= 0.98:
        return samples.astype(np.float32)
    return (samples * (0.98 / peak)).astype(np.float32)


if __name__ == "__main__":
    main()
