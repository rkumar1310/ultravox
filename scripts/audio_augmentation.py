"""Deterministic mild noise, room, and competing-speech augmentation."""

from __future__ import annotations

import functools
import hashlib
import pathlib
from dataclasses import dataclass

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve


PROFILE_NAME = "robust-v1"


@dataclass(frozen=True)
class AugmentationResult:
    samples: np.ndarray
    used_noise: bool
    used_room: bool
    used_overlap: bool


class RobustAudioAugmenter:
    """Adds reproducible, intentionally mild corruption to primary speech.

    The transcript remains the primary speaker's transcript.  A bounded
    reservoir of earlier utterances supplies occasional competing speech.
    """

    def __init__(
        self,
        assets_dir: pathlib.Path,
        *,
        seed: int,
        sample_rate: int = 16_000,
        reservoir_size: int = 64,
    ) -> None:
        self.assets_dir = assets_dir
        self.seed = seed
        self.sample_rate = sample_rate
        self.reservoir_size = reservoir_size
        self._reservoir: list[tuple[str, np.ndarray]] = []
        wav_paths = sorted(assets_dir.rglob("*.wav"))
        self._noise_paths = [path for path in wav_paths if _is_noise_path(path)]
        self._rir_paths = [path for path in wav_paths if _is_rir_path(path)]
        if not self._noise_paths:
            raise FileNotFoundError(f"no noise WAV files found below {assets_dir}")
        if not self._rir_paths:
            raise FileNotFoundError(f"no room impulse responses found below {assets_dir}")
        self.stats = {
            "samples": 0,
            "noise": 0,
            "room": 0,
            "overlap": 0,
        }

    @property
    def asset_counts(self) -> dict[str, int]:
        return {
            "noiseFiles": len(self._noise_paths),
            "roomImpulseFiles": len(self._rir_paths),
        }

    def augment(self, sample_id: str, samples: np.ndarray) -> AugmentationResult:
        rng = np.random.default_rng(_stable_seed(self.seed, sample_id))
        output = np.asarray(samples, dtype=np.float32).copy()
        used_room = bool(rng.random() < 0.40)
        used_noise = bool(rng.random() < 0.65)
        used_overlap = bool(self._reservoir and rng.random() < 0.35)

        if used_room:
            rir_path = self._rir_paths[int(rng.integers(len(self._rir_paths)))]
            output = _apply_room(output, _read_mono(rir_path, self.sample_rate), rng)
        if used_noise:
            noise_path = self._noise_paths[int(rng.integers(len(self._noise_paths)))]
            output = _add_noise(
                output,
                _read_mono(noise_path, self.sample_rate),
                snr_db=float(rng.uniform(14.0, 30.0)),
                rng=rng,
            )
        if used_overlap:
            _, competing = self._reservoir[int(rng.integers(len(self._reservoir)))]
            output = _add_competing_speech(
                output,
                competing,
                sir_db=float(rng.uniform(12.0, 24.0)),
                rng=rng,
            )

        peak = float(np.max(np.abs(output))) if output.size else 0.0
        if peak > 0.98:
            output *= 0.98 / peak
        self.stats["samples"] += 1
        self.stats["noise"] += int(used_noise)
        self.stats["room"] += int(used_room)
        self.stats["overlap"] += int(used_overlap)
        return AugmentationResult(
            samples=output.astype(np.float32, copy=False),
            used_noise=used_noise,
            used_room=used_room,
            used_overlap=used_overlap,
        )

    def observe(self, sample_id: str, samples: np.ndarray) -> None:
        self._reservoir.append(
            (sample_id, np.asarray(samples, dtype=np.float32).copy())
        )
        if len(self._reservoir) > self.reservoir_size:
            del self._reservoir[0]


def _stable_seed(seed: int, sample_id: str) -> int:
    digest = hashlib.sha256(f"{seed}:{sample_id}".encode()).digest()
    return int.from_bytes(digest[:8], "little")


def _is_noise_path(path: pathlib.Path) -> bool:
    relative = path.as_posix().lower()
    return "/noises/" in relative or "pointsource_noises" in relative


def _is_rir_path(path: pathlib.Path) -> bool:
    relative = path.as_posix().lower()
    return "rir" in relative and not _is_noise_path(path)


@functools.lru_cache(maxsize=24)
def _read_mono(path: pathlib.Path, expected_sample_rate: int) -> np.ndarray:
    samples, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    if sample_rate != expected_sample_rate:
        raise ValueError(
            f"expected {expected_sample_rate} Hz augmentation audio, "
            f"got {sample_rate} Hz: {path}"
        )
    return np.asarray(samples, dtype=np.float32)


def _rms(samples: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(samples), dtype=np.float64) + 1e-12))


def _apply_room(
    speech: np.ndarray, rir: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    if not speech.size or not rir.size:
        return speech
    rir = rir[: min(len(rir), 32_000)]
    rir_peak = float(np.max(np.abs(rir)))
    if rir_peak <= 1e-8:
        return speech
    rir = rir / rir_peak
    wet = fftconvolve(speech, rir, mode="full")[: len(speech)].astype(np.float32)
    wet_rms = _rms(wet)
    dry_rms = _rms(speech)
    if wet_rms > 1e-8:
        wet *= dry_rms / wet_rms
    wet_mix = float(rng.uniform(0.12, 0.32))
    return ((1.0 - wet_mix) * speech + wet_mix * wet).astype(np.float32)


def _add_noise(
    speech: np.ndarray,
    noise: np.ndarray,
    *,
    snr_db: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if not speech.size or not noise.size:
        return speech
    if len(noise) < len(speech):
        repeats = int(np.ceil(len(speech) / len(noise)))
        noise = np.tile(noise, repeats)
    start = int(rng.integers(0, len(noise) - len(speech) + 1))
    segment = noise[start : start + len(speech)].copy()
    segment -= float(segment.mean())
    noise_rms = _rms(segment)
    if noise_rms <= 1e-8:
        return speech
    scale = _rms(speech) / (10.0 ** (snr_db / 20.0) * noise_rms)
    return (speech + segment * scale).astype(np.float32)


def _add_competing_speech(
    speech: np.ndarray,
    competing: np.ndarray,
    *,
    sir_db: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if not speech.size or not competing.size:
        return speech
    requested = int(len(speech) * float(rng.uniform(0.15, 0.45)))
    overlap_length = min(len(competing), max(1, requested))
    source_start = int(rng.integers(0, len(competing) - overlap_length + 1))
    target_start = int(rng.integers(0, len(speech) - overlap_length + 1))
    segment = competing[source_start : source_start + overlap_length].copy()
    segment_rms = _rms(segment)
    primary = speech[target_start : target_start + overlap_length]
    if segment_rms <= 1e-8:
        return speech
    scale = _rms(primary) / (10.0 ** (sir_db / 20.0) * segment_rms)
    output = speech.copy()
    output[target_start : target_start + overlap_length] += segment * scale
    return output.astype(np.float32)
