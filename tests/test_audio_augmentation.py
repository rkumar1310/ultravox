from __future__ import annotations

import pathlib

import numpy as np
import soundfile as sf
import torch

from scripts.audio_augmentation import NoiseCurriculumAugmenter
from scripts.audio_augmentation import RobustAudioAugmenter
from ultravox.inference.shared_whisper_adapter import SpeechPresenceHead


def _assets(root: pathlib.Path) -> pathlib.Path:
    noise_dir = root / "pointsource_noises" / "noise-free-sound"
    rir_dir = root / "simulated_rirs" / "smallroom"
    noise_dir.mkdir(parents=True)
    rir_dir.mkdir(parents=True)
    rng = np.random.default_rng(7)
    sf.write(noise_dir / "environment.wav", rng.normal(0, 0.1, 32_000), 16_000)
    rir = np.zeros(2_000, dtype=np.float32)
    rir[0] = 1.0
    rir[320] = 0.3
    sf.write(rir_dir / "rir.wav", rir, 16_000)
    return root


def test_augmentation_is_deterministic_and_finite(tmp_path: pathlib.Path) -> None:
    assets = _assets(tmp_path / "RIRS_NOISES")
    speech = np.sin(np.linspace(0, 400, 24_000, dtype=np.float32)) * 0.1
    first = RobustAudioAugmenter(assets, seed=42)
    second = RobustAudioAugmenter(assets, seed=42)

    first.observe("context", speech[::-1])
    second.observe("context", speech[::-1])
    a = first.augment("sample", speech)
    b = second.augment("sample", speech)

    np.testing.assert_array_equal(a.samples, b.samples)
    assert a.samples.shape == speech.shape
    assert np.isfinite(a.samples).all()
    assert np.max(np.abs(a.samples)) <= 0.98
    assert first.asset_counts == {"noiseFiles": 1, "roomImpulseFiles": 1}


def test_reservoir_is_bounded(tmp_path: pathlib.Path) -> None:
    augmenter = RobustAudioAugmenter(
        _assets(tmp_path / "RIRS_NOISES"), seed=1, reservoir_size=3
    )
    audio = np.zeros(160, dtype=np.float32)
    for index in range(10):
        augmenter.observe(str(index), audio)

    assert len(augmenter._reservoir) == 3


def test_noise_curriculum_has_empty_and_noise_dominated_targets(
    tmp_path: pathlib.Path,
) -> None:
    assets = _assets(tmp_path / "RIRS_NOISES")
    speech = np.sin(np.linspace(0, 400, 24_000, dtype=np.float32)) * 0.1
    augmenter = NoiseCurriculumAugmenter(assets, seed=42)

    silence = augmenter.augment("silence", speech, force_kind="silence")
    noise_only = augmenter.augment("noise", speech, force_kind="noise-only")
    heavy = augmenter.augment("heavy", speech, force_kind="noise-dominated-speech")

    assert not silence.target_has_speech
    assert not noise_only.target_has_speech
    assert heavy.target_has_speech
    assert heavy.snr_db is not None
    assert -15.0 <= heavy.snr_db <= -8.0
    for result in (silence, noise_only, heavy):
        assert result.samples.shape == speech.shape
        assert np.isfinite(result.samples).all()
        assert np.max(np.abs(result.samples)) <= 0.98


def test_noise_curriculum_distribution_includes_both_failure_modes(
    tmp_path: pathlib.Path,
) -> None:
    assets = _assets(tmp_path / "RIRS_NOISES")
    speech = np.ones(8_000, dtype=np.float32) * 0.1
    augmenter = NoiseCurriculumAugmenter(assets, seed=99)

    results = [augmenter.augment(f"sample-{index}", speech) for index in range(200)]
    no_speech = sum(not result.target_has_speech for result in results)
    noise_dominated = sum(result.kind == "noise-dominated-speech" for result in results)

    assert 30 <= no_speech <= 70
    assert 75 <= noise_dominated <= 125


def test_speech_presence_head_ignores_padded_frames() -> None:
    torch.manual_seed(3)
    head = SpeechPresenceHead(hidden_size=8, projection_size=4).eval()
    source = torch.randn(2, 5, 8)
    mask = torch.tensor(
        [[True, True, True, False, False], [True, True, True, True, True]]
    )

    expected = head(source, mask)
    source[0, 3:] = 10_000
    actual = head(source, mask)

    torch.testing.assert_close(actual, expected)
