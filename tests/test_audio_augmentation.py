from __future__ import annotations

import pathlib

import numpy as np
import soundfile as sf

from scripts.audio_augmentation import RobustAudioAugmenter


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
