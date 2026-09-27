"""Run streamed LibriSpeech adapter training on a branch-scoped Modal T4."""

from __future__ import annotations

import json
import pathlib
import re

import modal

BRANCH = "streaming-audio-classifier-prototype"
APP_NAME = f"ultravox-adapter-training-{BRANCH}"
MINUTES = 60
REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[1]
CACHE_PATH = pathlib.Path("/cache")

app = modal.App(APP_NAME)
image = (
    modal.Image.from_registry("nvcr.io/nvidia/pytorch:24.12-py3")
    .entrypoint([])
    .env(
        {
            "HF_HOME": str(CACHE_PATH / "huggingface"),
            "PIP_CONFIG_FILE": "/dev/null",
            "PIP_EXTRA_INDEX_URL": "",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
        }
    )
    .apt_install("libsndfile1")
    .pip_install(
        "datasets==4.8.4",
        "huggingface_hub[hf_xet]==0.36.0",
        "jiwer==3.1.0",
        "peft==0.11.1",
        "safetensors==0.5.3",
        "scipy==1.14.1",
        "soundfile==0.13.1",
        "transformers==4.49.0",
    )
    .add_local_dir(
        REPOSITORY_ROOT / "ultravox",
        remote_path="/root/ultravox",
        copy=True,
    )
    .add_local_dir(
        REPOSITORY_ROOT / "scripts",
        remote_path="/root/scripts",
        copy=True,
    )
    .add_local_file(
        REPOSITORY_ROOT / ".model-cache" / "shared-whisper-adapter" / "adapter.pt",
        remote_path="/root/bootstrap-adapter.pt",
        copy=True,
    )
)
cache_volume = modal.Volume.from_name(
    "ultravox-streaming-adapter-training",
    create_if_missing=True,
)
volumes = {CACHE_PATH: cache_volume}


@app.function(image=image, volumes=volumes, timeout=30 * MINUTES)
def prepare_assets() -> dict[str, str]:
    import shutil
    import urllib.request
    import zipfile

    from huggingface_hub import snapshot_download

    ultravox_path = CACHE_PATH / "ultravox-v05"
    snapshot_download(
        "fixie-ai/ultravox-v0_5-llama-3_2-1b",
        revision="b95bec8ab291eeb04b5cd600dd473377f6b79026",
        local_dir=ultravox_path,
        allow_patterns=[
            "*.json",
            "*.py",
            "model.safetensors",
        ],
    )
    bootstrap_path = CACHE_PATH / "shared-whisper-adapter" / "adapter.pt"
    bootstrap_path.parent.mkdir(parents=True, exist_ok=True)
    if not bootstrap_path.exists():
        shutil.copy2("/root/bootstrap-adapter.pt", bootstrap_path)
    dataset_path = CACHE_PATH / "librispeech"
    dataset_path.mkdir(parents=True, exist_ok=True)
    train_target = dataset_path / "train.parquet"
    train_second_target = dataset_path / "train-0001.parquet"
    validation_target = dataset_path / "validation.parquet"
    if not train_target.exists():
        train_snapshot = snapshot_download(
            "openslr/librispeech_asr",
            repo_type="dataset",
            allow_patterns=["clean/train.100/0000.parquet"],
        )
        shutil.copy2(
            pathlib.Path(train_snapshot) / "clean/train.100/0000.parquet",
            train_target,
        )
    if not train_second_target.exists():
        train_second_snapshot = snapshot_download(
            "openslr/librispeech_asr",
            repo_type="dataset",
            allow_patterns=["clean/train.100/0001.parquet"],
        )
        shutil.copy2(
            pathlib.Path(train_second_snapshot) / "clean/train.100/0001.parquet",
            train_second_target,
        )
    train_snapshot = snapshot_download(
        "openslr/librispeech_asr",
        repo_type="dataset",
        allow_patterns=["clean/train.100/*.parquet"],
    )
    train_shards = sorted(
        (pathlib.Path(train_snapshot) / "clean/train.100").glob("*.parquet")
    )
    for shard in train_shards:
        if shard.name in {"0000.parquet", "0001.parquet"}:
            continue
        target = dataset_path / f"train-extra-{shard.name}"
        if not target.exists():
            shutil.copy2(shard, target)
    if not validation_target.exists():
        validation_snapshot = snapshot_download(
            "openslr/librispeech_asr",
            repo_type="dataset",
            allow_patterns=["clean/validation/0000.parquet"],
        )
        shutil.copy2(
            pathlib.Path(validation_snapshot) / "clean/validation/0000.parquet",
            validation_target,
        )
    augmentation_root = CACHE_PATH / "rirs-noises"
    extracted_root = augmentation_root / "RIRS_NOISES"
    if not extracted_root.exists():
        augmentation_root.mkdir(parents=True, exist_ok=True)
        archive = augmentation_root / "rirs_noises.zip"
        if not archive.exists():
            urllib.request.urlretrieve(
                "https://www.openslr.org/resources/28/rirs_noises.zip",
                archive,
            )
        with zipfile.ZipFile(archive) as source:
            source.extractall(augmentation_root)
        archive.unlink(missing_ok=True)
    augmentation_wavs = list(extracted_root.rglob("*.wav"))
    cache_volume.commit()
    return {
        "ultravoxWeights": str(ultravox_path / "model.safetensors"),
        "bootstrapAdapter": str(bootstrap_path),
        "trainParquet": str(train_target),
        "trainSecondParquet": str(train_second_target),
        "trainShardCount": str(len(train_shards)),
        "validationParquet": str(validation_target),
        "augmentationAssets": str(extracted_root),
        "augmentationWavCount": str(len(augmentation_wavs)),
    }


@app.function(
    image=image,
    gpu="T4",
    volumes=volumes,
    timeout=6 * 60 * MINUTES,
)
def train_block(
    block_start: int = 0,
    block_size: int = 100,
    validation_samples: int = 50,
    validation_loss_samples: int = 25,
    validation_decode_samples: int = 10,
    encode_batch_size: int = 8,
    training_microbatch_size: int = 1,
    checkpoint_every_samples: int = 0,
    robust_augmentation: bool = False,
    noise_curriculum: bool = False,
    dataset_start_index: int = 0,
) -> dict[str, object]:
    import os
    import shutil
    import statistics
    import subprocess
    import threading
    import time

    model_cache = pathlib.Path("/root/.model-cache")
    model_cache.mkdir(parents=True, exist_ok=True)
    ultravox_target = model_cache / "ultravox-v05"
    bootstrap_target = model_cache / "shared-whisper-adapter" / "adapter.pt"
    if ultravox_target.exists() or ultravox_target.is_symlink():
        if ultravox_target.is_symlink():
            ultravox_target.unlink()
        elif ultravox_target.resolve() != (CACHE_PATH / "ultravox-v05").resolve():
            shutil.rmtree(ultravox_target)
    if not ultravox_target.exists():
        ultravox_target.symlink_to(
            CACHE_PATH / "ultravox-v05", target_is_directory=True
        )
    bootstrap_target.parent.mkdir(parents=True, exist_ok=True)
    if not bootstrap_target.exists():
        bootstrap_target.symlink_to(
            CACHE_PATH / "shared-whisper-adapter" / "adapter.pt"
        )
    training_target = model_cache / "shared-whisper-adapter-librispeech"
    training_cache = CACHE_PATH / "shared-whisper-adapter-librispeech"
    training_cache.mkdir(parents=True, exist_ok=True)
    if training_target.exists() or training_target.is_symlink():
        if training_target.is_symlink():
            training_target.unlink()
        else:
            shutil.rmtree(training_target)
    training_target.symlink_to(training_cache, target_is_directory=True)

    block_end = block_start + block_size
    if noise_curriculum:
        checkpoint = training_cache / "adapter-asr-noise-latest.pt"
        checkpoint_dir = training_cache / "noise-checkpoints"
    elif robust_augmentation:
        checkpoint = training_cache / "adapter-asr-robust-latest.pt"
        checkpoint_dir = training_cache / "robust-checkpoints"
    else:
        checkpoint = training_cache / "adapter-asr.pt"
        checkpoint_dir = training_cache / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if noise_curriculum:
        if block_start == 20_000:
            start_checkpoint = (
                training_cache / "robust-checkpoints" / "adapter-20000.pt"
            )
        else:
            start_checkpoint = checkpoint_dir / f"adapter-{block_start:05d}.pt"
        if not start_checkpoint.exists():
            raise FileNotFoundError(
                f"No prior checkpoint available for noise run: {start_checkpoint}"
            )
    else:
        start_checkpoint = (
            training_cache / "checkpoints" / f"adapter-{block_start:04d}.pt"
        )
    if robust_augmentation:
        if not start_checkpoint.exists():
            raise FileNotFoundError(
                f"No clean checkpoint available for robust run: {start_checkpoint}"
            )
    elif block_start:
        if start_checkpoint.exists():
            shutil.copy2(start_checkpoint, checkpoint)
        elif checkpoint.exists():
            shutil.copy2(checkpoint, start_checkpoint)
        else:
            raise FileNotFoundError(
                f"No checkpoint available for block starting at {block_start}"
            )

    command = [
        "python3",
        "-m",
        "scripts.train_shared_whisper_adapter_librispeech",
        "--train-samples",
        str(block_size if robust_augmentation or noise_curriculum else block_end),
        "--train-start-index",
        str(0 if robust_augmentation or noise_curriculum else block_start),
        "--validation-samples",
        str(validation_samples),
        "--encode-batch-size",
        str(encode_batch_size),
        "--training-microbatch-size",
        str(training_microbatch_size),
        "--epochs",
        "1",
        "--local-dataset-dir",
        str(CACHE_PATH / "librispeech"),
        "--validation-loss-samples",
        str(min(validation_samples, validation_loss_samples)),
        "--validation-decode-samples",
        str(min(validation_samples, validation_decode_samples)),
    ]
    if robust_augmentation or noise_curriculum:
        command.extend(
            [
                "--checkpoint-offset",
                str(block_start),
                "--resume-checkpoint",
                str(start_checkpoint),
            ]
        )
        if robust_augmentation:
            command.extend(
                [
                    "--augmentation-assets-dir",
                    str(CACHE_PATH / "rirs-noises" / "RIRS_NOISES"),
                ]
            )
        else:
            command.extend(
                [
                    "--noise-curriculum-assets-dir",
                    str(CACHE_PATH / "rirs-noises" / "RIRS_NOISES"),
                    "--dataset-start-index",
                    str(dataset_start_index),
                ]
            )
    if checkpoint_every_samples:
        command.extend(["--checkpoint-every-samples", str(checkpoint_every_samples)])
    if block_start:
        command.append("--resume")
    stage = {"value": "startup"}
    telemetry: dict[str, list[tuple[float, float]]] = {}
    stop_monitor = threading.Event()

    def monitor_gpu() -> None:
        while not stop_monitor.wait(0.1):
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode != 0:
                continue
            match = re.match(r"\s*([0-9.]+)\s*,\s*([0-9.]+)", completed.stdout.strip())
            if match:
                telemetry.setdefault(stage["value"], []).append(
                    (float(match.group(1)), float(match.group(2)))
                )

    monitor = threading.Thread(target=monitor_gpu, daemon=True)
    monitor.start()
    started = time.monotonic()
    process = subprocess.Popen(
        command,
        cwd="/root",
        env={**os.environ},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert process.stdout is not None
    output: list[str] = []
    checkpoint_results: list[dict[str, object]] = []
    for line in process.stdout:
        stripped = line.rstrip()
        print(stripped, flush=True)
        output.append(stripped)
        if stripped.startswith("checkpoint_result="):
            checkpoint_results.append(
                json.loads(stripped.removeprefix("checkpoint_result="))
            )
            cache_volume.commit()
        if stripped.startswith("features=train"):
            stage["value"] = "encoder_train"
        elif stripped.startswith("features=validation"):
            stage["value"] = "encoder_validation"
        elif stripped.startswith("loading Whisper"):
            stage["value"] = "decoder_load"
        elif stripped.startswith("stage=adapter_training_start"):
            stage["value"] = "adapter_training"
        elif stripped.startswith("stage=adapter_training_complete"):
            stage["value"] = "validation_loss"
        elif stripped.startswith("stage=validation_decode_start"):
            stage["value"] = "validation_decode"
    return_code = process.wait()
    stop_monitor.set()
    monitor.join(timeout=2)
    if return_code:
        raise RuntimeError(f"Training process failed with exit code {return_code}")
    final_checkpoint = checkpoint_dir / f"adapter-{block_end:04d}.pt"
    if not final_checkpoint.exists():
        shutil.copy2(checkpoint, final_checkpoint)

    def summarize(samples: list[tuple[float, float]]) -> dict[str, float]:
        utilization = [sample[0] for sample in samples]
        memory = [sample[1] for sample in samples]
        return {
            "sampleCount": len(samples),
            "gpuUtilizationMeanPercent": round(statistics.mean(utilization), 1),
            "gpuUtilizationP95Percent": round(
                sorted(utilization)[max(0, int(0.95 * len(utilization)) - 1)], 1
            ),
            "gpuUtilizationMaxPercent": round(max(utilization), 1),
            "memoryMaximumMiB": round(max(memory), 1),
        }

    result: dict[str, object] = {
        "gpu": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
        ).strip(),
        "blockStart": block_start,
        "blockEnd": block_end,
        "trainingBlockSize": block_size,
        "encodeBatchSize": encode_batch_size,
        "trainingMicrobatchSize": training_microbatch_size,
        "checkpointEverySamples": checkpoint_every_samples,
        "robustAugmentation": robust_augmentation,
        "noiseCurriculum": noise_curriculum,
        "datasetStartIndex": dataset_start_index,
        "checkpointResults": checkpoint_results,
        "elapsedSeconds": round(time.monotonic() - started, 1),
        "telemetry": {
            name: summarize(samples) for name, samples in telemetry.items() if samples
        },
        "tail": output[-30:],
    }
    result_path = (
        CACHE_PATH
        / "shared-whisper-adapter-librispeech"
        / (
            f"t4-{'noise-' if noise_curriculum else ('robust-' if robust_augmentation else '')}block-"
            f"{block_start:04d}-{block_end:04d}-mb{training_microbatch_size}.json"
        )
    )
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2))
    cache_volume.commit()
    return result


@app.function(
    image=image,
    gpu="T4",
    volumes=volumes,
    timeout=6 * 60 * MINUTES,
)
def train_speech_gate_and_joint(
    head_train_samples: int = 10_000,
    joint_train_samples: int = 10_000,
    checkpoint_every_samples: int = 1_000,
    head_batch_size: int = 50,
    joint_batch_size: int = 4,
    encode_batch_size: int = 8,
) -> dict[str, object]:
    import os
    import shutil
    import statistics
    import subprocess
    import threading
    import time

    model_cache = pathlib.Path("/root/.model-cache")
    model_cache.mkdir(parents=True, exist_ok=True)
    for relative in ("ultravox-v05", "shared-whisper-adapter-librispeech"):
        target = model_cache / relative
        source = CACHE_PATH / relative
        if target.exists() or target.is_symlink():
            if target.is_symlink():
                target.unlink()
            elif target.resolve() != source.resolve():
                shutil.rmtree(target)
        if not target.exists():
            target.symlink_to(source, target_is_directory=True)

    bootstrap_target = model_cache / "shared-whisper-adapter" / "adapter.pt"
    bootstrap_target.parent.mkdir(parents=True, exist_ok=True)
    if not bootstrap_target.exists():
        bootstrap_target.symlink_to(
            CACHE_PATH / "shared-whisper-adapter" / "adapter.pt"
        )

    stage = {"value": "feature_preparation"}
    telemetry: dict[str, list[tuple[float, float]]] = {}
    stop_monitor = threading.Event()

    def monitor_gpu() -> None:
        while not stop_monitor.wait(0.1):
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            match = re.match(r"\s*([0-9.]+)\s*,\s*([0-9.]+)", completed.stdout.strip())
            if completed.returncode == 0 and match:
                telemetry.setdefault(stage["value"], []).append(
                    (float(match.group(1)), float(match.group(2)))
                )

    monitor = threading.Thread(target=monitor_gpu, daemon=True)
    monitor.start()
    started = time.monotonic()
    output: list[str] = []
    checkpoint_results: list[dict[str, object]] = []

    def run(command: list[str]) -> None:
        process = subprocess.Popen(
            command,
            cwd="/root",
            env={**os.environ},
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            stripped = line.rstrip()
            print(stripped, flush=True)
            output.append(stripped)
            if stripped == "stage=head_training_start":
                stage["value"] = "head_training"
            elif stripped == "stage=joint_training_start":
                stage["value"] = "joint_training"
            elif stripped.startswith("loading Whisper"):
                stage["value"] = "decoder_load"
            elif stripped.startswith("checkpoint_result="):
                checkpoint_results.append(
                    json.loads(stripped.removeprefix("checkpoint_result="))
                )
                cache_volume.commit()
        return_code = process.wait()
        if return_code:
            raise RuntimeError(
                f"Training subprocess failed with exit code {return_code}"
            )

    def read_manifest(path: pathlib.Path) -> list[dict[str, object]]:
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line]

    def write_joint_manifest(
        tail_path: pathlib.Path,
        fill_path: pathlib.Path,
        head_path: pathlib.Path,
        output_path: pathlib.Path,
        count: int,
    ) -> None:
        head_ids = {str(row["sampleId"]) for row in read_manifest(head_path)}
        selected: list[dict[str, object]] = []
        selected_ids: set[str] = set()
        for path in (tail_path, fill_path):
            for row in read_manifest(path):
                sample_id = str(row["sampleId"])
                if sample_id in head_ids or sample_id in selected_ids:
                    continue
                selected.append(row)
                selected_ids.add(sample_id)
                if len(selected) == count:
                    break
            if len(selected) == count:
                break
        if len(selected) != count:
            raise RuntimeError(
                f"Only found {len(selected)} distinct phase-two records; "
                f"expected {count}"
            )
        missing: list[str] = []
        for row in selected:
            feature_path = pathlib.Path(str(row["path"]))
            if not feature_path.is_absolute():
                feature_path = pathlib.Path("/root") / feature_path
            if not feature_path.exists():
                missing.append(str(feature_path))
        if missing:
            raise FileNotFoundError(f"Missing phase-two feature state: {missing[0]}")
        temporary = output_path.with_suffix(".jsonl.tmp")
        temporary.write_text("".join(json.dumps(row) + "\n" for row in selected))
        temporary.replace(output_path)
        print(
            f"features={output_path.stem} merged={len(selected)}/{count} "
            f"tail={len(read_manifest(tail_path))} fill={len(read_manifest(fill_path))}",
            flush=True,
        )

    try:
        features = CACHE_PATH / "shared-whisper-adapter-librispeech" / "features"
        tail_manifest = features / "train-noise-v2-20000.jsonl"
        fill_manifest = features / "train-noise-v2-00000.jsonl"
        head_manifest = features / "train-noise-v2-10000.jsonl"
        joint_manifest = features / "train-noise-v2-joint-10000.jsonl"
        if len(read_manifest(joint_manifest)) < joint_train_samples:
            run(
                [
                    "python3",
                    "-m",
                    "scripts.train_shared_whisper_adapter_librispeech",
                    "--train-samples",
                    str(joint_train_samples),
                    "--dataset-start-index",
                    "20000",
                    "--validation-samples",
                    "50",
                    "--encode-batch-size",
                    str(encode_batch_size),
                    "--local-dataset-dir",
                    str(CACHE_PATH / "librispeech"),
                    "--noise-curriculum-assets-dir",
                    str(CACHE_PATH / "rirs-noises" / "RIRS_NOISES"),
                    "--prepare-only",
                    "--allow-partial-prepare",
                ]
            )
            tail_count = len(read_manifest(tail_manifest))
            fill_count = max(0, joint_train_samples - tail_count)
            if fill_count:
                run(
                    [
                        "python3",
                        "-m",
                        "scripts.train_shared_whisper_adapter_librispeech",
                        "--train-samples",
                        str(fill_count),
                        "--dataset-start-index",
                        "0",
                        "--validation-samples",
                        "50",
                        "--encode-batch-size",
                        str(encode_batch_size),
                        "--local-dataset-dir",
                        str(CACHE_PATH / "librispeech"),
                        "--noise-curriculum-assets-dir",
                        str(CACHE_PATH / "rirs-noises" / "RIRS_NOISES"),
                        "--prepare-only",
                    ]
                )
            write_joint_manifest(
                tail_manifest,
                fill_manifest,
                head_manifest,
                joint_manifest,
                joint_train_samples,
            )
            cache_volume.commit()
        else:
            print(
                f"features={joint_manifest.stem} cached="
                f"{joint_train_samples}/{joint_train_samples}",
                flush=True,
            )
        stage["value"] = "head_training"
        run(
            [
                "python3",
                "-m",
                "scripts.train_speech_presence_head",
                "--adapter-checkpoint",
                str(
                    CACHE_PATH
                    / "shared-whisper-adapter-librispeech"
                    / "robust-checkpoints"
                    / "adapter-20000.pt"
                ),
                "--head-train-samples",
                str(head_train_samples),
                "--joint-train-samples",
                str(joint_train_samples),
                "--joint-train-manifest",
                str(joint_manifest),
                "--resume-head-checkpoint",
                str(
                    CACHE_PATH
                    / "shared-whisper-adapter-librispeech"
                    / "speech-head-10000.pt"
                ),
                "--checkpoint-every-samples",
                str(checkpoint_every_samples),
                "--head-batch-size",
                str(head_batch_size),
                "--joint-batch-size",
                str(joint_batch_size),
            ]
        )
    finally:
        stop_monitor.set()
        monitor.join(timeout=2)

    def summarize(samples: list[tuple[float, float]]) -> dict[str, float]:
        utilization = [sample[0] for sample in samples]
        memory = [sample[1] for sample in samples]
        return {
            "sampleCount": len(samples),
            "gpuUtilizationMeanPercent": round(statistics.mean(utilization), 1),
            "gpuUtilizationP95Percent": round(
                sorted(utilization)[max(0, int(0.95 * len(utilization)) - 1)], 1
            ),
            "gpuUtilizationMaxPercent": round(max(utilization), 1),
            "memoryMaximumMiB": round(max(memory), 1),
        }

    result: dict[str, object] = {
        "gpu": subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            text=True,
        ).strip(),
        "headTrainSamples": head_train_samples,
        "jointTrainSamples": joint_train_samples,
        "checkpointEverySamples": checkpoint_every_samples,
        "elapsedSeconds": round(time.monotonic() - started, 1),
        "checkpointResults": checkpoint_results,
        "telemetry": {
            name: summarize(samples) for name, samples in telemetry.items() if samples
        },
        "tail": output[-30:],
    }
    result_path = (
        CACHE_PATH
        / "shared-whisper-adapter-librispeech"
        / "t4-speech-head-10000-joint-10000.json"
    )
    result_path.write_text(json.dumps(result, indent=2))
    cache_volume.commit()
    return result


@app.local_entrypoint()
def main(
    block_start: int = 0,
    block_size: int = 100,
    end_samples: int = 0,
    validation_samples: int = 50,
    validation_loss_samples: int = 25,
    validation_decode_samples: int = 10,
    encode_batch_size: int = 8,
    training_microbatch_size: int = 1,
    checkpoint_every_samples: int = 0,
    robust_augmentation: bool = False,
    noise_curriculum: bool = False,
    dataset_start_index: int = 0,
    speech_gate: bool = False,
) -> None:
    print(json.dumps(prepare_assets.remote(), indent=2))
    if speech_gate:
        print(json.dumps(train_speech_gate_and_joint.remote(), indent=2))
        return
    final_sample = end_samples or block_start + block_size
    while block_start < final_sample:
        current_size = min(block_size, final_sample - block_start)
        result = train_block.remote(
            block_start=block_start,
            block_size=current_size,
            validation_samples=validation_samples,
            validation_loss_samples=validation_loss_samples,
            validation_decode_samples=validation_decode_samples,
            encode_batch_size=encode_batch_size,
            training_microbatch_size=training_microbatch_size,
            checkpoint_every_samples=checkpoint_every_samples,
            robust_augmentation=robust_augmentation,
            noise_curriculum=noise_curriculum,
            dataset_start_index=dataset_start_index,
        )
        print(json.dumps(result, indent=2))
        block_start += current_size
