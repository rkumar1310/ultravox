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
    cache_volume.commit()
    return {
        "ultravoxWeights": str(ultravox_path / "model.safetensors"),
        "bootstrapAdapter": str(bootstrap_path),
        "trainParquet": str(train_target),
        "trainSecondParquet": str(train_second_target),
        "trainShardCount": str(len(train_shards)),
        "validationParquet": str(validation_target),
    }


@app.function(
    image=image,
    gpu="T4",
    volumes=volumes,
    timeout=30 * MINUTES,
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
    checkpoint = training_cache / "adapter-asr.pt"
    checkpoint_dir = training_cache / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if block_start:
        start_checkpoint = checkpoint_dir / f"adapter-{block_start:04d}.pt"
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
        str(block_end),
        "--train-start-index",
        str(block_start),
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
    if checkpoint_every_samples:
        command.extend(
            ["--checkpoint-every-samples", str(checkpoint_every_samples)]
        )
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
    shutil.copy2(checkpoint, checkpoint_dir / f"adapter-{block_end:04d}.pt")

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
        / f"t4-block-{block_start:04d}-{block_end:04d}-mb{training_microbatch_size}.json"
    )
    result_path.parent.mkdir(parents=True, exist_ok=True)
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
) -> None:
    print(json.dumps(prepare_assets.remote(), indent=2))
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
        )
        print(json.dumps(result, indent=2))
        block_start += current_size
