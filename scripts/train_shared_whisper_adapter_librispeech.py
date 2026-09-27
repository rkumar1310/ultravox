"""Train the shared-state Whisper adapter on streamed LibriSpeech audio.

Only the selected examples are fetched from Hugging Face.  The Ultravox audio
states are cached locally so subsequent ASR-loss epochs do not rerun the large
audio encoder or redownload audio.
"""

from __future__ import annotations

import argparse
import gc
import io
import json
import pathlib
import re
import shutil
import time
from dataclasses import dataclass

import numpy as np
import soundfile as sf
import torch
import transformers
from datasets import load_dataset
from jiwer import wer
from torch.nn.utils.rnn import pad_sequence

from scripts.audio_augmentation import NOISE_PROFILE_NAME
from scripts.audio_augmentation import PROFILE_NAME
from scripts.audio_augmentation import NoiseCurriculumAugmenter
from scripts.audio_augmentation import RobustAudioAugmenter
from scripts.dual_output_probe import _decode_whisper_official
from scripts.dual_output_probe import _load_whisper_decoder_model
from scripts.train_shared_whisper_adapter import HOLDOUT_AUDIO
from scripts.train_shared_whisper_adapter import MODEL_ID
from scripts.train_shared_whisper_adapter import ULTRAVOX_WEIGHTS
from scripts.train_shared_whisper_adapter import _load_prefixed_weights
from scripts.train_shared_whisper_adapter import _read_pcm16_wav
from scripts.train_shared_whisper_adapter import _teacher_forcing_tokens
from scripts.train_shared_whisper_adapter import _transcript_loss
from ultravox.inference.shared_whisper_adapter import ShortStateAdapter
from ultravox.model.ultravox_model import ModifiedWhisperEncoder

DATASET_ID = "openslr/librispeech_asr"
DATASET_CONFIG = "clean"
WORK_DIR = pathlib.Path(".model-cache/shared-whisper-adapter-librispeech")
FEATURE_DIR = WORK_DIR / "features"
CHECKPOINT = WORK_DIR / "adapter-asr.pt"
SYNTHETIC_BOOTSTRAP = pathlib.Path(".model-cache/shared-whisper-adapter/adapter.pt")


@dataclass(frozen=True)
class FeatureRecord:
    dataset_split: str
    sample_id: str
    text: str
    duration_seconds: float
    path: pathlib.Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-samples", type=int, default=2_000)
    parser.add_argument("--validation-samples", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle-buffer", type=int, default=256)
    parser.add_argument("--encode-batch-size", type=int, default=4)
    parser.add_argument("--training-microbatch-size", type=int, default=1)
    parser.add_argument("--train-start-index", type=int, default=0)
    parser.add_argument(
        "--dataset-start-index",
        type=int,
        default=0,
        help="Skip this many eligible source rows before preparing training audio.",
    )
    parser.add_argument(
        "--checkpoint-offset",
        type=int,
        default=0,
        help="Add this offset to checkpoint sample numbers.",
    )
    parser.add_argument("--max-duration-seconds", type=float, default=29.5)
    parser.add_argument(
        "--local-dataset-dir",
        type=pathlib.Path,
        help="Optional directory containing train.parquet and validation.parquet.",
    )
    parser.add_argument("--validation-loss-samples", type=int, default=50)
    parser.add_argument("--validation-decode-samples", type=int, default=20)
    parser.add_argument(
        "--checkpoint-every-samples",
        type=int,
        default=0,
        help="Save and evaluate an in-process checkpoint after this many samples.",
    )
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--allow-partial-prepare",
        action="store_true",
        help=(
            "Persist and return all eligible records when the source split ends "
            "before train-samples; valid only with --prepare-only."
        ),
    )
    parser.add_argument("--rebuild-features", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--resume-checkpoint",
        type=pathlib.Path,
        help="Explicit adapter checkpoint to load before training.",
    )
    parser.add_argument(
        "--augmentation-assets-dir",
        type=pathlib.Path,
        help="Enable deterministic mild noise, room, and speech-overlap augmentation.",
    )
    parser.add_argument(
        "--noise-curriculum-assets-dir",
        type=pathlib.Path,
        help=(
            "Train on silence/noise-only negatives and noise-dominated speech "
            "using deterministic real noise recordings."
        ),
    )
    parser.add_argument(
        "--no-synthetic-bootstrap",
        action="store_true",
        help="Start from random adapter weights rather than the prior synthetic checkpoint.",
    )
    args = parser.parse_args()

    if args.train_samples < 1 or args.validation_samples < 1:
        parser.error("train and validation sample counts must be positive")
    if not 0 <= args.train_start_index < args.train_samples:
        parser.error("train-start-index must be within the selected training samples")
    if args.checkpoint_every_samples < 0:
        parser.error("checkpoint-every-samples cannot be negative")
    if args.checkpoint_every_samples and args.epochs != 1:
        parser.error("checkpoint-every-samples currently requires exactly one epoch")
    if args.checkpoint_offset < 0:
        parser.error("checkpoint-offset cannot be negative")
    if args.dataset_start_index < 0:
        parser.error("dataset-start-index cannot be negative")
    if args.allow_partial_prepare and not args.prepare_only:
        parser.error("allow-partial-prepare requires prepare-only")
    if (
        args.augmentation_assets_dir is not None
        and args.noise_curriculum_assets_dir is not None
    ):
        parser.error("choose either robust augmentation or the noise curriculum")
    if args.resume_checkpoint is not None and not args.resume_checkpoint.exists():
        parser.error(f"resume checkpoint does not exist: {args.resume_checkpoint}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    dtype = torch.float16 if device.type in {"cuda", "mps"} else torch.float32
    print(f"device={device} dtype={dtype}", flush=True)
    processor = transformers.AutoProcessor.from_pretrained(MODEL_ID)
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    training_augmenter = None
    robust_validation_augmenter = None
    noise_training_augmenter = None
    noise_speech_validation_augmenter = None
    no_speech_validation_augmenter = None
    if args.augmentation_assets_dir is not None:
        training_augmenter = RobustAudioAugmenter(
            args.augmentation_assets_dir, seed=args.seed
        )
        robust_validation_augmenter = RobustAudioAugmenter(
            args.augmentation_assets_dir, seed=args.seed + 10_000
        )
        print(
            "augmentation="
            + json.dumps(
                {
                    "profile": PROFILE_NAME,
                    **training_augmenter.asset_counts,
                }
            ),
            flush=True,
        )
    if args.noise_curriculum_assets_dir is not None:
        noise_training_augmenter = NoiseCurriculumAugmenter(
            args.noise_curriculum_assets_dir, seed=args.seed + 20_000
        )
        noise_speech_validation_augmenter = NoiseCurriculumAugmenter(
            args.noise_curriculum_assets_dir, seed=args.seed + 30_000
        )
        no_speech_validation_augmenter = NoiseCurriculumAugmenter(
            args.noise_curriculum_assets_dir, seed=args.seed + 40_000
        )
        print(
            "augmentation="
            + json.dumps(
                {
                    "profile": NOISE_PROFILE_NAME,
                    **noise_training_augmenter.asset_counts,
                }
            ),
            flush=True,
        )

    train_records = _prepare_split(
        processor=processor,
        device=device,
        dtype=dtype,
        dataset_split="train.100",
        output_split=(
            f"train-{NOISE_PROFILE_NAME}-{args.dataset_start_index:05d}"
            if noise_training_augmenter
            else (f"train-{PROFILE_NAME}" if training_augmenter else "train")
        ),
        sample_count=args.train_samples,
        seed=args.seed,
        shuffle_buffer=args.shuffle_buffer,
        encode_batch_size=args.encode_batch_size,
        max_duration_seconds=args.max_duration_seconds,
        rebuild=args.rebuild_features,
        local_dataset_dir=args.local_dataset_dir,
        augmenter=training_augmenter,
        noise_augmenter=noise_training_augmenter,
        noise_kind=None,
        dataset_start_index=args.dataset_start_index,
        allow_partial=args.allow_partial_prepare,
    )
    validation_records = _prepare_split(
        processor=processor,
        device=device,
        dtype=dtype,
        dataset_split="validation",
        output_split="validation",
        sample_count=args.validation_samples,
        seed=args.seed,
        shuffle_buffer=0,
        encode_batch_size=args.encode_batch_size,
        max_duration_seconds=args.max_duration_seconds,
        rebuild=args.rebuild_features,
        local_dataset_dir=args.local_dataset_dir,
        augmenter=None,
        noise_augmenter=None,
        noise_kind=None,
        dataset_start_index=0,
        allow_partial=False,
    )
    robust_validation_records: list[FeatureRecord] = []
    if robust_validation_augmenter is not None:
        robust_validation_records = _prepare_split(
            processor=processor,
            device=device,
            dtype=dtype,
            dataset_split="validation",
            output_split=f"validation-{PROFILE_NAME}",
            sample_count=args.validation_samples,
            seed=args.seed + 10_000,
            shuffle_buffer=0,
            encode_batch_size=args.encode_batch_size,
            max_duration_seconds=args.max_duration_seconds,
            rebuild=args.rebuild_features,
            local_dataset_dir=args.local_dataset_dir,
            augmenter=robust_validation_augmenter,
            noise_augmenter=None,
            noise_kind=None,
            dataset_start_index=0,
            allow_partial=False,
        )
    noise_speech_validation_records: list[FeatureRecord] = []
    no_speech_validation_records: list[FeatureRecord] = []
    if noise_speech_validation_augmenter is not None:
        noise_speech_validation_records = _prepare_split(
            processor=processor,
            device=device,
            dtype=dtype,
            dataset_split="validation",
            output_split=f"validation-{NOISE_PROFILE_NAME}-speech",
            sample_count=args.validation_samples,
            seed=args.seed + 30_000,
            shuffle_buffer=0,
            encode_batch_size=args.encode_batch_size,
            max_duration_seconds=args.max_duration_seconds,
            rebuild=args.rebuild_features,
            local_dataset_dir=args.local_dataset_dir,
            augmenter=None,
            noise_augmenter=noise_speech_validation_augmenter,
            noise_kind="noise-dominated-speech",
            dataset_start_index=0,
            allow_partial=False,
        )
        no_speech_validation_records = _prepare_split(
            processor=processor,
            device=device,
            dtype=dtype,
            dataset_split="validation",
            output_split=f"validation-{NOISE_PROFILE_NAME}-empty",
            sample_count=args.validation_samples,
            seed=args.seed + 40_000,
            shuffle_buffer=0,
            encode_batch_size=args.encode_batch_size,
            max_duration_seconds=args.max_duration_seconds,
            rebuild=args.rebuild_features,
            local_dataset_dir=args.local_dataset_dir,
            augmenter=None,
            noise_augmenter=no_speech_validation_augmenter,
            noise_kind="no-speech",
            dataset_start_index=0,
            allow_partial=False,
        )
    print(
        json.dumps(
            {
                "dataset": DATASET_ID,
                "streamed": True,
                "authenticationRequired": False,
                "trainSamples": len(train_records),
                "validationSamples": len(validation_records),
                "robustValidationSamples": len(robust_validation_records),
                "noiseSpeechValidationSamples": len(noise_speech_validation_records),
                "noSpeechValidationSamples": len(no_speech_validation_records),
                "trainHours": round(
                    sum(record.duration_seconds for record in train_records) / 3600,
                    3,
                ),
                "augmentationStats": (
                    training_augmenter.stats if training_augmenter else None
                ),
                "robustValidationAugmentationStats": (
                    robust_validation_augmenter.stats
                    if robust_validation_augmenter
                    else None
                ),
                "noiseTrainingStats": (
                    noise_training_augmenter.stats if noise_training_augmenter else None
                ),
                "noiseSpeechValidationStats": (
                    noise_speech_validation_augmenter.stats
                    if noise_speech_validation_augmenter
                    else None
                ),
                "noSpeechValidationStats": (
                    no_speech_validation_augmenter.stats
                    if no_speech_validation_augmenter
                    else None
                ),
                "validationHours": round(
                    sum(record.duration_seconds for record in validation_records)
                    / 3600,
                    3,
                ),
            },
            indent=2,
        ),
        flush=True,
    )
    if args.prepare_only:
        return

    training_records = train_records[args.train_start_index :]
    robust_run = args.augmentation_assets_dir is not None
    noise_run = args.noise_curriculum_assets_dir is not None
    if noise_run:
        run_checkpoint = WORK_DIR / "adapter-asr-noise-latest.pt"
        checkpoint_dir = WORK_DIR / "noise-checkpoints"
        best_checkpoint = WORK_DIR / "adapter-asr-noise-best.pt"
        results_path = WORK_DIR / (
            f"noise-{args.checkpoint_offset:05d}-"
            f"{args.checkpoint_offset + len(training_records):05d}-results.jsonl"
        )
    elif robust_run:
        run_checkpoint = WORK_DIR / "adapter-asr-robust-latest.pt"
        checkpoint_dir = WORK_DIR / "robust-checkpoints"
        best_checkpoint = WORK_DIR / "adapter-asr-robust-best.pt"
        results_path = WORK_DIR / "robust-10000-20000-results.jsonl"
    else:
        run_checkpoint = CHECKPOINT
        checkpoint_dir = WORK_DIR / "checkpoints"
        best_checkpoint = run_checkpoint
        results_path = None
    if results_path is not None:
        results_path.unlink(missing_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    adapter = ShortStateAdapter().to(device=device, dtype=torch.float32)
    if args.resume_checkpoint is not None:
        adapter.load_state_dict(
            torch.load(args.resume_checkpoint, map_location=device, weights_only=True)
        )
        print(f"resumed={args.resume_checkpoint}", flush=True)
    elif args.resume and run_checkpoint.exists():
        adapter.load_state_dict(
            torch.load(run_checkpoint, map_location=device, weights_only=True)
        )
        print(f"resumed={run_checkpoint}", flush=True)
    elif not args.no_synthetic_bootstrap and SYNTHETIC_BOOTSTRAP.exists():
        adapter.load_state_dict(
            torch.load(SYNTHETIC_BOOTSTRAP, map_location=device, weights_only=True)
        )
        print(f"bootstrapped={SYNTHETIC_BOOTSTRAP}", flush=True)

    decoder = _load_whisper_decoder_model(MODEL_ID, device, dtype).eval()
    for parameter in decoder.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.learning_rate)

    checkpoint_results: list[dict[str, object]] = []
    best_validation_loss = float("inf")
    best_validation_score = float("inf")
    best_trained_samples = args.checkpoint_offset + args.train_start_index
    if robust_run or noise_run:
        baseline_result = _evaluate_checkpoint(
            trained_samples=best_trained_samples,
            checkpoint_path=args.resume_checkpoint or run_checkpoint,
            train_loss=None,
            adapter=adapter,
            decoder=decoder,
            processor=processor,
            clean_records=validation_records,
            robust_records=robust_validation_records,
            noise_speech_records=noise_speech_validation_records,
            no_speech_records=no_speech_validation_records,
            validation_loss_samples=args.validation_loss_samples,
            validation_decode_samples=args.validation_decode_samples,
            device=device,
            dtype=dtype,
        )
        checkpoint_results.append(baseline_result)
        best_validation_loss = float(baseline_result["validationCe"])
        best_validation_score = float(baseline_result["selectionScore"])
        torch.save(adapter.state_dict(), best_checkpoint)
        _append_jsonl(results_path, baseline_result)
        print("checkpoint_result=" + json.dumps(baseline_result), flush=True)
    for epoch in range(1, args.epochs + 1):
        checkpoint_span = args.checkpoint_every_samples or len(training_records)
        epoch_started = time.monotonic()
        epoch_processed = 0
        for chunk_offset in range(0, len(training_records), checkpoint_span):
            chunk_records = training_records[
                chunk_offset : chunk_offset + checkpoint_span
            ]
            adapter.train()
            losses: list[torch.Tensor] = []
            length_sorted = sorted(
                range(len(chunk_records)),
                key=lambda index: (
                    chunk_records[index].duration_seconds,
                    len(chunk_records[index].text),
                ),
            )
            batches = [
                length_sorted[offset : offset + args.training_microbatch_size]
                for offset in range(
                    0, len(length_sorted), args.training_microbatch_size
                )
            ]
            batch_order = torch.randperm(len(batches)).tolist()
            prepared_batches = []
            for batch_index in batch_order:
                indexes = batches[batch_index]
                records = [chunk_records[index] for index in indexes]
                tensors = _load_training_batch(records, processor, torch.device("cpu"))
                if device.type == "cuda":
                    tensors = tuple(tensor.pin_memory() for tensor in tensors)
                prepared_batches.append((len(records), tensors))
            absolute_end = (
                args.checkpoint_offset
                + args.train_start_index
                + chunk_offset
                + len(chunk_records)
            )
            print(
                f"stage=adapter_training_start epoch={epoch} checkpoint={absolute_end}",
                flush=True,
            )
            chunk_processed = 0
            for record_count, cpu_tensors in prepared_batches:
                source, source_mask, tokens, labels = (
                    tensor.to(device=device, non_blocking=device.type == "cuda")
                    for tensor in cpu_tensors
                )
                optimizer.zero_grad(set_to_none=True)
                loss = _batched_transcript_loss(
                    decoder,
                    adapter(source, source_mask),
                    tokens,
                    labels,
                    dtype,
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
                optimizer.step()
                losses.append(loss.detach())
                chunk_processed += record_count
                epoch_processed += record_count
                if chunk_processed % 50 == 0 or chunk_processed == len(chunk_records):
                    recent_loss = float(torch.stack(losses[-50:]).mean().cpu())
                    elapsed = time.monotonic() - epoch_started
                    print(
                        f"epoch={epoch} samples={epoch_processed}/"
                        f"{len(training_records)} microbatch={record_count} "
                        f"train_ce={recent_loss:.5f} "
                        f"samples_per_second={epoch_processed / elapsed:.3f}",
                        flush=True,
                    )

            train_loss = float(torch.stack(losses).mean().cpu())
            print(
                f"stage=adapter_training_complete epoch={epoch} "
                f"checkpoint={absolute_end}",
                flush=True,
            )
            print(
                f"stage=validation_loss_start epoch={epoch} checkpoint={absolute_end}",
                flush=True,
            )
            torch.save(adapter.state_dict(), run_checkpoint)
            milestone_checkpoint = checkpoint_dir / f"adapter-{absolute_end:04d}.pt"
            torch.save(adapter.state_dict(), milestone_checkpoint)
            print(
                f"stage=validation_decode_start checkpoint={absolute_end}",
                flush=True,
            )
            checkpoint_result = _evaluate_checkpoint(
                trained_samples=absolute_end,
                checkpoint_path=milestone_checkpoint,
                train_loss=train_loss,
                adapter=adapter,
                decoder=decoder,
                processor=processor,
                clean_records=validation_records,
                robust_records=robust_validation_records,
                noise_speech_records=noise_speech_validation_records,
                no_speech_records=no_speech_validation_records,
                validation_loss_samples=args.validation_loss_samples,
                validation_decode_samples=args.validation_decode_samples,
                device=device,
                dtype=dtype,
            )
            validation_loss = float(checkpoint_result["validationCe"])
            checkpoint_score = float(checkpoint_result["selectionScore"])
            best_validation_loss = min(best_validation_loss, validation_loss)
            if checkpoint_score < best_validation_score:
                best_validation_score = checkpoint_score
                best_trained_samples = absolute_end
                shutil.copy2(milestone_checkpoint, best_checkpoint)
            print(
                f"epoch={epoch} checkpoint={absolute_end} "
                f"train_ce={train_loss:.5f} "
                f"validation_ce={validation_loss:.5f}",
                flush=True,
            )
            print(
                f"stage=validation_loss_complete epoch={epoch} "
                f"checkpoint={absolute_end}",
                flush=True,
            )
            checkpoint_results.append(checkpoint_result)
            if results_path is not None:
                _append_jsonl(results_path, checkpoint_result)
            print(
                "checkpoint_result=" + json.dumps(checkpoint_result),
                flush=True,
            )
            print(
                f"stage=validation_decode_complete checkpoint={absolute_end}",
                flush=True,
            )
            del prepared_batches

    final_checkpoint = checkpoint_results[-1]
    result: dict[str, object] = {
        "checkpoint": str(run_checkpoint),
        "bestCheckpoint": str(best_checkpoint),
        "bestCheckpointSamples": best_trained_samples,
        "bestValidationScore": best_validation_score,
        "checkpointResultsPath": str(results_path) if results_path else None,
        "trainStartIndex": args.train_start_index,
        "datasetStartIndex": args.dataset_start_index,
        "trainEndIndex": args.train_samples,
        "checkpointOffset": args.checkpoint_offset,
        "trainingMicrobatchSize": args.training_microbatch_size,
        "bestValidationCe": best_validation_loss,
        "decodedValidationSamples": final_checkpoint["decodedValidationSamples"],
        "validationWer": final_checkpoint["validationWer"],
        "examples": final_checkpoint["examples"],
        "checkpointResults": checkpoint_results,
    }
    if HOLDOUT_AUDIO.exists():
        result["blueBicycleTranscript"] = _decode_local_holdout(
            adapter, decoder, processor, device, dtype
        )
    print(json.dumps(result, indent=2), flush=True)


def _evaluate_checkpoint(
    *,
    trained_samples: int,
    checkpoint_path: pathlib.Path,
    train_loss: float | None,
    adapter: ShortStateAdapter,
    decoder,
    processor,
    clean_records: list[FeatureRecord],
    robust_records: list[FeatureRecord],
    noise_speech_records: list[FeatureRecord],
    no_speech_records: list[FeatureRecord],
    validation_loss_samples: int,
    validation_decode_samples: int,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, object]:
    validation_loss = _validation_loss(
        adapter=adapter,
        decoder=decoder,
        processor=processor,
        records=clean_records[:validation_loss_samples],
        device=device,
        dtype=dtype,
    )
    clean_decoded = _decode_validation(
        adapter=adapter,
        decoder=decoder,
        processor=processor,
        records=clean_records[:validation_decode_samples],
        device=device,
        dtype=dtype,
    )
    robust_decoded = _decode_validation(
        adapter=adapter,
        decoder=decoder,
        processor=processor,
        records=robust_records[:validation_decode_samples],
        device=device,
        dtype=dtype,
    )
    noise_speech_decoded = _decode_validation(
        adapter=adapter,
        decoder=decoder,
        processor=processor,
        records=noise_speech_records[:validation_decode_samples],
        device=device,
        dtype=dtype,
    )
    no_speech_decoded = _decode_validation(
        adapter=adapter,
        decoder=decoder,
        processor=processor,
        records=no_speech_records[:validation_decode_samples],
        device=device,
        dtype=dtype,
    )
    clean_wer = _decoded_wer(clean_decoded)
    robust_wer = _decoded_wer(robust_decoded) if robust_decoded else clean_wer
    noise_speech_wer = (
        _decoded_wer(noise_speech_decoded) if noise_speech_decoded else clean_wer
    )
    false_transcriptions = [
        item for item in no_speech_decoded if _normalize_for_wer(item["hypothesis"])
    ]
    no_speech_false_positive_rate = (
        len(false_transcriptions) / len(no_speech_decoded) if no_speech_decoded else 0.0
    )
    no_speech_hallucinated_words = sum(
        len(_normalize_for_wer(item["hypothesis"]).split())
        for item in no_speech_decoded
    )
    combined = [*clean_decoded, *robust_decoded]
    selection_score = clean_wer
    if robust_decoded:
        selection_score += robust_wer
    if noise_speech_decoded:
        selection_score += noise_speech_wer
    if no_speech_decoded:
        selection_score += no_speech_false_positive_rate
    return {
        "trainedSamples": trained_samples,
        "checkpoint": str(checkpoint_path),
        "trainCe": train_loss,
        "validationCe": validation_loss,
        "validationWer": _decoded_wer(combined),
        "cleanValidationWer": clean_wer,
        "robustValidationWer": robust_wer,
        "noiseSpeechValidationWer": noise_speech_wer,
        "noSpeechFalsePositiveRate": no_speech_false_positive_rate,
        "noSpeechHallucinatedWords": no_speech_hallucinated_words,
        "selectionScore": selection_score,
        "decodedValidationSamples": (
            len(combined) + len(noise_speech_decoded) + len(no_speech_decoded)
        ),
        "cleanExamples": clean_decoded[:3],
        "robustExamples": robust_decoded[:3],
        "noiseSpeechExamples": noise_speech_decoded[:3],
        "noSpeechExamples": no_speech_decoded[:3],
        "examples": [
            *clean_decoded[:2],
            *robust_decoded[:1],
            *noise_speech_decoded[:1],
            *no_speech_decoded[:1],
        ],
    }


def _decoded_wer(decoded: list[dict[str, str]]) -> float:
    if not decoded:
        raise ValueError("cannot compute WER for an empty validation set")
    return wer(
        [_normalize_for_wer(item["reference"]) for item in decoded],
        [_normalize_for_wer(item["hypothesis"]) for item in decoded],
    )


def _append_jsonl(path: pathlib.Path, item: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as destination:
        destination.write(json.dumps(item) + "\n")
        destination.flush()


def _prepare_split(
    *,
    processor,
    device: torch.device,
    dtype: torch.dtype,
    dataset_split: str,
    output_split: str,
    sample_count: int,
    seed: int,
    shuffle_buffer: int,
    encode_batch_size: int,
    max_duration_seconds: float,
    rebuild: bool,
    local_dataset_dir: pathlib.Path | None,
    augmenter: RobustAudioAugmenter | None,
    noise_augmenter: NoiseCurriculumAugmenter | None,
    noise_kind: str | None,
    dataset_start_index: int,
    allow_partial: bool,
) -> list[FeatureRecord]:
    output_dir = FEATURE_DIR / output_split
    manifest_path = FEATURE_DIR / f"{output_split}.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [] if rebuild else _read_manifest(manifest_path)
    if len(existing) >= sample_count and all(
        record.path.exists() for record in existing[:sample_count]
    ):
        print(
            f"features={output_split} cached={sample_count}/{sample_count}",
            flush=True,
        )
        return existing[:sample_count]

    if rebuild:
        for path in output_dir.glob("*.pt"):
            path.unlink()
        manifest_path.unlink(missing_ok=True)
        existing = []

    if local_dataset_dir is None:
        dataset = load_dataset(
            DATASET_ID,
            DATASET_CONFIG,
            split=dataset_split,
            streaming=True,
        ).decode(False)
    else:
        if dataset_split == "train.100":
            # Preserve the order used by the first 2,000 cached features, then
            # append the additional LibriSpeech shards.
            data_files = [
                local_dataset_dir / "train-0001.parquet",
                local_dataset_dir / "train.parquet",
                *sorted(local_dataset_dir.glob("train-extra-*.parquet")),
            ]
            data_files = [path for path in data_files if path.exists()]
        else:
            data_files = [local_dataset_dir / "validation.parquet"]
        if not data_files:
            raise FileNotFoundError(
                f"No local Parquet files found for {dataset_split} in "
                f"{local_dataset_dir}"
            )
        dataset = load_dataset(
            "parquet",
            data_files=[str(path) for path in data_files],
            split="train",
            streaming=True,
        ).decode(False)
    if shuffle_buffer:
        dataset = dataset.shuffle(seed=seed, buffer_size=shuffle_buffer)

    config = transformers.WhisperConfig.from_pretrained(MODEL_ID)
    encoder = ModifiedWhisperEncoder(config).to(device=device, dtype=dtype).eval()
    encoder.init_latency_mask(None, dtype=dtype)
    _load_prefixed_weights(
        encoder,
        ULTRAVOX_WEIGHTS,
        prefix="audio_tower.",
        device=device,
    )

    records: list[FeatureRecord] = []
    pending: list[tuple[str, str, np.ndarray, float, pathlib.Path]] = []
    last_manifest_count = len(existing)
    started = time.monotonic()
    eligible_index = 0
    for row in dataset:
        if len(records) + len(pending) >= sample_count:
            break
        audio, sample_rate = _decode_audio(row["audio"])
        duration_seconds = len(audio) / sample_rate
        if duration_seconds > max_duration_seconds or not row["text"].strip():
            continue
        if eligible_index < dataset_start_index:
            eligible_index += 1
            continue
        eligible_index += 1

        index = len(records) + len(pending)
        sample_id = str(row["id"])
        target_text = str(row["text"])
        clean_audio = audio
        if noise_augmenter is not None:
            augmented = noise_augmenter.augment(
                sample_id,
                clean_audio,
                force_kind=noise_kind,
            )
            audio = augmented.samples
            if not augmented.target_has_speech:
                target_text = ""
        feature_path = output_dir / f"{index:05d}-{sample_id}.pt"
        if not pending and index < len(existing):
            cached = existing[index]
            if cached.sample_id == sample_id and cached.path.exists():
                records.append(cached)
                if augmenter is not None:
                    augmenter.observe(sample_id, audio)
                continue
        if feature_path.exists():
            records.append(
                FeatureRecord(
                    dataset_split=dataset_split,
                    sample_id=sample_id,
                    text=target_text,
                    duration_seconds=duration_seconds,
                    path=feature_path,
                )
            )
            if augmenter is not None:
                augmenter.observe(sample_id, audio)
            if len(records) - last_manifest_count >= 32:
                _write_manifest(manifest_path, records)
                last_manifest_count = len(records)
            _report_feature_progress(output_split, len(records), sample_count, started)
            continue
        if augmenter is not None:
            audio = augmenter.augment(sample_id, clean_audio).samples
            augmenter.observe(sample_id, clean_audio)
        pending.append(
            (
                sample_id,
                target_text,
                audio,
                duration_seconds,
                feature_path,
            )
        )
        if len(pending) >= encode_batch_size:
            records.extend(
                _encode_feature_batch(
                    encoder=encoder,
                    processor=processor,
                    pending=pending,
                    dataset_split=dataset_split,
                    device=device,
                    dtype=dtype,
                )
            )
            pending.clear()
            _report_feature_progress(output_split, len(records), sample_count, started)
            if len(records) - last_manifest_count >= 32:
                _write_manifest(manifest_path, records)
                last_manifest_count = len(records)

    if pending:
        records.extend(
            _encode_feature_batch(
                encoder=encoder,
                processor=processor,
                pending=pending,
                dataset_split=dataset_split,
                device=device,
                dtype=dtype,
            )
        )
        _report_feature_progress(output_split, len(records), sample_count, started)

    del encoder
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
    _write_manifest(manifest_path, records)
    if len(records) != sample_count and not allow_partial:
        raise RuntimeError(
            f"Only prepared {len(records)} of {sample_count} requested "
            f"examples for {dataset_split}"
        )
    if len(records) != sample_count:
        print(
            f"features={output_split} partial={len(records)}/{sample_count}",
            flush=True,
        )
    return records


def _encode_feature_batch(
    *,
    encoder: ModifiedWhisperEncoder,
    processor,
    pending: list[tuple[str, str, np.ndarray, float, pathlib.Path]],
    dataset_split: str,
    device: torch.device,
    dtype: torch.dtype,
) -> list[FeatureRecord]:
    try:
        return _encode_feature_batch_once(
            encoder=encoder,
            processor=processor,
            pending=pending,
            dataset_split=dataset_split,
            device=device,
            dtype=dtype,
        )
    except RuntimeError as error:
        if "out of memory" not in str(error).lower() or len(pending) == 1:
            raise
        print(
            f"encoder_batch_oom={len(pending)} retrying_as_smaller_batches",
            flush=True,
        )
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()
        midpoint = len(pending) // 2
        return [
            *_encode_feature_batch(
                encoder=encoder,
                processor=processor,
                pending=pending[:midpoint],
                dataset_split=dataset_split,
                device=device,
                dtype=dtype,
            ),
            *_encode_feature_batch(
                encoder=encoder,
                processor=processor,
                pending=pending[midpoint:],
                dataset_split=dataset_split,
                device=device,
                dtype=dtype,
            ),
        ]


def _encode_feature_batch_once(
    *,
    encoder: ModifiedWhisperEncoder,
    processor,
    pending: list[tuple[str, str, np.ndarray, float, pathlib.Path]],
    dataset_split: str,
    device: torch.device,
    dtype: torch.dtype,
) -> list[FeatureRecord]:
    audio_arrays = [item[2] for item in pending]
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
            features.input_features.to(device=device, dtype=dtype),
            audio_len=audio_lengths,
        ).last_hidden_state
    encoded_lengths = encoder._get_feat_extract_output_lengths(audio_lengths)

    records: list[FeatureRecord] = []
    for index, (sample_id, text, _, duration_seconds, feature_path) in enumerate(
        pending
    ):
        state = short[index : index + 1, : int(encoded_lengths[index])]
        torch.save({"short": state.cpu().half()}, feature_path)
        records.append(
            FeatureRecord(
                dataset_split=dataset_split,
                sample_id=sample_id,
                text=text,
                duration_seconds=duration_seconds,
                path=feature_path,
            )
        )
    del state, short, encoded_lengths, features
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
    return records


def _report_feature_progress(
    output_split: str,
    prepared: int,
    requested: int,
    started: float,
) -> None:
    if prepared == requested or prepared <= 8 or prepared % 16 == 0:
        elapsed = time.monotonic() - started
        print(
            f"features={output_split} prepared={prepared}/{requested} "
            f"samples_per_second={prepared / elapsed:.3f}",
            flush=True,
        )


def _decode_audio(audio: dict[str, object]) -> tuple[np.ndarray, int]:
    audio_bytes = audio.get("bytes")
    audio_path = audio.get("path")
    source = io.BytesIO(audio_bytes) if audio_bytes is not None else audio_path
    if source is None:
        raise ValueError("Streamed audio row has neither bytes nor a path")
    samples, sample_rate = sf.read(source, dtype="float32", always_2d=False)
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    if sample_rate != 16_000:
        raise ValueError(f"Expected 16 kHz LibriSpeech audio, got {sample_rate}")
    return samples, sample_rate


def _normalize_for_wer(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s']", " ", text.lower()).split())


def _read_manifest(path: pathlib.Path) -> list[FeatureRecord]:
    if not path.exists():
        return []
    records: list[FeatureRecord] = []
    with path.open() as manifest:
        for line in manifest:
            item = json.loads(line)
            records.append(
                FeatureRecord(
                    dataset_split=item["datasetSplit"],
                    sample_id=item["sampleId"],
                    text=item["text"],
                    duration_seconds=float(item["durationSeconds"]),
                    path=pathlib.Path(item["path"]),
                )
            )
    return records


def _write_manifest(path: pathlib.Path, records: list[FeatureRecord]) -> None:
    temporary = path.with_suffix(".jsonl.tmp")
    with temporary.open("w") as manifest:
        for record in records:
            manifest.write(
                json.dumps(
                    {
                        "datasetSplit": record.dataset_split,
                        "sampleId": record.sample_id,
                        "text": record.text,
                        "durationSeconds": record.duration_seconds,
                        "path": str(record.path),
                    }
                )
                + "\n"
            )
    temporary.replace(path)


def _load_state(path: pathlib.Path, device: torch.device) -> torch.Tensor:
    pair = torch.load(path, map_location="cpu", weights_only=True)
    return pair["short"].to(device=device, dtype=torch.float32)


def _load_training_batch(
    records: list[FeatureRecord],
    processor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    states = [_load_state(record.path, device).squeeze(0) for record in records]
    lengths = torch.tensor([state.shape[0] for state in states], device=device)
    source = pad_sequence(states, batch_first=True)
    source_mask = (
        torch.arange(source.shape[1], device=device)[None, :] < lengths[:, None]
    )

    token_rows = [
        _teacher_forcing_tokens(processor, record.text, device).squeeze(0)
        for record in records
    ]
    token_lengths = [row.shape[0] for row in token_rows]
    tokens = pad_sequence(
        token_rows,
        batch_first=True,
        padding_value=processor.tokenizer.eos_token_id,
    )
    labels = tokens[:, 1:].clone()
    for index, length in enumerate(token_lengths):
        labels[index, length - 1 :] = -100
    return source, source_mask, tokens, labels


def _batched_transcript_loss(
    model: transformers.WhisperForConditionalGeneration,
    encoder_state: torch.Tensor,
    tokens: torch.Tensor,
    labels: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    output = model.model.decoder(
        input_ids=tokens[:, :-1],
        encoder_hidden_states=encoder_state.to(dtype),
        use_cache=False,
        return_dict=True,
    )
    logits = model.proj_out(output.last_hidden_state).float()
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=-100,
    )


@torch.no_grad()
def _validation_loss(
    *,
    adapter: ShortStateAdapter,
    decoder,
    processor,
    records: list[FeatureRecord],
    device: torch.device,
    dtype: torch.dtype,
) -> float:
    adapter.eval()
    losses: list[float] = []
    for record in records:
        source = _load_state(record.path, device)
        tokens = _teacher_forcing_tokens(processor, record.text, device)
        losses.append(
            float(_transcript_loss(decoder, adapter(source), tokens, dtype).cpu())
        )
    return float(np.mean(losses))


@torch.inference_mode()
def _decode_validation(
    *,
    adapter: ShortStateAdapter,
    decoder,
    processor,
    records: list[FeatureRecord],
    device: torch.device,
    dtype: torch.dtype,
) -> list[dict[str, str]]:
    adapter.eval()
    results: list[dict[str, str]] = []
    for record in records:
        state = adapter(_load_state(record.path, device)).to(dtype)
        token_ids = _decode_whisper_official(
            model=decoder,
            encoder_state=state,
            max_new_tokens=128,
        )
        hypothesis = processor.batch_decode([token_ids], skip_special_tokens=True)[
            0
        ].strip()
        results.append(
            {
                "id": record.sample_id,
                "reference": record.text,
                "hypothesis": hypothesis,
            }
        )
    return results


@torch.inference_mode()
def _decode_local_holdout(
    adapter: ShortStateAdapter,
    decoder,
    processor,
    device: torch.device,
    dtype: torch.dtype,
) -> str:
    config = transformers.WhisperConfig.from_pretrained(MODEL_ID)
    encoder = ModifiedWhisperEncoder(config).to(device=device, dtype=dtype).eval()
    encoder.init_latency_mask(None, dtype=dtype)
    _load_prefixed_weights(
        encoder,
        ULTRAVOX_WEIGHTS,
        prefix="audio_tower.",
        device=device,
    )
    audio = _read_pcm16_wav(HOLDOUT_AUDIO)
    features = processor(
        audio,
        sampling_rate=16_000,
        padding="longest",
        max_length=len(audio),
        return_attention_mask=True,
        return_tensors="pt",
    )
    short = encoder(
        features.input_features.to(device=device, dtype=dtype),
        audio_len=torch.tensor([features.input_features.shape[-1]], device=device),
    ).last_hidden_state
    adapted = adapter(short.float()).to(dtype)
    token_ids = _decode_whisper_official(
        model=decoder,
        encoder_state=adapted,
        max_new_tokens=128,
    )
    return processor.batch_decode([token_ids], skip_special_tokens=True)[0].strip()


if __name__ == "__main__":
    main()
