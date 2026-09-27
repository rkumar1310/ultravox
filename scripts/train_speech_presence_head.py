"""Train a speech gate in two phases while preserving the ASR adapter."""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import shutil
import time

import numpy as np
import torch
import transformers
from jiwer import wer
from torch.nn.utils.rnn import pad_sequence

from scripts.dual_output_probe import _decode_whisper_official
from scripts.dual_output_probe import _load_whisper_decoder_model
from scripts.train_shared_whisper_adapter import MODEL_ID
from scripts.train_shared_whisper_adapter_librispeech import FEATURE_DIR
from scripts.train_shared_whisper_adapter_librispeech import WORK_DIR
from scripts.train_shared_whisper_adapter_librispeech import FeatureRecord
from scripts.train_shared_whisper_adapter_librispeech import _batched_transcript_loss
from scripts.train_shared_whisper_adapter_librispeech import _load_state
from scripts.train_shared_whisper_adapter_librispeech import _load_training_batch
from scripts.train_shared_whisper_adapter_librispeech import _normalize_for_wer
from scripts.train_shared_whisper_adapter_librispeech import _read_manifest
from ultravox.inference.shared_whisper_adapter import ShortStateAdapter
from ultravox.inference.shared_whisper_adapter import SpeechPresenceHead


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-checkpoint", type=pathlib.Path, required=True)
    parser.add_argument(
        "--head-train-manifest",
        type=pathlib.Path,
        default=FEATURE_DIR / "train-noise-v2-10000.jsonl",
    )
    parser.add_argument(
        "--joint-train-manifest",
        type=pathlib.Path,
        default=FEATURE_DIR / "train-noise-v2-20000.jsonl",
    )
    parser.add_argument("--head-train-samples", type=int, default=10_000)
    parser.add_argument("--joint-train-samples", type=int, default=10_000)
    parser.add_argument("--checkpoint-every-samples", type=int, default=1_000)
    parser.add_argument("--head-batch-size", type=int, default=50)
    parser.add_argument("--joint-batch-size", type=int, default=4)
    parser.add_argument("--head-learning-rate", type=float, default=3e-4)
    parser.add_argument("--joint-head-learning-rate", type=float, default=1e-4)
    parser.add_argument("--adapter-learning-rate", type=float, default=1e-5)
    parser.add_argument("--classification-loss-weight", type=float, default=0.5)
    parser.add_argument("--validation-decode-samples", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--resume-head-checkpoint",
        type=pathlib.Path,
        help="Use an already completed head-only checkpoint instead of retraining phase one.",
    )
    parser.add_argument(
        "--update-adapter",
        action="store_true",
        help=(
            "Experimentally update the ASR adapter in phase two. The safe default "
            "fine-tunes only the speech head."
        ),
    )
    args = parser.parse_args()

    if not args.adapter_checkpoint.exists():
        parser.error(f"missing adapter checkpoint: {args.adapter_checkpoint}")
    if (
        args.resume_head_checkpoint is not None
        and not args.resume_head_checkpoint.exists()
    ):
        parser.error(f"missing head checkpoint: {args.resume_head_checkpoint}")
    if args.checkpoint_every_samples <= 0:
        parser.error("checkpoint-every-samples must be positive")
    if args.head_train_samples % args.checkpoint_every_samples:
        parser.error("head training must end on a checkpoint boundary")
    if args.joint_train_samples % args.checkpoint_every_samples:
        parser.error("joint training must end on a checkpoint boundary")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    decoder_dtype = torch.float16 if device.type == "cuda" else torch.float32

    head_records = _required_records(args.head_train_manifest, args.head_train_samples)
    joint_records = _required_records(
        args.joint_train_manifest, args.joint_train_samples
    )
    clean_validation = _required_records(FEATURE_DIR / "validation.jsonl", 50)
    heavy_validation = _required_records(
        FEATURE_DIR / "validation-noise-v2-speech.jsonl", 50
    )
    empty_validation = _required_records(
        FEATURE_DIR / "validation-noise-v2-empty.jsonl", 50
    )
    print(
        json.dumps(
            {
                "device": str(device),
                "phaseOneSamples": len(head_records),
                "phaseTwoSamples": len(joint_records),
                "phaseOneSpeech": sum(bool(record.text) for record in head_records),
                "phaseOneNoSpeech": sum(not record.text for record in head_records),
                "phaseTwoSpeech": sum(bool(record.text) for record in joint_records),
                "phaseTwoNoSpeech": sum(not record.text for record in joint_records),
            }
        ),
        flush=True,
    )

    head = SpeechPresenceHead().to(device)
    head_dir = WORK_DIR / "speech-head-checkpoints"
    joint_dir = WORK_DIR / "joint-checkpoints"
    head_dir.mkdir(parents=True, exist_ok=True)
    joint_dir.mkdir(parents=True, exist_ok=True)
    phase_one_results = WORK_DIR / "speech-head-00000-10000-results.jsonl"
    joint_results = WORK_DIR / "joint-20000-30000-results.jsonl"
    phase_one_checkpoint = WORK_DIR / "speech-head-10000.pt"
    if args.resume_head_checkpoint is not None:
        head.load_state_dict(
            torch.load(
                args.resume_head_checkpoint,
                map_location=device,
                weights_only=True,
            )
        )
        if args.resume_head_checkpoint.resolve() != phase_one_checkpoint.resolve():
            shutil.copy2(args.resume_head_checkpoint, phase_one_checkpoint)
        print(
            f"stage=head_training_resumed checkpoint={phase_one_checkpoint}", flush=True
        )
    else:
        phase_one_results.unlink(missing_ok=True)
        phase_one_started = time.monotonic()
        print("stage=head_training_start", flush=True)
        head_optimizer = torch.optim.AdamW(
            head.parameters(), lr=args.head_learning_rate
        )
        class_weights = _class_weights(head_records, device)
        phase_one_order = torch.randperm(len(head_records)).tolist()
        for chunk_start in range(0, len(head_records), args.checkpoint_every_samples):
            indexes = phase_one_order[
                chunk_start : chunk_start + args.checkpoint_every_samples
            ]
            loss = _train_head_chunk(
                head=head,
                records=[head_records[index] for index in indexes],
                optimizer=head_optimizer,
                batch_size=args.head_batch_size,
                class_weights=class_weights,
                device=device,
            )
            trained = chunk_start + len(indexes)
            checkpoint = head_dir / f"speech-head-{trained:05d}.pt"
            torch.save(head.state_dict(), checkpoint)
            metrics = {
                "phase": "head-only",
                "trainedSamples": trained,
                "checkpoint": str(checkpoint),
                "trainBce": loss,
                **_classification_metrics(
                    head,
                    clean_validation,
                    heavy_validation,
                    empty_validation,
                    threshold=args.threshold,
                    device=device,
                ),
            }
            _append_jsonl(phase_one_results, metrics)
            print("checkpoint_result=" + json.dumps(metrics), flush=True)

        shutil.copy2(head_dir / "speech-head-10000.pt", phase_one_checkpoint)
        print(
            f"stage=head_training_complete elapsed_seconds="
            f"{time.monotonic() - phase_one_started:.1f}",
            flush=True,
        )

    if not args.update_adapter:
        _fine_tune_head_only(
            head=head,
            records=joint_records,
            clean_validation=clean_validation,
            heavy_validation=heavy_validation,
            empty_validation=empty_validation,
            checkpoint_dir=joint_dir,
            adapter_checkpoint=args.adapter_checkpoint,
            checkpoint_every_samples=args.checkpoint_every_samples,
            batch_size=args.head_batch_size,
            learning_rate=args.joint_head_learning_rate,
            threshold=args.threshold,
            seed=args.seed + 1,
            device=device,
        )
        return

    joint_results.unlink(missing_ok=True)

    processor = transformers.AutoProcessor.from_pretrained(MODEL_ID)
    decoder = _load_whisper_decoder_model(MODEL_ID, device, decoder_dtype).eval()
    for parameter in decoder.parameters():
        parameter.requires_grad_(False)
    adapter = ShortStateAdapter().to(device=device, dtype=torch.float32)
    adapter.load_state_dict(
        torch.load(args.adapter_checkpoint, map_location=device, weights_only=True)
    )
    joint_optimizer = torch.optim.AdamW(
        [
            {"params": adapter.parameters(), "lr": args.adapter_learning_rate},
            {"params": head.parameters(), "lr": args.joint_head_learning_rate},
        ]
    )
    joint_class_weights = _class_weights(joint_records, device)
    joint_order = torch.randperm(len(joint_records)).tolist()
    baseline = _joint_metrics(
        adapter=adapter,
        head=head,
        decoder=decoder,
        processor=processor,
        clean_records=clean_validation,
        heavy_records=heavy_validation,
        empty_records=empty_validation,
        decode_samples=args.validation_decode_samples,
        threshold=args.threshold,
        device=device,
        dtype=decoder_dtype,
    )
    baseline.update(
        {
            "phase": "joint-baseline",
            "adapterTrainedSamples": 20_000,
            "headTrainedSamples": 10_000,
            "adapterCheckpoint": str(args.adapter_checkpoint),
            "headCheckpoint": str(phase_one_checkpoint),
            "trainLoss": None,
        }
    )
    _append_jsonl(joint_results, baseline)
    print("checkpoint_result=" + json.dumps(baseline), flush=True)

    best_score = float(baseline["selectionScore"])
    best_joint_samples = 0
    best_adapter = WORK_DIR / "adapter-joint-best.pt"
    best_head = WORK_DIR / "speech-head-joint-best.pt"
    torch.save(adapter.state_dict(), best_adapter)
    torch.save(head.state_dict(), best_head)
    joint_started = time.monotonic()
    print("stage=joint_training_start", flush=True)
    for chunk_start in range(0, len(joint_records), args.checkpoint_every_samples):
        indexes = joint_order[chunk_start : chunk_start + args.checkpoint_every_samples]
        train_result = _train_joint_chunk(
            adapter=adapter,
            head=head,
            decoder=decoder,
            processor=processor,
            records=[joint_records[index] for index in indexes],
            optimizer=joint_optimizer,
            batch_size=args.joint_batch_size,
            class_weights=joint_class_weights,
            classification_loss_weight=args.classification_loss_weight,
            device=device,
            dtype=decoder_dtype,
        )
        joint_trained = chunk_start + len(indexes)
        adapter_total = 20_000 + joint_trained
        head_total = 10_000 + joint_trained
        adapter_checkpoint = joint_dir / f"adapter-{adapter_total:05d}.pt"
        head_checkpoint = joint_dir / f"speech-head-{head_total:05d}.pt"
        torch.save(adapter.state_dict(), adapter_checkpoint)
        torch.save(head.state_dict(), head_checkpoint)
        metrics = _joint_metrics(
            adapter=adapter,
            head=head,
            decoder=decoder,
            processor=processor,
            clean_records=clean_validation,
            heavy_records=heavy_validation,
            empty_records=empty_validation,
            decode_samples=args.validation_decode_samples,
            threshold=args.threshold,
            device=device,
            dtype=decoder_dtype,
        )
        metrics.update(
            {
                "phase": "joint",
                "adapterTrainedSamples": adapter_total,
                "headTrainedSamples": head_total,
                "adapterCheckpoint": str(adapter_checkpoint),
                "headCheckpoint": str(head_checkpoint),
                **train_result,
            }
        )
        if float(metrics["selectionScore"]) < best_score:
            best_score = float(metrics["selectionScore"])
            best_joint_samples = joint_trained
            shutil.copy2(adapter_checkpoint, best_adapter)
            shutil.copy2(head_checkpoint, best_head)
        _append_jsonl(joint_results, metrics)
        print("checkpoint_result=" + json.dumps(metrics), flush=True)

    latest_adapter = WORK_DIR / "adapter-joint-latest.pt"
    latest_head = WORK_DIR / "speech-head-joint-latest.pt"
    shutil.copy2(joint_dir / "adapter-30000.pt", latest_adapter)
    shutil.copy2(joint_dir / "speech-head-20000.pt", latest_head)
    print(
        json.dumps(
            {
                "phaseOneCheckpoint": str(phase_one_checkpoint),
                "latestAdapter": str(latest_adapter),
                "latestHead": str(latest_head),
                "bestAdapter": str(best_adapter),
                "bestHead": str(best_head),
                "bestJointSamples": best_joint_samples,
                "bestSelectionScore": best_score,
                "phaseOneResults": str(phase_one_results),
                "jointResults": str(joint_results),
                "jointElapsedSeconds": round(time.monotonic() - joint_started, 1),
            },
            indent=2,
        ),
        flush=True,
    )


def _fine_tune_head_only(
    *,
    head: SpeechPresenceHead,
    records: list[FeatureRecord],
    clean_validation: list[FeatureRecord],
    heavy_validation: list[FeatureRecord],
    empty_validation: list[FeatureRecord],
    checkpoint_dir: pathlib.Path,
    adapter_checkpoint: pathlib.Path,
    checkpoint_every_samples: int,
    batch_size: int,
    learning_rate: float,
    threshold: float,
    seed: int,
    device: torch.device,
) -> None:
    results_path = WORK_DIR / "speech-head-10000-20000-results.jsonl"
    results_path.unlink(missing_ok=True)
    optimizer = torch.optim.AdamW(head.parameters(), lr=learning_rate)
    class_weights = _class_weights(records, device)
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(records), generator=generator).tolist()
    best_score = float("inf")
    best_checkpoint = WORK_DIR / "speech-head-finetune-best.pt"
    started = time.monotonic()
    print("stage=head_finetune_start", flush=True)
    for chunk_start in range(0, len(records), checkpoint_every_samples):
        indexes = order[chunk_start : chunk_start + checkpoint_every_samples]
        loss = _train_head_chunk(
            head=head,
            records=[records[index] for index in indexes],
            optimizer=optimizer,
            batch_size=batch_size,
            class_weights=class_weights,
            device=device,
        )
        fine_tuned = chunk_start + len(indexes)
        total = 10_000 + fine_tuned
        checkpoint = checkpoint_dir / f"speech-head-{total:05d}.pt"
        torch.save(head.state_dict(), checkpoint)
        metrics = _classification_metrics(
            head,
            clean_validation,
            heavy_validation,
            empty_validation,
            threshold=threshold,
            device=device,
        )
        score = (
            1.0
            - float(metrics["cleanSpeechRecall"])
            + 1.0
            - float(metrics["heavySpeechRecall"])
            + float(metrics["noSpeechFalsePositiveRate"])
            + float(metrics["noSpeechProbabilityMean"])
            + 1.0
            - float(metrics["cleanSpeechProbabilityMean"])
            + 1.0
            - float(metrics["heavySpeechProbabilityMean"])
        )
        result = {
            "phase": "head-finetune",
            "fineTunedSamples": fine_tuned,
            "headTrainedSamples": total,
            "headCheckpoint": str(checkpoint),
            "adapterCheckpoint": str(adapter_checkpoint),
            "adapterUpdated": False,
            "trainBce": loss,
            "classificationSelectionScore": score,
            **metrics,
        }
        if score < best_score:
            best_score = score
            shutil.copy2(checkpoint, best_checkpoint)
        _append_jsonl(results_path, result)
        print("checkpoint_result=" + json.dumps(result), flush=True)

    latest_checkpoint = WORK_DIR / "speech-head-finetune-latest.pt"
    shutil.copy2(checkpoint_dir / "speech-head-20000.pt", latest_checkpoint)
    print(
        json.dumps(
            {
                "phase": "head-finetune-complete",
                "fineTunedSamples": len(records),
                "headTrainedSamples": 20_000,
                "latestHead": str(latest_checkpoint),
                "bestHead": str(best_checkpoint),
                "adapterCheckpoint": str(adapter_checkpoint),
                "adapterUpdated": False,
                "results": str(results_path),
                "elapsedSeconds": round(time.monotonic() - started, 1),
            },
            indent=2,
        ),
        flush=True,
    )


def _required_records(path: pathlib.Path, count: int) -> list[FeatureRecord]:
    records = _read_manifest(path)
    if len(records) < count:
        raise RuntimeError(f"{path} has {len(records)} records; expected {count}")
    selected = records[:count]
    missing = [str(record.path) for record in selected if not record.path.exists()]
    if missing:
        raise FileNotFoundError(f"missing feature state: {missing[0]}")
    return selected


def _load_presence_batch(
    records: list[FeatureRecord], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    states = [_load_state(record.path, device).squeeze(0) for record in records]
    lengths = torch.tensor([state.shape[0] for state in states], device=device)
    source = pad_sequence(states, batch_first=True)
    mask = torch.arange(source.shape[1], device=device)[None, :] < lengths[:, None]
    targets = torch.tensor(
        [bool(record.text) for record in records],
        device=device,
        dtype=torch.float32,
    )
    return source, mask, targets


def _class_weights(records: list[FeatureRecord], device: torch.device) -> torch.Tensor:
    positive_fraction = sum(bool(record.text) for record in records) / len(records)
    negative_fraction = 1.0 - positive_fraction
    return torch.tensor(
        [0.5 / negative_fraction, 0.5 / positive_fraction],
        device=device,
        dtype=torch.float32,
    )


def _weighted_binary_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    class_weights: torch.Tensor,
) -> torch.Tensor:
    weights = torch.where(targets.bool(), class_weights[1], class_weights[0])
    return torch.nn.functional.binary_cross_entropy_with_logits(
        logits, targets, weight=weights
    )


def _train_head_chunk(
    *,
    head: SpeechPresenceHead,
    records: list[FeatureRecord],
    optimizer,
    batch_size: int,
    class_weights: torch.Tensor,
    device: torch.device,
) -> float:
    head.train()
    losses: list[float] = []
    ordered = sorted(records, key=lambda record: record.duration_seconds)
    for offset in range(0, len(ordered), batch_size):
        source, mask, targets = _load_presence_batch(
            ordered[offset : offset + batch_size], device
        )
        optimizer.zero_grad(set_to_none=True)
        loss = _weighted_binary_loss(head(source, mask), targets, class_weights)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def _train_joint_chunk(
    *,
    adapter: ShortStateAdapter,
    head: SpeechPresenceHead,
    decoder,
    processor,
    records: list[FeatureRecord],
    optimizer,
    batch_size: int,
    class_weights: torch.Tensor,
    classification_loss_weight: float,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, float]:
    adapter.train()
    head.train()
    total_losses: list[float] = []
    asr_losses: list[float] = []
    classification_losses: list[float] = []
    ordered = sorted(records, key=lambda record: record.duration_seconds)
    for offset in range(0, len(ordered), batch_size):
        batch = ordered[offset : offset + batch_size]
        source, source_mask, tokens, labels = _load_training_batch(
            batch, processor, device
        )
        targets = torch.tensor(
            [bool(record.text) for record in batch],
            device=device,
            dtype=torch.float32,
        )
        optimizer.zero_grad(set_to_none=True)
        classification_loss = _weighted_binary_loss(
            head(source, source_mask), targets, class_weights
        )
        speech = targets.bool()
        if bool(speech.any()):
            asr_loss = _batched_transcript_loss(
                decoder,
                adapter(source[speech], source_mask[speech]),
                tokens[speech],
                labels[speech],
                dtype,
            )
        else:
            asr_loss = classification_loss.new_zeros(())
        loss = asr_loss + classification_loss_weight * classification_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_([*adapter.parameters(), *head.parameters()], 1.0)
        optimizer.step()
        total_losses.append(float(loss.detach().cpu()))
        asr_losses.append(float(asr_loss.detach().cpu()))
        classification_losses.append(float(classification_loss.detach().cpu()))
    return {
        "trainLoss": float(np.mean(total_losses)),
        "trainAsrCe": float(np.mean(asr_losses)),
        "trainSpeechBce": float(np.mean(classification_losses)),
    }


@torch.inference_mode()
def _probabilities(
    head: SpeechPresenceHead,
    records: list[FeatureRecord],
    device: torch.device,
    batch_size: int = 50,
) -> list[float]:
    head.eval()
    probabilities: list[float] = []
    for offset in range(0, len(records), batch_size):
        source, mask, _ = _load_presence_batch(
            records[offset : offset + batch_size], device
        )
        probabilities.extend(head(source, mask).sigmoid().cpu().tolist())
    return probabilities


def _classification_metrics(
    head: SpeechPresenceHead,
    clean_records: list[FeatureRecord],
    heavy_records: list[FeatureRecord],
    empty_records: list[FeatureRecord],
    *,
    threshold: float,
    device: torch.device,
) -> dict[str, float]:
    clean = _probabilities(head, clean_records, device)
    heavy = _probabilities(head, heavy_records, device)
    empty = _probabilities(head, empty_records, device)
    clean_recall = sum(value >= threshold for value in clean) / len(clean)
    heavy_recall = sum(value >= threshold for value in heavy) / len(heavy)
    false_positive_rate = sum(value >= threshold for value in empty) / len(empty)
    return {
        "threshold": threshold,
        "cleanSpeechRecall": clean_recall,
        "heavySpeechRecall": heavy_recall,
        "noSpeechFalsePositiveRate": false_positive_rate,
        "balancedAccuracy": (clean_recall + heavy_recall + (1.0 - false_positive_rate))
        / 3.0,
        "cleanSpeechProbabilityMean": float(np.mean(clean)),
        "heavySpeechProbabilityMean": float(np.mean(heavy)),
        "noSpeechProbabilityMean": float(np.mean(empty)),
    }


def _joint_metrics(
    *,
    adapter: ShortStateAdapter,
    head: SpeechPresenceHead,
    decoder,
    processor,
    clean_records: list[FeatureRecord],
    heavy_records: list[FeatureRecord],
    empty_records: list[FeatureRecord],
    decode_samples: int,
    threshold: float,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, object]:
    classification = _classification_metrics(
        head,
        clean_records,
        heavy_records,
        empty_records,
        threshold=threshold,
        device=device,
    )
    clean_decoded = _gated_decode(
        adapter,
        head,
        decoder,
        processor,
        clean_records[:decode_samples],
        threshold,
        device,
        dtype,
    )
    heavy_decoded = _gated_decode(
        adapter,
        head,
        decoder,
        processor,
        heavy_records[:decode_samples],
        threshold,
        device,
        dtype,
    )
    clean_wer = _wer(clean_decoded)
    heavy_wer = _wer(heavy_decoded)
    selection_score = (
        clean_wer + heavy_wer + float(classification["noSpeechFalsePositiveRate"])
    )
    return {
        **classification,
        "cleanValidationWer": clean_wer,
        "heavyNoiseValidationWer": heavy_wer,
        "selectionScore": selection_score,
        "cleanExamples": clean_decoded[:3],
        "heavyExamples": heavy_decoded[:3],
    }


@torch.inference_mode()
def _gated_decode(
    adapter: ShortStateAdapter,
    head: SpeechPresenceHead,
    decoder,
    processor,
    records: list[FeatureRecord],
    threshold: float,
    device: torch.device,
    dtype: torch.dtype,
) -> list[dict[str, object]]:
    adapter.eval()
    head.eval()
    results: list[dict[str, object]] = []
    for record in records:
        source = _load_state(record.path, device)
        mask = torch.ones(source.shape[:2], device=device, dtype=torch.bool)
        probability = float(head(source, mask).sigmoid().item())
        hypothesis = ""
        if probability >= threshold:
            state = adapter(source, mask).to(dtype)
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
                "speechProbability": probability,
            }
        )
    return results


def _wer(decoded: list[dict[str, object]]) -> float:
    return wer(
        [_normalize_for_wer(str(item["reference"])) for item in decoded],
        [_normalize_for_wer(str(item["hypothesis"])) for item in decoded],
    )


def _append_jsonl(path: pathlib.Path, item: dict[str, object]) -> None:
    with path.open("a") as destination:
        destination.write(json.dumps(item) + "\n")
        destination.flush()


if __name__ == "__main__":
    main()
