"""Continue shared-state Whisper adapter training entirely on the local machine.

The runner keeps one checkpoint interval of Ultravox states in memory at a
time.  This avoids retaining a 20,000-example feature cache on disk while
preserving a checkpoint, sample manifest, and WER result every 1,000 samples.
"""

from __future__ import annotations

import argparse
import gc
import io
import json
import pathlib
import shutil
import time
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import soundfile as sf
import torch
import transformers
from datasets import load_dataset
from jiwer import wer
from torch.nn.utils.rnn import pad_sequence

from scripts.dual_output_probe import _decode_whisper_official
from scripts.dual_output_probe import _load_whisper_decoder_model
from scripts.train_shared_whisper_adapter import MODEL_ID
from scripts.train_shared_whisper_adapter import ULTRAVOX_WEIGHTS
from scripts.train_shared_whisper_adapter import _load_prefixed_weights
from scripts.train_shared_whisper_adapter import _teacher_forcing_tokens
from scripts.train_shared_whisper_adapter_librispeech import DATASET_CONFIG
from scripts.train_shared_whisper_adapter_librispeech import DATASET_ID
from scripts.train_shared_whisper_adapter_librispeech import _batched_transcript_loss
from scripts.train_shared_whisper_adapter_librispeech import _normalize_for_wer
from ultravox.inference.shared_whisper_adapter import ShortStateAdapter
from ultravox.model.ultravox_model import ModifiedWhisperEncoder

WORK_DIR = pathlib.Path(".model-cache/shared-whisper-adapter-librispeech")
CHECKPOINT_DIR = WORK_DIR / "checkpoints"
CANONICAL_CHECKPOINT = WORK_DIR / "adapter-asr.pt"
BASE_MANIFEST = WORK_DIR / "modal-train-10000.jsonl"
VALIDATION_MANIFEST = WORK_DIR / "modal-validation-50.jsonl"
CONTINUATION_MANIFEST = WORK_DIR / "local-continuation-samples.jsonl"
RESULTS_PATH = WORK_DIR / "local-10000-20000-results.jsonl"
WORKING_BLOCK_DIR = WORK_DIR / "local-working-block"
WORKING_BLOCK_MANIFEST = WORKING_BLOCK_DIR / "manifest.jsonl"


@dataclass(frozen=True)
class MemoryFeature:
    sample_id: str
    text: str
    duration_seconds: float
    state: torch.Tensor


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-samples", type=int, default=10_000)
    parser.add_argument("--target-samples", type=int, default=20_000)
    parser.add_argument("--checkpoint-every-samples", type=int, default=1_000)
    parser.add_argument("--encode-batch-size", type=int, default=8)
    parser.add_argument("--training-microbatch-size", type=int, default=4)
    parser.add_argument("--validation-samples", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-duration-seconds", type=float, default=29.5)
    args = parser.parse_args()

    if args.target_samples <= args.start_samples:
        parser.error("target-samples must be greater than start-samples")
    if args.checkpoint_every_samples <= 0:
        parser.error("checkpoint-every-samples must be positive")
    if (args.target_samples - args.start_samples) % args.checkpoint_every_samples:
        parser.error("the requested range must divide into complete checkpoints")
    if not BASE_MANIFEST.exists() or not VALIDATION_MANIFEST.exists():
        parser.error(
            "the downloaded 10k training and validation manifests are required"
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        encoder_dtype = torch.float16
        decoder_dtype = torch.float32
    else:
        device = torch.device("cpu")
        encoder_dtype = torch.float32
        decoder_dtype = torch.float32
    print(
        f"device={device} encoder_dtype={encoder_dtype} "
        f"decoder_dtype={decoder_dtype} mode=local-only",
        flush=True,
    )

    completed_samples = _read_jsonl(CONTINUATION_MANIFEST)
    completed_count = len(completed_samples)
    current_total = args.start_samples + completed_count
    if completed_count % args.checkpoint_every_samples:
        raise RuntimeError("continuation manifest ends between checkpoints")
    if current_total > args.target_samples:
        raise RuntimeError("continuation manifest is beyond the requested target")
    if current_total == args.target_samples:
        print(f"already_complete={current_total}", flush=True)
        return

    start_checkpoint = CHECKPOINT_DIR / f"adapter-{current_total:04d}.pt"
    if not start_checkpoint.exists():
        if current_total == args.start_samples and CANONICAL_CHECKPOINT.exists():
            start_checkpoint = CANONICAL_CHECKPOINT
        else:
            raise FileNotFoundError(f"missing resume checkpoint: {start_checkpoint}")

    processor = transformers.AutoProcessor.from_pretrained(MODEL_ID)
    resume_checkpoint = start_checkpoint
    resume_optimizer = CHECKPOINT_DIR / f"optimizer-{current_total:04d}.pt"
    validation_ids = [
        str(item["sampleId"])
        for item in _read_jsonl(VALIDATION_MANIFEST)[: args.validation_samples]
    ]
    validation: list[MemoryFeature] | None = None

    excluded_ids = {
        str(item["sampleId"])
        for item in [*_read_jsonl(BASE_MANIFEST), *completed_samples]
    }
    cached_rows = _read_jsonl(WORKING_BLOCK_MANIFEST)
    if cached_rows and all(
        int(item.get("blockStart", -1)) == current_total for item in cached_rows
    ):
        excluded_ids.update(str(item["sampleId"]) for item in cached_rows)
    rows = _fresh_training_rows(
        excluded_ids=excluded_ids,
        max_duration_seconds=args.max_duration_seconds,
    )
    remaining = args.target_samples - current_total
    while remaining:
        block_size = min(args.checkpoint_every_samples, remaining)
        block_start = current_total
        print(
            f"stage=local_feature_encoding_start block={block_start}-{block_start + block_size}",
            flush=True,
        )
        features = _load_working_block(block_start, block_size)
        encoder = None
        if validation is None or features is None:
            encoder = _load_audio_encoder(device, encoder_dtype)
        if validation is None:
            validation = _prepare_validation(
                encoder=encoder,
                processor=processor,
                expected_ids=validation_ids,
                encode_batch_size=args.encode_batch_size,
                device=device,
                dtype=encoder_dtype,
            )
        if features is None:
            features = _prepare_training_block(
                rows=rows,
                count=block_size,
                encoder=encoder,
                processor=processor,
                encode_batch_size=args.encode_batch_size,
                device=device,
                dtype=encoder_dtype,
            )
            print(
                f"stage=local_feature_encoding_complete count={len(features)}",
                flush=True,
            )
            _save_working_block(features, block_start)
        else:
            print(
                f"stage=local_feature_cache_reused count={len(features)}",
                flush=True,
            )
        if encoder is not None:
            del encoder
            gc.collect()
            if device.type == "mps":
                torch.mps.synchronize()
                torch.mps.empty_cache()

        print(f"loading_checkpoint={resume_checkpoint}", flush=True)
        decoder = _load_whisper_decoder_model(MODEL_ID, device, decoder_dtype).eval()
        for parameter in decoder.parameters():
            parameter.requires_grad_(False)
        adapter = ShortStateAdapter().to(device=device, dtype=torch.float32)
        adapter.load_state_dict(
            torch.load(resume_checkpoint, map_location=device, weights_only=True)
        )
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.learning_rate)
        if resume_optimizer.exists():
            optimizer.load_state_dict(
                torch.load(resume_optimizer, map_location=device, weights_only=True)
            )
            print(f"loading_optimizer={resume_optimizer}", flush=True)
        train_ce, samples_per_second, sanitized_gradient_elements = _train_block(
            features=features,
            adapter=adapter,
            decoder=decoder,
            processor=processor,
            optimizer=optimizer,
            microbatch_size=args.training_microbatch_size,
            device=device,
            dtype=decoder_dtype,
        )
        current_total += len(features)
        validation_ce = _validation_loss(
            records=validation,
            adapter=adapter,
            decoder=decoder,
            processor=processor,
            microbatch_size=args.training_microbatch_size,
            device=device,
            dtype=decoder_dtype,
        )
        decoded = _decode_validation(
            records=validation,
            adapter=adapter,
            decoder=decoder,
            processor=processor,
            device=device,
            dtype=decoder_dtype,
        )
        checkpoint_wer = wer(
            [_normalize_for_wer(item["reference"]) for item in decoded],
            [_normalize_for_wer(item["hypothesis"]) for item in decoded],
        )
        checkpoint_path = CHECKPOINT_DIR / f"adapter-{current_total:04d}.pt"
        optimizer_path = CHECKPOINT_DIR / f"optimizer-{current_total:04d}.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(adapter.state_dict(), checkpoint_path)
        torch.save(optimizer.state_dict(), optimizer_path)
        torch.save(adapter.state_dict(), CANONICAL_CHECKPOINT)
        _append_jsonl(
            CONTINUATION_MANIFEST,
            [
                {
                    "sampleId": record.sample_id,
                    "text": record.text,
                    "durationSeconds": record.duration_seconds,
                }
                for record in features
            ],
        )
        result = {
            "trainedSamples": current_total,
            "checkpoint": str(checkpoint_path),
            "trainCe": train_ce,
            "validationCe": validation_ce,
            "validationWer": checkpoint_wer,
            "decodedValidationSamples": len(decoded),
            "trainingSamplesPerSecond": samples_per_second,
            "sanitizedGradientElements": sanitized_gradient_elements,
            "examples": decoded[:5],
        }
        _append_jsonl(RESULTS_PATH, [result])
        print("checkpoint_result=" + json.dumps(result), flush=True)
        _clear_working_block()
        resume_checkpoint = checkpoint_path
        resume_optimizer = optimizer_path
        remaining = args.target_samples - current_total
        del adapter, decoder, optimizer, features
        gc.collect()
        if device.type == "mps":
            torch.mps.synchronize()
            torch.mps.empty_cache()

    print(f"training_complete={current_total}", flush=True)


def _save_working_block(features: list[MemoryFeature], block_start: int) -> None:
    if WORKING_BLOCK_DIR.exists():
        shutil.rmtree(WORKING_BLOCK_DIR)
    WORKING_BLOCK_DIR.mkdir(parents=True)
    rows: list[dict[str, object]] = []
    for index, record in enumerate(features):
        tensor_path = WORKING_BLOCK_DIR / f"{index:04d}.pt"
        torch.save(record.state, tensor_path)
        rows.append(
            {
                "blockStart": block_start,
                "sampleId": record.sample_id,
                "text": record.text,
                "durationSeconds": record.duration_seconds,
                "tensorPath": str(tensor_path),
            }
        )
    temporary = WORKING_BLOCK_MANIFEST.with_suffix(".tmp")
    with temporary.open("w") as destination:
        for row in rows:
            destination.write(json.dumps(row) + "\n")
    temporary.replace(WORKING_BLOCK_MANIFEST)
    print(
        f"stage=local_feature_cache_saved count={len(features)} "
        f"path={WORKING_BLOCK_DIR}",
        flush=True,
    )


def _load_working_block(
    block_start: int, expected_count: int
) -> list[MemoryFeature] | None:
    rows = _read_jsonl(WORKING_BLOCK_MANIFEST)
    if not rows:
        return None
    if len(rows) != expected_count or any(
        int(row.get("blockStart", -1)) != block_start for row in rows
    ):
        return None
    records: list[MemoryFeature] = []
    for row in rows:
        tensor_path = pathlib.Path(str(row["tensorPath"]))
        if not tensor_path.exists():
            return None
        state = torch.load(tensor_path, map_location="cpu", weights_only=True)
        if not bool(torch.isfinite(state).all()):
            raise RuntimeError(f"non-finite cached state: {tensor_path}")
        records.append(
            MemoryFeature(
                sample_id=str(row["sampleId"]),
                text=str(row["text"]),
                duration_seconds=float(row["durationSeconds"]),
                state=state,
            )
        )
    return records


def _clear_working_block() -> None:
    if WORKING_BLOCK_DIR.exists():
        shutil.rmtree(WORKING_BLOCK_DIR)


def _load_audio_encoder(
    device: torch.device, dtype: torch.dtype
) -> ModifiedWhisperEncoder:
    config = transformers.WhisperConfig.from_pretrained(MODEL_ID)
    encoder = ModifiedWhisperEncoder(config).to(device=device, dtype=dtype).eval()
    encoder.init_latency_mask(None, dtype=dtype)
    _load_prefixed_weights(
        encoder,
        ULTRAVOX_WEIGHTS,
        prefix="audio_tower.",
        device=device,
    )
    return encoder


def _prepare_validation(
    *,
    encoder: ModifiedWhisperEncoder,
    processor,
    expected_ids: list[str],
    encode_batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> list[MemoryFeature]:
    expected = set(expected_ids)
    ordered: dict[str, MemoryFeature] = {}
    dataset = load_dataset(
        DATASET_ID, DATASET_CONFIG, split="validation", streaming=True
    ).decode(False)
    pending: list[tuple[str, str, np.ndarray, float]] = []
    for row in dataset:
        sample_id = str(row["id"])
        if sample_id not in expected:
            continue
        audio, sample_rate = _decode_audio(row["audio"])
        pending.append((sample_id, str(row["text"]), audio, len(audio) / sample_rate))
        if len(pending) >= encode_batch_size:
            for record in _encode_pending(encoder, processor, pending, device, dtype):
                ordered[record.sample_id] = record
            pending.clear()
        if len(ordered) + len(pending) == len(expected_ids):
            break
    if pending:
        for record in _encode_pending(encoder, processor, pending, device, dtype):
            ordered[record.sample_id] = record
    if set(ordered) != expected:
        raise RuntimeError("could not recover the exact validation set")
    print(f"validation_features_ready={len(ordered)}", flush=True)
    return [ordered[sample_id] for sample_id in expected_ids]


def _fresh_training_rows(
    *, excluded_ids: set[str], max_duration_seconds: float
) -> Iterator[tuple[str, str, np.ndarray, float]]:
    dataset = load_dataset(
        DATASET_ID, DATASET_CONFIG, split="train.100", streaming=True
    ).decode(False)
    for row in dataset:
        sample_id = str(row["id"])
        if sample_id in excluded_ids or not str(row["text"]).strip():
            continue
        audio, sample_rate = _decode_audio(row["audio"])
        duration_seconds = len(audio) / sample_rate
        if duration_seconds > max_duration_seconds:
            continue
        excluded_ids.add(sample_id)
        yield sample_id, str(row["text"]), audio, duration_seconds


def _prepare_training_block(
    *,
    rows: Iterator[tuple[str, str, np.ndarray, float]],
    count: int,
    encoder: ModifiedWhisperEncoder,
    processor,
    encode_batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> list[MemoryFeature]:
    records: list[MemoryFeature] = []
    pending: list[tuple[str, str, np.ndarray, float]] = []
    started = time.monotonic()
    while len(records) + len(pending) < count:
        pending.append(next(rows))
        if len(pending) >= encode_batch_size:
            records.extend(_encode_pending(encoder, processor, pending, device, dtype))
            pending.clear()
            if len(records) % 100 == 0:
                elapsed = time.monotonic() - started
                print(
                    f"local_features={len(records)}/{count} "
                    f"samples_per_second={len(records) / elapsed:.3f}",
                    flush=True,
                )
    if pending:
        records.extend(_encode_pending(encoder, processor, pending, device, dtype))
    return records


@torch.inference_mode()
def _encode_pending(
    encoder: ModifiedWhisperEncoder,
    processor,
    pending: list[tuple[str, str, np.ndarray, float]],
    device: torch.device,
    dtype: torch.dtype,
) -> list[MemoryFeature]:
    audio_arrays = [item[2] for item in pending]
    features = processor(
        audio_arrays,
        sampling_rate=16_000,
        padding="max_length",
        truncation=True,
        max_length=30 * 16_000,
        return_attention_mask=True,
        return_tensors="pt",
    )
    audio_lengths = features.attention_mask.sum(dim=-1).to(device=device)
    states = encoder(
        features.input_features.to(device=device, dtype=dtype),
        audio_len=audio_lengths,
    ).last_hidden_state
    state_lengths = encoder._get_feat_extract_output_lengths(audio_lengths)
    result = [
        MemoryFeature(
            sample_id=sample_id,
            text=text,
            duration_seconds=duration_seconds,
            state=states[index, : int(state_lengths[index])].cpu().half(),
        )
        for index, (sample_id, text, _, duration_seconds) in enumerate(pending)
    ]
    for record in result:
        if not bool(torch.isfinite(record.state).all()):
            raise RuntimeError(f"non-finite encoded state: {record.sample_id}")
    del states, state_lengths, features, audio_lengths, audio_arrays
    gc.collect()
    if device.type == "mps":
        torch.mps.synchronize()
        torch.mps.empty_cache()
    return result


def _train_block(
    *,
    features: list[MemoryFeature],
    adapter: ShortStateAdapter,
    decoder,
    processor,
    optimizer,
    microbatch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[float, float, int]:
    adapter.train()
    indexes = sorted(
        range(len(features)),
        key=lambda index: (
            features[index].duration_seconds,
            len(features[index].text),
        ),
    )
    batches = [
        indexes[offset : offset + microbatch_size]
        for offset in range(0, len(indexes), microbatch_size)
    ]
    order = torch.randperm(len(batches)).tolist()
    losses: list[torch.Tensor] = []
    processed = 0
    sanitized_gradient_elements = 0
    started = time.monotonic()
    for batch_index in order:
        records = [features[index] for index in batches[batch_index]]
        source, source_mask, tokens, labels = _load_batch(records, processor, device)
        optimizer.zero_grad(set_to_none=True)
        loss = _batched_transcript_loss(
            decoder, adapter(source, source_mask), tokens, labels, dtype
        )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(
                "non-finite training loss for samples "
                + ",".join(record.sample_id for record in records)
            )
        loss.backward()
        grad_norm, sanitized = _sanitize_and_clip_grad_norm_(
            adapter.parameters(), max_norm=1.0
        )
        sanitized_gradient_elements += sanitized
        if sanitized:
            print(
                f"sanitized_gradient_elements={sanitized} "
                "samples=" + ",".join(record.sample_id for record in records),
                flush=True,
            )
        optimizer.step()
        losses.append(loss.detach().cpu())
        processed += len(records)
        grad_norm_value = float(grad_norm.cpu())
        del source, source_mask, tokens, labels, loss, grad_norm
        if device.type == "mps" and processed % 100 == 0:
            torch.mps.synchronize()
            torch.mps.empty_cache()
        if processed % 100 == 0 or processed == len(features):
            elapsed = time.monotonic() - started
            recent = float(torch.stack(losses[-25:]).mean())
            print(
                f"local_training={processed}/{len(features)} "
                f"train_ce={recent:.5f} "
                f"grad_norm={grad_norm_value:.3f} "
                f"sanitized_total={sanitized_gradient_elements} "
                f"samples_per_second={processed / elapsed:.3f}",
                flush=True,
            )
    elapsed = time.monotonic() - started
    return (
        float(torch.stack(losses).mean()),
        len(features) / elapsed,
        sanitized_gradient_elements,
    )


def _sanitize_and_clip_grad_norm_(
    parameters, *, max_norm: float, reduction_scale: float = 1024.0
) -> tuple[torch.Tensor, int]:
    gradients = [
        parameter.grad for parameter in parameters if parameter.grad is not None
    ]
    if not gradients:
        raise RuntimeError("adapter backward pass produced no gradients")
    sanitized = 0
    for gradient in gradients:
        invalid = ~torch.isfinite(gradient)
        invalid_count = int(invalid.sum().cpu())
        if invalid_count:
            gradient.masked_fill_(invalid, 0.0)
            sanitized += invalid_count

    # torch.nn.utils.clip_grad_norm_ can overflow its MPS reduction even when
    # every individual gradient is finite.  Rescaling before squaring avoids
    # that reduction bug while preserving the actual norm and clip factor.
    scaled_sum = torch.zeros((), device=gradients[0].device, dtype=torch.float32)
    for gradient in gradients:
        scaled_sum.add_((gradient.float() / reduction_scale).square().sum())
    total_norm = scaled_sum.sqrt() * reduction_scale
    if not bool(torch.isfinite(total_norm)):
        raise RuntimeError("gradient norm remained non-finite after sanitization")
    clip_coefficient = (max_norm / (total_norm + 1e-6)).clamp(max=1.0)
    for gradient in gradients:
        gradient.mul_(clip_coefficient)
    return total_norm, sanitized


def _load_batch(
    records: list[MemoryFeature], processor, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    states = [record.state.float() for record in records]
    lengths = torch.tensor([state.shape[0] for state in states], device=device)
    source = pad_sequence(states, batch_first=True).to(device)
    source_mask = (
        torch.arange(source.shape[1], device=device)[None, :] < lengths[:, None]
    )
    source_bucket = _next_bucket(
        source.shape[1], (128, 192, 256, 384, 512, 768, 1024, 1280, 1500)
    )
    if source.shape[1] < source_bucket:
        padding = source_bucket - source.shape[1]
        source = torch.nn.functional.pad(source, (0, 0, 0, padding))
        source_mask = torch.nn.functional.pad(source_mask, (0, padding), value=False)
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
    token_bucket = _next_bucket(
        tokens.shape[1], (16, 24, 32, 48, 64, 96, 128, 192, 256, 448)
    )
    if tokens.shape[1] < token_bucket:
        tokens = torch.nn.functional.pad(
            tokens,
            (0, token_bucket - tokens.shape[1]),
            value=processor.tokenizer.eos_token_id,
        )
    labels = tokens[:, 1:].clone()
    for index, length in enumerate(token_lengths):
        labels[index, length - 1 :] = -100
    return source, source_mask, tokens, labels


def _next_bucket(length: int, buckets: tuple[int, ...]) -> int:
    for bucket in buckets:
        if length <= bucket:
            return bucket
    raise ValueError(f"length {length} exceeds largest bucket {buckets[-1]}")


@torch.no_grad()
def _validation_loss(
    *,
    records: list[MemoryFeature],
    adapter: ShortStateAdapter,
    decoder,
    processor,
    microbatch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> float:
    adapter.eval()
    losses: list[float] = []
    for offset in range(0, len(records), microbatch_size):
        source, source_mask, tokens, labels = _load_batch(
            records[offset : offset + microbatch_size], processor, device
        )
        losses.append(
            float(
                _batched_transcript_loss(
                    decoder, adapter(source, source_mask), tokens, labels, dtype
                ).cpu()
            )
        )
    return float(np.mean(losses))


@torch.inference_mode()
def _decode_validation(
    *,
    records: list[MemoryFeature],
    adapter: ShortStateAdapter,
    decoder,
    processor,
    device: torch.device,
    dtype: torch.dtype,
) -> list[dict[str, str]]:
    adapter.eval()
    decoded: list[dict[str, str]] = []
    for record in records:
        state = adapter(record.state.float().unsqueeze(0).to(device)).to(dtype)
        token_ids = _decode_whisper_official(
            model=decoder,
            encoder_state=state,
            max_new_tokens=128,
        )
        hypothesis = processor.batch_decode([token_ids], skip_special_tokens=True)[
            0
        ].strip()
        decoded.append(
            {
                "id": record.sample_id,
                "reference": record.text,
                "hypothesis": hypothesis,
            }
        )
    return decoded


def _decode_audio(audio: dict[str, object]) -> tuple[np.ndarray, int]:
    source = (
        io.BytesIO(audio["bytes"])
        if audio.get("bytes") is not None
        else audio.get("path")
    )
    if source is None:
        raise ValueError("audio row has neither bytes nor path")
    samples, sample_rate = sf.read(source, dtype="float32", always_2d=False)
    if samples.ndim == 2:
        samples = samples.mean(axis=1)
    if sample_rate != 16_000:
        raise ValueError(f"expected 16 kHz audio, got {sample_rate}")
    return samples, sample_rate


def _read_jsonl(path: pathlib.Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open() as source:
        return [json.loads(line) for line in source if line.strip()]


def _append_jsonl(path: pathlib.Path, items: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as destination:
        for item in items:
            destination.write(json.dumps(item) + "\n")
        destination.flush()


if __name__ == "__main__":
    main()
