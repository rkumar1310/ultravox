"""Train a small bridge from short Ultravox states to stock Whisper states."""

from __future__ import annotations

import argparse
import gc
import json
import pathlib
import subprocess

import numpy as np
import torch
import torch.nn as nn
import transformers
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from transformers.models.whisper.modeling_whisper import WhisperEncoder

from scripts.dual_output_probe import _decode_whisper_official
from scripts.dual_output_probe import _load_whisper_decoder_model
from scripts.dual_output_probe import _read_pcm16_wav
from ultravox.inference.shared_whisper_adapter import ShortStateAdapter
from ultravox.model.ultravox_model import ModifiedWhisperEncoder

MODEL_ID = "openai/whisper-large-v3-turbo"
ULTRAVOX_WEIGHTS = pathlib.Path(".model-cache/ultravox-v05/model.safetensors")
WORK_DIR = pathlib.Path(".model-cache/shared-whisper-adapter")
CHECKPOINT = WORK_DIR / "adapter.pt"
ASR_CHECKPOINT = WORK_DIR / "adapter-asr.pt"
HOLDOUT_AUDIO = pathlib.Path(".model-cache/dual-output.wav")
EXPECTED_HOLDOUT = (
    "My project codename is blue bicycle seven. Please schedule the launch "
    "for Tuesday at nine in the morning."
)

TRAIN_SENTENCES = [
    "Please reserve a conference room for Thursday afternoon.",
    "The security code is amber river twenty four.",
    "Move tomorrow's planning call to eleven in the morning.",
    "My delivery reference is copper lantern eight.",
    "Cancel the dentist appointment scheduled for Friday.",
    "The customer requested a refund for the damaged package.",
    "Remind me to telephone Morgan after lunch.",
    "Book a table for four people at seven thirty tonight.",
    "The experiment passed all seventeen validation checks.",
    "Schedule the product demonstration for next Wednesday.",
    "Our internal project is called silver compass nine.",
    "Send the revised contract before five in the evening.",
    "The train departs from platform six at eight fifteen.",
    "Add a follow up meeting to Monday's calendar.",
    "The replacement part number is delta forty two.",
    "Please postpone the launch until the final review is complete.",
    "Call the operations team when the shipment arrives.",
    "The account balance was updated earlier this morning.",
]
VOICES = ["Daniel", "Flo (English (US))", "Aman"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--rebuild-features", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--asr-epochs", type=int, default=0)
    parser.add_argument("--asr-learning-rate", type=float, default=1e-4)
    args = parser.parse_args()

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    dtype = torch.float16 if device.type == "mps" else torch.float32
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    audio_paths = _ensure_corpus()
    processor = transformers.AutoProcessor.from_pretrained(MODEL_ID)

    feature_paths = _prepare_feature_pairs(
        audio_paths,
        processor,
        device,
        dtype,
        rebuild=args.rebuild_features,
    )
    adapter = ShortStateAdapter().to(device=device, dtype=torch.float32)
    if args.resume and CHECKPOINT.exists():
        adapter.load_state_dict(
            torch.load(CHECKPOINT, map_location=device, weights_only=True)
        )
        print(f"resumed={CHECKPOINT}", flush=True)
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.learning_rate)

    print(
        f"training_samples={len(feature_paths) - 1} holdout=1 "
        f"trainable_parameters={sum(p.numel() for p in adapter.parameters())}",
        flush=True,
    )
    best_holdout = float("inf")
    train_paths = feature_paths[:-1]
    holdout_path = feature_paths[-1]
    for epoch in range(1, args.epochs + 1):
        adapter.train()
        losses: list[float] = []
        order = torch.randperm(len(train_paths)).tolist()
        for index in order:
            pair = torch.load(train_paths[index], map_location="cpu", weights_only=True)
            source = pair["short"].to(device=device, dtype=torch.float32)
            target = pair["target"].to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            prediction = adapter(source)
            loss = torch.nn.functional.mse_loss(prediction, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        holdout_loss = _feature_loss(adapter, holdout_path, device)
        print(
            f"epoch={epoch} train_mse={np.mean(losses):.7f} "
            f"holdout_mse={holdout_loss:.7f}",
            flush=True,
        )
        if holdout_loss < best_holdout:
            best_holdout = holdout_loss
            torch.save(adapter.state_dict(), CHECKPOINT)

    adapter.load_state_dict(
        torch.load(CHECKPOINT, map_location=device, weights_only=True)
    )
    if args.asr_epochs:
        _train_asr_loss(
            adapter=adapter,
            train_paths=train_paths,
            train_sentences=TRAIN_SENTENCES,
            holdout_path=holdout_path,
            processor=processor,
            device=device,
            dtype=dtype,
            epochs=args.asr_epochs,
            learning_rate=args.asr_learning_rate,
        )
        adapter.load_state_dict(
            torch.load(ASR_CHECKPOINT, map_location=device, weights_only=True)
        )
    result = _evaluate_transcript(adapter, holdout_path, processor, device, dtype)
    result.update(
        {
            "expected": EXPECTED_HOLDOUT,
            "bestHoldoutMse": best_holdout,
            "checkpoint": str(CHECKPOINT),
            "asrCheckpoint": str(ASR_CHECKPOINT) if args.asr_epochs else None,
        }
    )
    print(json.dumps(result, indent=2), flush=True)


def _ensure_corpus() -> list[pathlib.Path]:
    audio_dir = WORK_DIR / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    paths: list[pathlib.Path] = []
    for index, sentence in enumerate(TRAIN_SENTENCES):
        output = audio_dir / f"train-{index:02d}.wav"
        paths.append(output)
        if output.exists():
            continue
        aiff = audio_dir / f"train-{index:02d}.aiff"
        subprocess.run(
            ["say", "-v", VOICES[index % len(VOICES)], sentence, "-o", str(aiff)],
            check=True,
        )
        subprocess.run(
            ["afconvert", "-f", "WAVE", "-d", "LEI16@16000", str(aiff), str(output)],
            check=True,
        )
        aiff.unlink()
    if not HOLDOUT_AUDIO.exists():
        raise FileNotFoundError(HOLDOUT_AUDIO)
    paths.append(HOLDOUT_AUDIO)
    return paths


def _prepare_feature_pairs(
    audio_paths: list[pathlib.Path],
    processor,
    device: torch.device,
    dtype: torch.dtype,
    *,
    rebuild: bool,
) -> list[pathlib.Path]:
    feature_dir = WORK_DIR / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    paths = [feature_dir / f"sample-{i:02d}.pt" for i in range(len(audio_paths))]
    if not rebuild and all(path.exists() for path in paths):
        return paths

    config = transformers.WhisperConfig.from_pretrained(MODEL_ID)
    short_encoder = ModifiedWhisperEncoder(config).to(device=device, dtype=dtype).eval()
    short_encoder.init_latency_mask(None, dtype=dtype)
    _load_prefixed_weights(
        short_encoder,
        ULTRAVOX_WEIGHTS,
        prefix="audio_tower.",
        device=device,
    )
    print("encoding shortened Ultravox states...", flush=True)
    for audio_path, feature_path in zip(audio_paths, paths):
        audio = _read_pcm16_wav(audio_path)
        features = processor(
            audio,
            sampling_rate=16_000,
            padding="longest",
            max_length=len(audio),
            return_attention_mask=True,
            return_tensors="pt",
        )
        with torch.inference_mode():
            short = short_encoder(
                features.input_features.to(device=device, dtype=dtype),
                audio_len=torch.tensor(
                    [features.input_features.shape[-1]], device=device
                ),
            ).last_hidden_state
        torch.save({"short": short.cpu().half()}, feature_path)
    del short_encoder
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()

    target_encoder = WhisperEncoder(config).to(device=device, dtype=dtype).eval()
    original_weights = pathlib.Path(hf_hub_download(MODEL_ID, "model.safetensors"))
    _load_prefixed_weights(
        target_encoder,
        original_weights,
        prefix="model.encoder.",
        device=device,
    )
    print("encoding stock Whisper target states...", flush=True)
    for audio_path, feature_path in zip(audio_paths, paths):
        audio = _read_pcm16_wav(audio_path)
        padded = processor(
            audio,
            sampling_rate=16_000,
            return_tensors="pt",
        ).input_features.to(device=device, dtype=dtype)
        with torch.inference_mode():
            target = target_encoder(padded).last_hidden_state
        pair = torch.load(feature_path, map_location="cpu", weights_only=True)
        pair["target"] = target.cpu().half()
        torch.save(pair, feature_path)
    del target_encoder
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    return paths


def _load_prefixed_weights(
    module: nn.Module,
    weights_path: pathlib.Path,
    *,
    prefix: str,
    device: torch.device,
) -> None:
    state = module.state_dict()
    with (
        torch.no_grad(),
        safe_open(weights_path, framework="pt", device="cpu") as weights,
    ):
        available = set(weights.keys())
        for name, destination in state.items():
            key = prefix + name
            if key not in available:
                raise KeyError(key)
            destination.copy_(
                weights.get_tensor(key).to(device=device, dtype=destination.dtype)
            )


def _train_asr_loss(
    *,
    adapter: ShortStateAdapter,
    train_paths: list[pathlib.Path],
    train_sentences: list[str],
    holdout_path: pathlib.Path,
    processor,
    device: torch.device,
    dtype: torch.dtype,
    epochs: int,
    learning_rate: float,
) -> None:
    print("loading frozen Whisper decoder for ASR-loss training...", flush=True)
    model = _load_whisper_decoder_model(MODEL_ID, device, dtype)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    optimizer = torch.optim.AdamW(adapter.parameters(), lr=learning_rate)
    best_loss = float("inf")
    holdout_tokens = _teacher_forcing_tokens(processor, EXPECTED_HOLDOUT, device)
    for epoch in range(1, epochs + 1):
        adapter.train()
        losses: list[float] = []
        for index in torch.randperm(len(train_paths)).tolist():
            pair = torch.load(train_paths[index], map_location="cpu", weights_only=True)
            source = pair["short"].to(device=device, dtype=torch.float32)
            tokens = _teacher_forcing_tokens(processor, train_sentences[index], device)
            optimizer.zero_grad(set_to_none=True)
            loss = _transcript_loss(model, adapter(source), tokens, dtype)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        adapter.eval()
        holdout_pair = torch.load(holdout_path, map_location="cpu", weights_only=True)
        holdout_source = holdout_pair["short"].to(device=device, dtype=torch.float32)
        with torch.no_grad():
            holdout_loss = float(
                _transcript_loss(
                    model, adapter(holdout_source), holdout_tokens, dtype
                ).cpu()
            )
        print(
            f"asr_epoch={epoch} train_ce={np.mean(losses):.5f} "
            f"holdout_ce={holdout_loss:.5f}",
            flush=True,
        )
        if holdout_loss < best_loss:
            best_loss = holdout_loss
            torch.save(adapter.state_dict(), ASR_CHECKPOINT)

    del model
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()


def _teacher_forcing_tokens(processor, text: str, device: torch.device) -> torch.Tensor:
    forced = processor.get_decoder_prompt_ids(
        language="english", task="transcribe", no_timestamps=True
    )
    prefix = [processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")]
    prefix.extend(token_id for _, token_id in forced)
    text_ids = processor.tokenizer.encode(text, add_special_tokens=False)
    sequence = [*prefix, *text_ids, processor.tokenizer.eos_token_id]
    return torch.tensor([sequence], dtype=torch.long, device=device)


def _transcript_loss(
    model: transformers.WhisperForConditionalGeneration,
    encoder_state: torch.Tensor,
    tokens: torch.Tensor,
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
        tokens[:, 1:].reshape(-1),
    )


@torch.inference_mode()
def _feature_loss(
    adapter: ShortStateAdapter, feature_path: pathlib.Path, device: torch.device
) -> float:
    adapter.eval()
    pair = torch.load(feature_path, map_location="cpu", weights_only=True)
    source = pair["short"].to(device=device, dtype=torch.float32)
    target = pair["target"].to(device=device, dtype=torch.float32)
    return float(torch.nn.functional.mse_loss(adapter(source), target).cpu())


@torch.inference_mode()
def _evaluate_transcript(
    adapter: ShortStateAdapter,
    feature_path: pathlib.Path,
    processor,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, str]:
    adapter.eval()
    pair = torch.load(feature_path, map_location="cpu", weights_only=True)
    source = pair["short"].to(device=device, dtype=torch.float32)
    adapted = adapter(source).to(dtype)
    target = pair["target"].to(device=device, dtype=dtype)

    decoder = _load_whisper_decoder_model(MODEL_ID, device, dtype)
    adapted_ids = _decode_whisper_official(
        model=decoder,
        encoder_state=adapted,
        max_new_tokens=64,
    )
    target_ids = _decode_whisper_official(
        model=decoder,
        encoder_state=target,
        max_new_tokens=64,
    )
    return {
        "adaptedTranscript": processor.batch_decode(
            [adapted_ids], skip_special_tokens=True
        )[0].strip(),
        "targetTranscript": processor.batch_decode(
            [target_ids], skip_special_tokens=True
        )[0].strip(),
    }


if __name__ == "__main__":
    main()
