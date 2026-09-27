# Speech-gated shared Whisper adapter

## Runtime design

The Ultravox audio encoder runs once. Its variable-length hidden state feeds two
independent consumers: the existing Ultravox/Llama path and a gated Whisper
transcription path.

```text
PCM audio
   |
   v
Ultravox Whisper audio encoder (one shared pass)
   |
   +---------------------------> Ultravox projector -> Llama
   |
   v
Shared audio state [batch, frames, 1280]
   |
   v
SpeechPresenceHead (323,393 parameters)
   |
   v
P(speech) >= 0.50?
   |
   +-- no  -> suppress transcription; adapter and decoder do not run
   |
   +-- yes -> ShortStateAdapter -> Whisper decoder -> transcript
```

`GatedWhisperAdapter` preserves batch identity with `active_indices`. It adapts
only accepted rows and returns those indices with the resulting Whisper encoder
states, allowing the caller to map decoded transcripts back to their original
requests.

The speech head reads the raw shared encoder state, not the adapter output. The
gate therefore does not alter ASR features, transcription, or the Llama path.

## Selected checkpoints

Both files are in `checkpoints/shared-whisper-adapter/` and stored with Git LFS.

| Component | File | Training point | SHA-256 |
|---|---|---:|---|
| Robust ASR adapter | `adapter-asr-robust-20000.pt` | 20,000 | `3d692502c679815b67f5f148873b79866b16521b881140d6a89389868721346b` |
| Speech gate | `speech-head-best-16000.pt` | 10,000 initial + 6,000 fine-tune | `393e1227a44d148044a68be86c6a3eeb9b460b71827d668c96ebe92cc7b5da20` |

The latest 20,000-example head also achieved the same classification accuracy,
but 16,000 had the best combined validation confidence and is the selected
runtime checkpoint.

## Training and results

The head was initialized randomly and trained on 10,000 examples, then
fine-tuned on a distinct 10,000 examples. The ASR adapter was frozen during the
second phase.

The fixed-threshold validation sets contained 50 clean-speech, 50
heavy-noise-speech, and 50 no-speech examples.

| Head checkpoint | Clean-speech recall | Heavy-noise-speech recall | No-speech false-positive rate |
|---:|---:|---:|---:|
| 10,000 | 100% | 100% | 0% |
| 16,000 selected | 100% | 100% | 0% |
| 20,000 latest | 100% | 100% | 0% |

With the frozen 20,000-example ASR adapter, measured transcription WER remains:

| Evaluation | WER |
|---|---:|
| Clean speech | 4.07% |
| Heavy-noise speech | 38.37% |

The gate fixes the prior 100% no-speech false-positive rate by preventing
Whisper from decoding silence or noise. It does not fix words misheard in heavy
noise; that requires separate ASR-adapter training with clean-speech replay and
anti-forgetting constraints.

An experimental attempt to update the ASR adapter alongside the head was
rejected: after 2,000 examples, heavy-noise WER had regressed to 109.30%. The
runtime therefore deliberately freezes the adapter.

## Code entry points

- `ultravox/inference/shared_whisper_adapter.py`: adapter, speech head, gated
  batch routing, and strict checkpoint loading.
- `scripts/dual_output_probe.py`: one-encoder-pass Llama plus gated Whisper
  inference example.
- `scripts/train_speech_presence_head.py`: two-phase head training; ASR-adapter
  updates are disabled by default and require explicit experimental opt-in.
- `scripts/modal_train_shared_whisper_adapter.py`: reproducible single-T4
  training orchestration and checkpoint persistence.

Example:

```bash
python3 scripts/dual_output_probe.py path/to/16khz-mono-pcm.wav
```

The probe loads the selected checkpoints by default and reports
`speechProbability`, `transcriptionSuppressed`, and the transcript.

## Remaining validation

The 100% gate result is from small, fixed validation sets. Before production,
evaluate the selected 16,000 checkpoint on a larger unseen corpus containing
room tone, music, transient noise, overlapping speakers, quiet speech, and
languages expected in deployment.
