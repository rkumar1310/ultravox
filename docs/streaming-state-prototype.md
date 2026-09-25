# Incremental Ultravox v0.5 state prototype

This experiment keeps one Llama KV cache alive while text tokens and projected
audio embeddings arrive. It does not update model weights during inference.
The per-session hidden state and KV cache are what change.

```text
text prefix ───────────────┐
                          v
PCM → mel → causal Whisper → projector → append to Llama KV → classifier
                          ^                             │
context text ─────────────┘                             └→ decode only on decision
```

## What is implemented

- Text tokens and audio embeddings share one append-only Llama state.
- Whisper runs block-causally and keeps per-layer attention KV caches.
- One 160 ms block maps to one Llama audio token for the v0.5 configuration.
- One mel-frame (10 ms) look-ahead protects the Whisper convolution boundary.
- A small classifier can run from the latest hidden state without the expensive
  full-vocabulary language-model head.
- Selected single-token labels can be scored for a no-training smoke test.

## What is not yet production-ready

- The v0.5 checkpoint was released with causal masking disabled. Quality under
  this mask must be measured and will likely require causal-mask fine-tuning.
- The log-mel extractor and tiny convolutional frontend are replayed over the
  accumulated utterance. The expensive Whisper transformer and Llama both keep
  incremental caches; frontend caching is still needed for strictly constant
  per-chunk work.
- The classifier head is untrained. It needs labelled examples for the desired
  decisions such as `continue`, `respond`, `interrupt`, and `backchannel`.
- The prototype currently supports a single stream; production batching comes
  after semantic correctness is established.

## Local probe

The model repository points at Meta's gated Llama repository. The probe defaults
to the public `unsloth/Llama-3.2-1B-Instruct` mirror so it can run without a local
Hugging Face login. Provide a 16 kHz mono/stereo PCM16 WAV:

```bash
PYTHONPATH=. python scripts/streaming_state_probe.py speech.wav \
  --device mps --chunk-ms 160
```

Every emitted line shows that another audio token entered the same Llama cache
and that an intermediate hidden state was available before text decoding.
