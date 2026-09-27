# Shared Whisper adapter and speech-gate checkpoints

`adapter-asr-robust-17000.pt` is the selected checkpoint from the robust
LibriSpeech adapter run at 17,000 training samples.

- Combined validation WER: 4.183%
- Clean validation WER: 4.056%
- Augmented validation WER: 4.309%
- SHA-256: `b2ef5f97bb4a9bb3ac4b3689076e15606f22b3e7b2309759c2d511399757f210`

The checkpoint is stored with Git LFS.

The current speech-gated runtime pair is:

- `adapter-asr-robust-20000.pt`: robust ASR adapter; SHA-256
  `3d692502c679815b67f5f148873b79866b16521b881140d6a89389868721346b`.
- `speech-head-best-16000.pt`: speech/no-speech head trained for 10,000 examples
  and fine-tuned for 6,000 more; SHA-256
  `393e1227a44d148044a68be86c6a3eeb9b460b71827d668c96ebe92cc7b5da20`.

At a 0.5 threshold, the selected head achieved 100% clean-speech recall, 100%
heavy-noise-speech recall, and 0% no-speech false positives on three fixed
50-example validation sets. The paired adapter measured 4.07% clean WER and
38.37% heavy-noise WER. See
[`docs/speech-gated-shared-whisper-adapter.md`](../../docs/speech-gated-shared-whisper-adapter.md)
for architecture, training, limitations, and integration details.
