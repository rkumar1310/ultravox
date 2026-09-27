"""Backward-compatible import for the shared-state speech classifier."""

from __future__ import annotations

from ultravox.inference.shared_whisper_adapter import SpeechPresenceHead

__all__ = ["SpeechPresenceHead"]
