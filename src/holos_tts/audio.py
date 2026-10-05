"""Audio formats of the OpenAI speech API, made from float32 mono samples."""

from __future__ import annotations

import io

import numpy as np
import soundfile as sf

# response_format of the OpenAI API: (soundfile format, soundfile subtype, media type)
FORMATS = {
    "mp3": ("MP3", None, "audio/mpeg"),
    "wav": ("WAV", "PCM_16", "audio/wav"),
    "flac": ("FLAC", "PCM_16", "audio/flac"),
    "opus": ("OGG", "OPUS", "audio/ogg"),
    "pcm": (None, None, "audio/pcm"),
}


class UnsupportedFormatError(ValueError):
    """The server cannot make the asked audio format."""


def fit_level(audio: np.ndarray) -> np.ndarray:
    """Scale the audio down when a sample goes past full scale, so that the 16-bit formats do not clip."""
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        return audio / peak

    return audio


def to_pcm16(audio: np.ndarray) -> bytes:
    """Return signed 16-bit little-endian samples."""
    return (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()


def encode(audio: np.ndarray, sample_rate: int, response_format: str) -> tuple[bytes, str]:
    """Return the audio file bytes and the media type for ``response_format``."""
    try:
        file_format, subtype, media_type = FORMATS[response_format]
    except KeyError as e:
        msg = f"response_format {response_format!r} is not supported. Use one of: {', '.join(FORMATS)}."
        raise UnsupportedFormatError(msg) from e

    audio = fit_level(audio.astype(np.float32, copy=False))
    if file_format is None:
        return to_pcm16(audio), media_type

    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format=file_format, subtype=subtype)
    return buffer.getvalue(), media_type
