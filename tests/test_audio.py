import io

import numpy as np
import pytest
import soundfile as sf

from holos_tts.audio import UnsupportedFormatError, encode, fit_level, to_pcm16

RATE = 24000
TONE = (0.5 * np.sin(np.linspace(0, 2 * np.pi * 440, RATE))).astype(np.float32)


@pytest.mark.parametrize(
    ("fmt", "media_type"),
    [
        ("mp3", "audio/mpeg"),
        ("wav", "audio/wav"),
        ("flac", "audio/flac"),
        ("opus", "audio/ogg"),
    ],
)
def test_encode_makes_a_readable_file(fmt, media_type):
    data, mime = encode(TONE, RATE, fmt)
    assert mime == media_type
    audio, rate = sf.read(io.BytesIO(data))
    assert rate in (RATE, 48000)
    assert abs(len(audio) / rate - 1.0) < 0.1


def test_encode_pcm_is_raw_16_bit():
    data, mime = encode(TONE, RATE, "pcm")
    assert mime == "audio/pcm"
    assert len(data) == RATE * 2


def test_encode_refuses_an_unknown_format():
    with pytest.raises(UnsupportedFormatError, match="aac"):
        encode(TONE, RATE, "aac")


def test_fit_level_scales_down_only_loud_audio():
    loud = np.array([0.0, 2.0, -1.0], dtype=np.float32)
    assert np.max(np.abs(fit_level(loud))) == pytest.approx(1.0)
    quiet = np.array([0.0, 0.5], dtype=np.float32)
    assert fit_level(quiet) is quiet
    assert fit_level(np.zeros(0, dtype=np.float32)).size == 0


def test_to_pcm16_clips():
    data = np.frombuffer(to_pcm16(np.array([1.5, -1.5, 0.0], dtype=np.float32)), dtype="<i2")
    assert data.tolist() == [32767, -32767, 0]
