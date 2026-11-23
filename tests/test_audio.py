import io

import numpy as np
import pytest
import soundfile as sf

from holos_tts.audio import (
    LIMIT_CEILING_DB,
    MAX_GAIN_DB,
    UnsupportedFormatError,
    encode,
    fit_level,
    limit,
    loudness,
    normalize,
    to_pcm16,
)
from tests.fake_worker import speech_like

RATE = 24000
TONE = (0.5 * np.sin(np.linspace(0, 2 * np.pi * 440, RATE))).astype(np.float32)
# ffmpeg -af ebur128 reads -17.0 LUFS for this signal
NOISE = (0.1 * np.random.default_rng(0).standard_normal(3 * RATE)).astype(np.float32)
CEILING = 10 ** (LIMIT_CEILING_DB / 20)


def sine(amplitude, seconds=2.0, hz=997):
    t = np.arange(int(seconds * RATE)) / RATE
    return (amplitude * np.sin(2 * np.pi * hz * t)).astype(np.float32)


@pytest.mark.parametrize("amplitude", [1.0, 0.1, 0.01])
def test_loudness_of_a_997_hz_sine_follows_bs1770(amplitude):
    # BS.1770 calibrates a full-scale sine of 997 Hz to -3.01 LUFS
    assert loudness(sine(amplitude), RATE) == pytest.approx(-3.01 + 20 * np.log10(amplitude), abs=0.1)


def test_loudness_of_noise_matches_ffmpeg():
    assert loudness(NOISE, RATE) == pytest.approx(-17.0, abs=0.2)


def test_loudness_gates_out_the_silence_between_the_words():
    words = sine(0.1)
    with_pause = np.concatenate([words, np.zeros(2 * RATE, dtype=np.float32), words])
    # without the gates, a third of silence gives 1.76 dB less; only the blocks at the two edges still count
    assert loudness(words, RATE) - loudness(with_pause, RATE) == pytest.approx(0.34, abs=0.1)


def test_loudness_of_a_chunk_shorter_than_one_block():
    assert loudness(sine(0.1, seconds=0.2), RATE) == pytest.approx(-23.0, abs=0.2)


@pytest.mark.parametrize("silent", [np.zeros(RATE, dtype=np.float32), np.zeros(0, dtype=np.float32)])
def test_loudness_of_silence_is_none(silent):
    assert loudness(silent, RATE) is None


def test_limit_keeps_each_sample_at_or_below_the_ceiling():
    loud = sine(3.0)
    limited = limit(loud, RATE)
    assert np.max(np.abs(limited)) <= CEILING + 1e-6
    assert limited.dtype == np.float32


def test_limit_keeps_audio_below_the_ceiling_as_it_is():
    quiet = sine(0.5)
    assert limit(quiet, RATE) is quiet
    empty = np.zeros(0, dtype=np.float32)
    assert limit(empty, RATE) is empty


def test_limit_changes_the_gain_smoothly():
    # one loud click in quiet audio: the gain falls and rises over milliseconds, not in one sample
    audio = np.full(RATE, 0.1, dtype=np.float32)
    audio[RATE // 2] = 4.0
    gain = limit(audio, RATE) / audio
    assert np.max(np.abs(np.diff(gain))) < 0.05
    assert gain[0] == pytest.approx(1.0)
    assert gain[RATE // 2] * 4.0 <= CEILING + 1e-6


@pytest.mark.parametrize(("amplitude", "target"), [(0.05, -16.0), (0.05, -20.0), (0.9, -16.0)])
def test_normalize_reaches_the_target_without_peaks_above_the_ceiling(amplitude, target):
    result = normalize(speech_like(amplitude), RATE, target)
    assert loudness(result, RATE) == pytest.approx(target, abs=0.3)
    assert np.max(np.abs(result)) <= CEILING + 1e-6
    assert result.dtype == np.float32


def test_normalize_makes_up_the_loss_of_the_limiter():
    # at -8 LUFS the peaks of the tone pass the ceiling, so one pass ends below the target
    audio = speech_like(0.2)
    one_pass = limit(audio * 10 ** ((-8.0 - loudness(audio, RATE)) / 20), RATE)
    two_passes = loudness(normalize(audio, RATE, -8.0), RATE)
    assert loudness(one_pass, RATE) + 0.1 < two_passes < -8.0


def test_normalize_raises_a_very_quiet_chunk_by_the_largest_gain_only():
    faint = speech_like(1e-3)
    result = normalize(faint, RATE, -16.0)
    assert loudness(result, RATE) == pytest.approx(loudness(faint, RATE) + MAX_GAIN_DB, abs=0.1)


def test_normalize_keeps_the_audio_when_off_or_silent():
    audio = speech_like(0.05)
    assert normalize(audio, RATE, 0.0) is audio
    silent = np.zeros(RATE, dtype=np.float32)
    assert normalize(silent, RATE, -16.0) is silent


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
