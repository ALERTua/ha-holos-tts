"""Audio formats of the OpenAI speech API, made from float32 mono samples."""

from __future__ import annotations

import functools
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


# ITU-R BS.1770-4: 400 ms blocks with 75 % overlap, an absolute gate and a relative gate
BLOCK_SECONDS = 0.4
STEP_SECONDS = 0.1
ABSOLUTE_GATE_LUFS = -70.0
RELATIVE_GATE_LU = -10.0
LOUDNESS_OFFSET = -0.691
# the K-weighting filters of BS.1770 as analog prototypes, so that any sample rate gets its coefficients
SHELF_HZ = 1681.974450955533
SHELF_GAIN_DB = 3.999843853973347
SHELF_Q = 0.7071752369554196
SHELF_VB_POWER = 0.4996667741545416
HIGH_PASS_HZ = 38.13547087602444
HIGH_PASS_Q = 0.5003270373238773
# the impulse response of the K-weighting has decayed after this time, so the FFT filter does not wrap around
FILTER_TAIL_SECONDS = 0.5
# a sample ceiling of -2 dBFS keeps the true peak at or below -1 dBFS, also after an MP3 encode
LIMIT_CEILING_DB = -2.0
LIMIT_HALF_WINDOW_SECONDS = 0.005
# a quiet chunk, for example a breath, must not become loud noise
MAX_GAIN_DB = 24.0
# the limiter takes some loudness off; a second pass makes up a loss above this value
CORRECTION_LU = 0.2


class UnsupportedFormatError(ValueError):
    """The server cannot make the asked audio format."""


def _db_to_gain(db: float) -> float:
    return float(10 ** (db / 20))


def _biquad_response(b: tuple[float, float, float], a: tuple[float, float, float], z: np.ndarray) -> np.ndarray:
    """Return the response of one biquad at the points ``z`` on the unit circle."""
    zi = z.conj()
    zi2 = zi * zi
    return (b[0] + b[1] * zi + b[2] * zi2) / (a[0] + a[1] * zi + a[2] * zi2)


# the FFT sizes are powers of two, so a few responses serve all chunks
@functools.lru_cache(maxsize=8)
def _k_weighting(size: int, rate: int) -> np.ndarray:
    """Return the K-weighting response of BS.1770 for the bins of ``numpy.fft.rfft`` of ``size`` samples."""
    z = np.exp(2j * np.pi * np.fft.rfftfreq(size))
    k = np.tan(np.pi * SHELF_HZ / rate)
    vh = 10 ** (SHELF_GAIN_DB / 20)
    vb = vh**SHELF_VB_POWER
    a0 = 1 + k / SHELF_Q + k * k
    shelf = _biquad_response(
        ((vh + vb * k / SHELF_Q + k * k) / a0, 2 * (k * k - vh) / a0, (vh - vb * k / SHELF_Q + k * k) / a0),
        (1.0, 2 * (k * k - 1) / a0, (1 - k / SHELF_Q + k * k) / a0),
        z,
    )
    k = np.tan(np.pi * HIGH_PASS_HZ / rate)
    a0 = 1 + k / HIGH_PASS_Q + k * k
    high_pass = _biquad_response(
        (1.0, -2.0, 1.0),
        (1.0, 2 * (k * k - 1) / a0, (1 - k / HIGH_PASS_Q + k * k) / a0),
        z,
    )
    return shelf * high_pass


def _block_powers(weighted: np.ndarray, rate: int) -> np.ndarray:
    """Return the mean square of each gating block, or of the whole audio when it is shorter than one block."""
    block = round(BLOCK_SECONDS * rate)
    if weighted.size < block:
        return np.array([np.mean(weighted**2)])

    step = round(STEP_SECONDS * rate)
    sums = np.concatenate(([0.0], np.cumsum(weighted**2)))
    starts = np.arange(0, weighted.size - block + 1, step)
    return (sums[starts + block] - sums[starts]) / block


def _power_to_lufs(power: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore"):
        return LOUDNESS_OFFSET + 10 * np.log10(power)


def loudness(audio: np.ndarray, rate: int) -> float | None:
    """Return the integrated loudness of ITU-R BS.1770-4 in LUFS, or None for silence."""
    if audio.size == 0:
        return None

    # a power of two keeps the FFT fast for each length
    size = 1 << (audio.size + round(FILTER_TAIL_SECONDS * rate) - 1).bit_length()
    spectrum = np.fft.rfft(audio.astype(np.float64), size) * _k_weighting(size, rate)
    weighted = np.fft.irfft(spectrum, size)[: audio.size]
    powers = _block_powers(weighted, rate)
    powers = powers[_power_to_lufs(powers) > ABSOLUTE_GATE_LUFS]
    if powers.size == 0:
        return None

    relative_gate = float(_power_to_lufs(np.array([powers.mean()]))[0]) + RELATIVE_GATE_LU
    powers = powers[_power_to_lufs(powers) > relative_gate]
    return float(_power_to_lufs(np.array([powers.mean()]))[0])


def _running_min(values: np.ndarray, half: int) -> np.ndarray:
    """Return the minimum of each window of ``2 * half + 1`` values around each value, with the edges repeated."""
    width = 2 * half + 1
    padded = np.pad(values, half, mode="edge")
    size = padded.size
    blocks = np.pad(padded, (0, -size % width), mode="edge").reshape(-1, width)
    prefix = np.minimum.accumulate(blocks, axis=1).ravel()
    suffix = np.minimum.accumulate(blocks[:, ::-1], axis=1)[:, ::-1].ravel()
    # a window starts in one block and ends in the next one, or fills one block exactly
    return np.minimum(suffix[: size - width + 1], prefix[width - 1 : size])


def _running_mean(values: np.ndarray, half: int) -> np.ndarray:
    """Return the mean of each window of ``2 * half + 1`` values around each value, with the edges repeated."""
    width = 2 * half + 1
    sums = np.concatenate(([0.0], np.cumsum(np.pad(values, half, mode="edge"))))
    return (sums[width:] - sums[:-width]) / width


def limit(audio: np.ndarray, rate: int, ceiling_db: float = LIMIT_CEILING_DB) -> np.ndarray:
    """Lower the peaks above the ceiling with a smooth gain. Audio below the ceiling stays the same."""
    ceiling = _db_to_gain(ceiling_db)
    # float64, because a running sum of float32 values drifts by more than the margin of the ceiling
    peaks = np.abs(audio.astype(np.float64))
    if audio.size == 0 or float(peaks.max()) <= ceiling:
        return audio

    needed = np.minimum(1.0, ceiling / np.maximum(peaks, ceiling))
    half = max(1, round(LIMIT_HALF_WINDOW_SECONDS * rate))
    # each mean covers windows that all hold the sample, so the gain is never above the gain that the sample needs
    gain = _running_mean(_running_min(needed, half), half)
    return (audio * gain).astype(np.float32)


def normalize(audio: np.ndarray, rate: int, target_lufs: float) -> np.ndarray:
    """Bring the audio to the target loudness, with the peak limiter. A target of 0 keeps the audio as it is."""
    if not target_lufs:
        return audio

    level = loudness(audio, rate)
    if level is None:
        return audio

    gain_db = min(target_lufs - level, MAX_GAIN_DB)
    result = limit(audio * _db_to_gain(gain_db), rate)
    after = loudness(result, rate)
    if after is not None and target_lufs - after > CORRECTION_LU and gain_db < MAX_GAIN_DB:
        gain_db = min(gain_db + target_lufs - after, MAX_GAIN_DB)
        result = limit(audio * _db_to_gain(gain_db), rate)

    return result.astype(np.float32, copy=False)


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
