"""
HolosTTS acoustic model in ONNX Runtime: phonemes and a voice vector in, 24 kHz audio out.

The three-stage pass follows the MIT package ``holos-tts-onnx`` of the model author.
"""

from __future__ import annotations

import contextlib
import dataclasses
import itertools
import json
import logging
import platform
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from huggingface_hub import hf_hub_download

from .cache_files import remove_path
from .constants import CHUNK_CHARS, SAMPLE_RATE

if TYPE_CHECKING:
    from types import ModuleType

LOG = logging.getLogger(__name__)

VOICE_SIZE = 256

MODEL_REPO = "patriotyk/HolosTTS"
# The author changed the model architecture between revisions, so the server pins the files it was tested with
MODEL_REVISION = "fdd917a47933f8b2b79b50ee1c25a239b3f9e549"
BUNDLES = {"cpu": "onnx-int8", "cuda": "onnx"}
MANIFEST_FILE = "holos_onnx.json"
MANIFEST_FORMAT = ("holos-onnx", 2)
STAGES = ("encoder", "prosody", "decoder")
# both bundles hold the same voices
VOICES_FILE = "onnx-int8/voices.npz"
# the optimized graphs of the older single-graph model, which nothing reads now
OLD_CACHE_FOLDER = "onnx"
CPUINFO = Path("/proc/cpuinfo")
# ONNX Runtime: the U8S8 int8 matmul overflows on x86 without VNNI, and this entry selects the exact U8U8 kernels
QUANT_PRECISION_ENTRY = "session.x64quantprecision"
# idle intra-op threads that spin keep the cores busy between the runs and make each run slower
SPINNING_ENTRY = "session.intra_op.allow_spinning"
VNNI_FLAGS = frozenset({"avx_vnni", "avx512_vnni"})

# the checkpoint table repeats the apostrophe, and TOKEN_IDS keeps its last index as the checkpoint does
VOCAB = (
    '$-\xb4;:,.!?\xa1\xbf—…"\xab\xbb“” ()†/=ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    "abcdefghijklmnopqrstuvwxyz\xe9\xfd\xed\xf3'̯'͡ɑɐɒ\xe6ɓʙβ"
    "ɔɕ\xe7ɗɖ\xf0ʤəɘɚɛɜɝɞɟʄɡ"
    "ɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟ"
    "ɱɯɰŋɳɲɴ\xf8ɵɸθœɶʘɹɺ"
    "ɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤ"
    "ʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈ"
    "ˌːˑʼʴʰʱʲ'̩'ᵻ"
)
TOKEN_IDS = {symbol: index for index, symbol in enumerate(VOCAB)}
PAD_ID = 0

# the cost of a cut: the ear does not notice a cut after a sentence end, but a cut between words breaks a phrase
_CUT_COSTS = {**dict.fromkeys(".!?…:", 1), **dict.fromkeys(",;—–-", 3)}
_WORD_CUT_COST = 9
_SPACES = re.compile(r"\s+")
# tokenize adds a pad token on both ends of each pass
_PADS = 2

_NUMBER = re.compile(r"(\d+)")


def tokenize(phonemes: str) -> list[int]:
    """Return the token ids of ``phonemes`` with the pad token on both ends. Unknown symbols drop out."""
    return [
        PAD_ID,
        *(TOKEN_IDS[symbol] for symbol in phonemes if symbol in TOKEN_IDS),
        PAD_ID,
    ]


def split_phonemes(phonemes: str, budget: int) -> list[str]:
    """
    Split ``phonemes`` at spaces into parts of at most ``budget`` tokens. A longer word stays whole.

    The parts have the lowest sum of the cut costs, and of these the most even sizes.
    """
    if len(tokenize(phonemes)) <= budget:
        return [phonemes]

    phonemes = phonemes.strip()
    spaces = list(_SPACES.finditer(phonemes))
    # word i spans starts[i]:ends[i], and a part is a run of neighbor words
    starts = [0, *(space.end() for space in spaces)]
    ends = [*(space.start() for space in spaces), len(phonemes)]
    # the token count of each prefix, without the pads
    counts = [0, *itertools.accumulate(symbol in TOKEN_IDS for symbol in phonemes)]
    # best[i]: the cost, the sum of the squared part sizes and the first word of the last part, for words 0..i
    best: list[tuple[int, int, int]] = []
    for last in range(len(ends)):
        options = []
        for first in range(last, -1, -1):
            size = counts[ends[last]] - counts[starts[first]] + _PADS
            if size > budget and first < last:
                break

            cost, squares = 0, 0
            if first:
                cost = best[first - 1][0] + _CUT_COSTS.get(phonemes[ends[first - 1] - 1], _WORD_CUT_COST)
                squares = best[first - 1][1]

            options.append((cost, squares + size * size, first))

        best.append(min(options))

    parts: list[str] = []
    last = len(ends) - 1
    while last >= 0:
        first = best[last][2]
        parts.append(phonemes[starts[first] : ends[last]])
        last = first - 1

    return parts[::-1]


def _natural_key(name: str) -> list[int | str]:
    return [int(part) if part.isdigit() else part.lower() for part in _NUMBER.split(name)]


def order_voices(names: list[str] | set[str], default: str) -> list[str]:
    """Put the default voice first, then named voices, then numbered ``Speaker_N`` voices in number order."""
    rest = sorted(
        (name for name in names if name != default),
        key=lambda n: (n.startswith("Speaker_"), _natural_key(n)),
    )
    return [default, *rest] if default in names else rest


def _to_vector(value: Any, source: str) -> np.ndarray:
    array = np.asarray(value.numpy() if hasattr(value, "numpy") else value, dtype=np.float32).reshape(-1)
    if array.size != VOICE_SIZE:
        msg = f"{source}: a voice must have {VOICE_SIZE} numbers, it has {array.size}"
        raise ValueError(msg)

    return array


def load_voice_file(path: Path) -> np.ndarray:
    """Read one voice vector from a ``.npy`` file or a ``.pt`` file of the HolosTTS demo."""
    if path.suffix == ".npy":
        return _to_vector(np.load(path), str(path))

    import torch  # noqa: PLC0415 - the main process must not import torch

    return _to_vector(torch.load(path, map_location="cpu", weights_only=True), str(path))


def read_voices_npz(path: Path) -> dict[str, np.ndarray]:
    """Return the voices of a ``voices.npz`` file: a ``names`` array and a ``values`` array of the same length."""
    with np.load(path, allow_pickle=False) as archive:
        names = archive["names"].astype(str).tolist()
        values = archive["values"]

    if len(names) != len(values):
        msg = f"{path}: {len(names)} voice names for {len(values)} voices"
        raise ValueError(msg)

    return {name: _to_vector(value, name) for name, value in zip(names, values, strict=True)}


def load_voices(voices_dir: Path) -> dict[str, np.ndarray]:
    """Return the voices of the model and the extra voices from ``voices_dir``. An extra voice wins on a name clash."""
    voices = read_voices_npz(Path(hf_hub_download(MODEL_REPO, VOICES_FILE, revision=MODEL_REVISION)))
    if voices_dir.is_dir():
        for file in sorted(voices_dir.iterdir()):
            if file.suffix not in (".npy", ".pt"):
                continue

            try:
                voices[file.stem] = load_voice_file(file)
            except Exception:
                LOG.exception("Skipping the voice file %s", file)
            else:
                LOG.info("Added the voice %r from %s", file.stem, file)

    return voices


def _cpu_flags() -> str:
    """Return the CPU feature line of ``/proc/cpuinfo``."""
    with contextlib.suppress(OSError):
        for line in CPUINFO.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("flags"):
                return line

    return platform.processor() + platform.machine()


def _needs_quant_precision() -> bool:
    """Return True on an x86 CPU without VNNI, where the fast int8 matmul of ONNX Runtime can give broken speech."""
    label, _, flags = _cpu_flags().partition(":")
    return label.strip() == "flags" and VNNI_FLAGS.isdisjoint(flags.split())


def _session_options(ort: ModuleType, threads: int, *, cpu: bool) -> Any:
    options = ort.SessionOptions()
    options.log_severity_level = 3
    if threads:
        options.intra_op_num_threads = threads

    if cpu:
        # the arena keeps the buffers of the longest chunk (about 300 MB), with no speed gain
        options.enable_cpu_mem_arena = False
        options.add_session_config_entry(SPINNING_ENTRY, "0")
        if _needs_quant_precision():
            options.add_session_config_entry(QUANT_PRECISION_ENTRY, "1")

    return options


def check_manifest(manifest: dict[str, Any]) -> None:
    """Refuse a bundle that this engine cannot run: another format, another decoder mode or another vocabulary."""
    found = (manifest.get("format"), manifest.get("format_version"))
    if found != MANIFEST_FORMAT:
        msg = f"The model bundle has the format {found}, and the server reads only {MANIFEST_FORMAT}"
        raise ValueError(msg)

    if manifest.get("decoder_mode") != "exact":
        msg = f"The model bundle has the decoder mode {manifest.get('decoder_mode')!r}, and the server needs 'exact'"
        raise ValueError(msg)

    if "".join(manifest.get("vocab", [])) != VOCAB:
        msg = "The vocabulary of the model bundle differs from the vocabulary of the server"
        raise ValueError(msg)


@dataclasses.dataclass
class _Encoded:
    """The outputs of the encoder and the prosody stage for one pass."""

    voice: np.ndarray
    acoustic_tokens: np.ndarray
    frame_to_token: np.ndarray
    f0: np.ndarray
    energy: np.ndarray

    @property
    def frames(self) -> int:
        return int(self.frame_to_token.size)


class HolosEngine:
    """The three ONNX Runtime sessions of the HolosTTS model: encoder, prosody and decoder."""

    sample_rate = SAMPLE_RATE
    chunk_chars = CHUNK_CHARS

    def __init__(
        self,
        voices: dict[str, np.ndarray],
        device: str = "cpu",
        *,
        threads: int = 0,
        cache_dir: Path | None = None,
    ) -> None:
        import onnxruntime as ort  # noqa: PLC0415

        providers = ["CPUExecutionProvider"]
        if device == "cuda":
            # the CUDA and cuDNN libraries come from pip packages, so ONNX Runtime must load them first
            ort.preload_dlls()
            providers.insert(0, "CUDAExecutionProvider")

        if cache_dir is not None and (cache_dir / OLD_CACHE_FOLDER).exists():
            LOG.info("Removing the optimized graphs of the older model from %s", cache_dir / OLD_CACHE_FOLDER)
            remove_path(cache_dir / OLD_CACHE_FOLDER)

        bundle = BUNDLES[device]
        LOG.info("Loading the HolosTTS model %s on %s", bundle, device)
        if device == "cpu" and _needs_quant_precision():
            LOG.info("The CPU has no VNNI, so ONNX Runtime uses the slower exact int8 matrix multiplication")

        manifest_path = hf_hub_download(MODEL_REPO, f"{bundle}/{MANIFEST_FILE}", revision=MODEL_REVISION)
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        check_manifest(manifest)
        self._max_frames = int(manifest["max_frames"])

        options = _session_options(ort, threads, cpu=device == "cpu")
        sessions = {
            stage: ort.InferenceSession(
                hf_hub_download(MODEL_REPO, f"{bundle}/{manifest[stage]}", revision=MODEL_REVISION),
                options,
                providers=providers,
            )
            for stage in STAGES
        }
        for stage, session in sessions.items():
            active = session.get_providers()
            if device == "cuda" and "CUDAExecutionProvider" not in active:
                msg = f"DEVICE=cuda, but ONNX Runtime can use only {active} for the {stage}. Check the container GPU."
                raise RuntimeError(msg)

        self._encoder = sessions["encoder"]
        self._prosody = sessions["prosody"]
        self._decoder = sessions["decoder"]
        self._voices = voices

    def synthesize(self, phonemes: str, voice: str, speed: float) -> np.ndarray:
        """Return mono float32 audio for one chunk of phonemes, in as many passes as the frame limit asks."""
        return self._speak(phonemes, self._voices[voice].reshape(1, VOICE_SIZE), speed)

    def _speak(self, phonemes: str, voice: np.ndarray, speed: float) -> np.ndarray:
        """Run one pass. When the encoder predicts more frames than the decoder takes, speak the parts instead."""
        encoded = self._encode(phonemes, voice, speed)
        if encoded.frames <= self._max_frames:
            return self._decode(encoded)

        count = len(tokenize(phonemes))
        parts = split_phonemes(phonemes, max(1, count * self._max_frames // encoded.frames))
        if len(parts) == 1:
            LOG.warning(
                "One word of %d tokens is longer than %d frames, so the server cuts it", count, self._max_frames
            )
            middle = len(phonemes) // 2
            parts = [phonemes[:middle], phonemes[middle:]]

        LOG.debug(
            "A pass of %d frames is above %d, so it is spoken in %d parts", encoded.frames, self._max_frames, len(parts)
        )
        return np.concatenate([self._speak(part, voice, speed) for part in parts])

    def _encode(self, phonemes: str, voice: np.ndarray, speed: float) -> _Encoded:
        tokens = np.array([tokenize(phonemes)], dtype=np.int64)
        acoustic, prosodic, durations = self._encoder.run(
            ["acoustic_tokens", "prosodic_tokens", "durations"],
            {"tokens": tokens, "voice": voice, "speed": np.array([speed], dtype=np.float32)},
        )
        durations = np.asarray(durations, dtype=np.int64)
        # the decoder takes one token index for each audio frame
        frame_to_token = np.repeat(np.arange(durations.shape[1], dtype=np.int64), durations[0])
        f0, energy = self._prosody.run(
            ["f0", "energy"],
            {"prosodic_tokens": prosodic, "frame_to_token": frame_to_token, "voice": voice},
        )
        return _Encoded(voice, np.asarray(acoustic), frame_to_token, np.asarray(f0), np.asarray(energy))

    def _decode(self, encoded: _Encoded) -> np.ndarray:
        if encoded.frames > self._max_frames:
            msg = f"The decoder takes at most {self._max_frames} frames, and this pass has {encoded.frames}"
            raise ValueError(msg)

        audio, lengths = self._decoder.run(
            ["audio", "audio_lengths"],
            {
                "acoustic_tokens": encoded.acoustic_tokens,
                "frame_to_token": encoded.frame_to_token,
                "f0": encoded.f0,
                "energy": encoded.energy,
                "voice": encoded.voice,
            },
        )
        return np.asarray(audio).reshape(-1)[: int(np.asarray(lengths).reshape(-1)[0])].astype(np.float32)
