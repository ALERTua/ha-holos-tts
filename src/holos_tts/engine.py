"""HolosTTS acoustic model in ONNX Runtime: phonemes and a voice vector in, 24 kHz audio out."""

from __future__ import annotations

import contextlib
import hashlib
import logging
import platform
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from huggingface_hub import hf_hub_download

from .cache_files import atomic_path, remove_path
from .constants import CHUNK_CHARS, SAMPLE_RATE

if TYPE_CHECKING:
    from types import ModuleType

LOG = logging.getLogger(__name__)

VOICE_SIZE = 256

MODEL_REPO = "patriotyk/HolosTTS"
# The author changed the model architecture between revisions, so the server pins the files it was tested with
MODEL_REVISION = "16acf36a9df3780786e71bb1a845003f2c56779f"
ONNX_FILES = {"cpu": "holos_cpu_int8.onnx", "cuda": "holos.onnx"}
VOICES_FILE = "voices.pt"
ONNX_CACHE_FOLDER = "onnx"
CPUINFO = Path("/proc/cpuinfo")
# ONNX Runtime: the U8S8 int8 matmul overflows on x86 without VNNI, and this entry selects the exact U8U8 kernels
QUANT_PRECISION_ENTRY = "session.x64quantprecision"
VNNI_FLAGS = frozenset({"avx_vnni", "avx512_vnni"})

# The symbol table of the HolosTTS checkpoint at MODEL_REVISION. The table has the apostrophe three times.
# The tokenizer of the checkpoint keeps the last index of a repeated symbol, and so does this one.
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

_NUMBER = re.compile(r"(\d+)")
# the cache files whose save failed in this process, so that a later load does not try and fail again
_FAILED_SAVES: set[Path] = set()


def tokenize(phonemes: str) -> list[int]:
    """Return the token ids of ``phonemes`` with the pad token on both ends. Unknown symbols drop out."""
    return [
        PAD_ID,
        *(TOKEN_IDS[symbol] for symbol in phonemes if symbol in TOKEN_IDS),
        PAD_ID,
    ]


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


def load_voices(voices_dir: Path) -> dict[str, np.ndarray]:
    """Return the voices of the model and the extra voices from ``voices_dir``. An extra voice wins on a name clash."""
    import torch  # noqa: PLC0415

    path = hf_hub_download(MODEL_REPO, VOICES_FILE, revision=MODEL_REVISION)
    voices = {name: _to_vector(value, name) for name, value in torch.load(path, map_location="cpu").items()}
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
    """Return the CPU feature line that the optimized graph depends on."""
    with contextlib.suppress(OSError):
        for line in CPUINFO.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("flags"):
                return line

    return platform.processor() + platform.machine()


def _needs_quant_precision() -> bool:
    """Return True on an x86 CPU without VNNI, where the fast int8 matmul of ONNX Runtime gives broken speech."""
    label, _, flags = _cpu_flags().partition(":")
    return label.strip() == "flags" and VNNI_FLAGS.isdisjoint(flags.split())


def _cache_key(model_file: str, ort_version: str) -> str:
    """Return the key of a cached graph: the graph depends on the model, the ONNX Runtime version and the CPU."""
    # a graph saved without the exact int8 matmul stays broken when a session with it loads the graph
    parts = (MODEL_REVISION, model_file, ort_version, _cpu_flags(), f"quant-precision={_needs_quant_precision()}")
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]


def _session_options(ort: ModuleType, threads: int, *, cpu: bool) -> Any:
    options = ort.SessionOptions()
    options.log_severity_level = 3
    if threads:
        options.intra_op_num_threads = threads

    if cpu:
        # the arena keeps the buffers of the longest chunk (about 300 MB), with no speed gain
        options.enable_cpu_mem_arena = False
        if _needs_quant_precision():
            options.add_session_config_entry(QUANT_PRECISION_ENTRY, "1")

    return options


def _remove_temp_files(folder: Path, stem: str) -> None:
    """Remove the temp files of ``stem`` that a killed process left in ``folder``, also the old ``.tmp`` files."""
    for pattern in (f"{stem}.*.partial", f"{stem}.*.tmp"):
        for leftover in folder.glob(pattern):
            remove_path(leftover)


def _save_optimized_graph(ort: ModuleType, path: str, cached: Path, *, threads: int) -> bool:
    """
    Open the original graph and save the optimized graph to ``cached``. Return True when saved.

    After a failed save, this function skips the same ``cached`` until the process ends.
    """
    if cached in _FAILED_SAVES:
        return False

    stem = cached.name.rsplit(".", 2)[0]
    try:
        started = time.monotonic()
        with atomic_path(cached) as temp:
            _remove_temp_files(cached.parent, stem)
            options = _session_options(ort, threads, cpu=True)
            options.optimized_model_filepath = str(temp)
            # a saving session keeps about 281 MB extra for its life, so the caller reopens the saved file
            ort.InferenceSession(path, options, providers=["CPUExecutionProvider"])
    except Exception:
        _FAILED_SAVES.add(cached)
        LOG.warning("Cannot open the original graph %s to save its optimized copy to %s", path, cached, exc_info=True)
        return False
    else:
        LOG.info("Wrote the optimized graph to %s in %.1f s", cached, time.monotonic() - started)
        # the other keys are graphs of an older ONNX Runtime or of another CPU
        for stale in cached.parent.glob(f"{stem}.*.onnx"):
            if stale != cached:
                remove_path(stale)

        return True


def _open_cached_graph(ort: ModuleType, cached: Path, *, threads: int) -> Any:
    options = _session_options(ort, threads, cpu=True)
    # the cached graph is optimized already, and a second optimization only slows down the load
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(str(cached), options, providers=["CPUExecutionProvider"])


def _cpu_session(ort: ModuleType, path: str, model_file: str, *, threads: int, cache_dir: Path) -> Any:
    """Open the CPU session from the cached optimized graph. When there is none, save one and open it."""
    cached = cache_dir / ONNX_CACHE_FOLDER / f"{Path(model_file).stem}.{_cache_key(model_file, ort.__version__)}.onnx"
    if cached.is_file():
        try:
            return _open_cached_graph(ort, cached, threads=threads)
        except Exception:
            LOG.warning("Cannot load the cached graph %s, so the server removes it", cached, exc_info=True)
            with contextlib.suppress(OSError):
                cached.unlink(missing_ok=True)

    if _save_optimized_graph(ort, path, cached, threads=threads):
        try:
            return _open_cached_graph(ort, cached, threads=threads)
        except Exception:
            LOG.warning("Cannot load the saved graph %s, so the server removes it", cached, exc_info=True)
            with contextlib.suppress(OSError):
                cached.unlink(missing_ok=True)

    # no usable cache: open the original graph
    return ort.InferenceSession(path, _session_options(ort, threads, cpu=True), providers=["CPUExecutionProvider"])


class HolosEngine:
    """One ONNX Runtime session of the HolosTTS model."""

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

        model_file = ONNX_FILES[device]
        LOG.info("Loading the HolosTTS model %s on %s", model_file, device)
        if device == "cpu" and _needs_quant_precision():
            LOG.info("The CPU has no VNNI, so ONNX Runtime uses the slower exact int8 matrix multiplication")
        path = hf_hub_download(MODEL_REPO, model_file, revision=MODEL_REVISION)
        if device == "cpu" and cache_dir is not None:
            self._session = _cpu_session(ort, path, model_file, threads=threads, cache_dir=cache_dir)
        else:
            options = _session_options(ort, threads, cpu=device == "cpu")
            self._session = ort.InferenceSession(path, options, providers=providers)

        active = self._session.get_providers()
        if device == "cuda" and "CUDAExecutionProvider" not in active:
            msg = f"DEVICE=cuda, but ONNX Runtime can use only {active}. Check the GPU of the container."
            raise RuntimeError(msg)

        self._voices = voices

    def synthesize(self, phonemes: str, voice: str, speed: float) -> np.ndarray:
        """Return mono float32 audio for one chunk of phonemes."""
        tokens = np.array([tokenize(phonemes)], dtype=np.int64)
        audio, lengths = self._session.run(
            None,
            {
                "tokens": tokens,
                "voice": self._voices[voice].reshape(1, VOICE_SIZE),
                "speed": np.array([speed], dtype=np.float32),
            },
        )
        return np.asarray(audio).reshape(-1)[: int(np.asarray(lengths).reshape(-1)[0])].astype(np.float32)
