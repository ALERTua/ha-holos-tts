"""
Verbalizer: an M2M100 model that writes numbers, dates, units and acronyms as Ukrainian words.

The CTranslate2 build of the model needs only the sentencepiece tokenizer, so the server does not load the
transformers library for it. The token rules repeat the ``M2M100Tokenizer`` of transformers.
"""

from __future__ import annotations

import contextlib
import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any

import ctranslate2
import sentencepiece
from huggingface_hub import hf_hub_download, snapshot_download

from .cache_files import partials
from .ct2_int8 import convert_to_int8

LOG = logging.getLogger(__name__)

MODEL_REPO = "skypro1111/m2m100-ukr-verbalization-ct2"
MODEL_REVISION = "ff98ecaacef2058527ed69f381cfe65d40a67333"
TOKENIZER_REPO = "skypro1111/m2m100-ukr-verbalization"
TOKENIZER_REVISION = "87757c5b2b06b0c6546f8f65fc0f08a3b5c5051b"
LANGUAGE_TOKEN = "__uk__"  # noqa: S105 - a token of the model vocabulary, not a secret
END_TOKEN = "</s>"  # noqa: S105
UNKNOWN_TOKEN = "<unk>"  # noqa: S105
SPECIAL_TOKENS = frozenset({"<s>", END_TOKEN, "<pad>", UNKNOWN_TOKEN})
INT8_CACHE_FOLDER = "verbalizer-int8"
# float16 needs a GPU of compute capability 7.0 or newer, so an older GPU takes the next type
CUDA_COMPUTE_TYPES = ("int8_float16", "int8_float32", "float32")


def _compute_type(device: str) -> str:
    """Return int8 on the CPU, and the first type of ``CUDA_COMPUTE_TYPES`` that the GPU supports on CUDA."""
    if device == "cpu":
        return "int8"

    supported = ctranslate2.get_supported_compute_types(device)
    compute_type = next((name for name in CUDA_COMPUTE_TYPES if name in supported), CUDA_COMPUTE_TYPES[-1])
    if compute_type != CUDA_COMPUTE_TYPES[0]:
        LOG.info("The GPU does not support %s, the verbalizer uses %s", CUDA_COMPUTE_TYPES[0], compute_type)

    return compute_type


def _remove_other_copies(int8_dir: Path) -> None:
    """Remove the int8 copies of other model revisions. The folder of a running conversion of this revision stays."""
    keep = {int8_dir, *partials(int8_dir)}
    with contextlib.suppress(OSError):
        for other in int8_dir.parent.iterdir():
            if other not in keep and other.is_dir():
                shutil.rmtree(other, ignore_errors=True)


def _load_translator(model_dir: Path, cache_dir: Path | None, options: dict[str, Any]) -> ctranslate2.Translator:
    """Load the int8 copy of the model from the cache, and make the copy when it is missing."""
    if cache_dir is not None:
        int8_dir = cache_dir / INT8_CACHE_FOLDER / MODEL_REVISION[:12]
        try:
            if not int8_dir.is_dir():
                started = time.monotonic()
                convert_to_int8(model_dir, int8_dir)
                LOG.info("Wrote the int8 verbalizer model to %s in %.1f s", int8_dir, time.monotonic() - started)
                _remove_other_copies(int8_dir)

            return ctranslate2.Translator(str(int8_dir), **options)
        except (OSError, RuntimeError, ValueError):
            LOG.warning(
                "Cannot use the int8 verbalizer model in %s, the original model loads instead", int8_dir, exc_info=True
            )
            shutil.rmtree(int8_dir, ignore_errors=True)

    return ctranslate2.Translator(str(model_dir), **options)


class Verbalizer:
    """Rewrite one sentence with its numbers and symbols as words."""

    def __init__(self, device: str = "cpu", threads: int = 0, cache_dir: Path | None = None) -> None:
        LOG.info("Loading the verbalizer model on %s", device)
        model_dir = snapshot_download(MODEL_REPO, revision=MODEL_REVISION, allow_patterns=["*.bin", "*.json"])
        sp_model = hf_hub_download(TOKENIZER_REPO, "sentencepiece.bpe.model", revision=TOKENIZER_REVISION)
        vocab_file = hf_hub_download(TOKENIZER_REPO, "vocab.json", revision=TOKENIZER_REVISION)
        self._vocab = frozenset(json.loads(Path(vocab_file).read_text(encoding="utf-8")))
        self._sp = sentencepiece.SentencePieceProcessor(model_file=sp_model)
        options = {
            "device": device,
            "compute_type": _compute_type(device),
            "intra_threads": threads,
        }
        self._translator = _load_translator(Path(model_dir), cache_dir, options)

    def __call__(self, text: str) -> str:
        """Return ``text`` with its numbers and symbols written as words."""
        pieces = [piece if piece in self._vocab else UNKNOWN_TOKEN for piece in self._sp.encode(text, out_type=str)]
        result = self._translator.translate_batch(
            [[LANGUAGE_TOKEN, *pieces, END_TOKEN]],
            target_prefix=[[LANGUAGE_TOKEN]],
            beam_size=1,
            num_hypotheses=1,
        )
        tokens = [
            token
            for token in result[0].hypotheses[0][1:]
            if token not in SPECIAL_TOKENS and not (token.startswith("__") and token.endswith("__"))
        ]
        return self._sp.decode(tokens)
