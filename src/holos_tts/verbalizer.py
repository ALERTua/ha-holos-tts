"""
Verbalizer: an M2M100 model that writes numbers, dates, units and acronyms as Ukrainian words.

The CTranslate2 build of the model needs only the sentencepiece tokenizer, so the server does not load the
transformers library for it. The token rules repeat the ``M2M100Tokenizer`` of transformers.
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import ctranslate2
import sentencepiece
from huggingface_hub import hf_hub_download, snapshot_download

from .cache_files import partials
from .ct2_int8 import convert_to_int8
from .text import halve_for_verbalizer

if TYPE_CHECKING:
    from collections.abc import Callable

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
# The model learned to end each output at this length with its language token, so a longer output is cut
MAX_OUTPUT_TOKENS = 127
# Without a word before it, the model reads a part such as "1333, 1370," as one number or as ordinals,
# so a later part that starts with a number gets "і" before it, and the output loses that "і" again
_STARTS_WITH_NUMBER = re.compile(r"[+\-−]?\d")
_CONTEXT = "і "
_CONTEXT_IN_OUTPUT = re.compile(r"^[ійІЙ]\s+")
# the model ends each part as a sentence, so a part that ends at one of these marks gets the mark back
_CLAUSE_MARKS = ",;:—–"


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


def verbalize_in_parts(text: str, translate: Callable[[str], tuple[str, bool]]) -> str:
    """
    Verbalize ``text`` whole, and verbalize its two halves instead when the model cut the output.

    ``translate`` returns the words of one model pass and whether the model ended the output itself.
    """
    return _verbalize_part(text, translate, first=True, last=True)


def _verbalize_part(part: str, translate: Callable[[str], tuple[str, bool]], *, first: bool, last: bool) -> str:
    """Verbalize one part of a text. A part in the middle of the text keeps the case and the marks of that place."""
    context = _CONTEXT if not first and _STARTS_WITH_NUMBER.match(part) else ""
    words, complete = translate(context + part)
    if not complete:
        halves = halve_for_verbalizer(part)
        if len(halves) > 1:
            LOG.debug("The verbalizer cut its output for %r, so it gets the two halves of the text", part)
            left = _verbalize_part(halves[0], translate, first=first, last=False)
            right = _verbalize_part(halves[1], translate, first=False, last=last)
            return f"{left} {right}"

        LOG.warning("The verbalizer cut its output for %r, and the text has no place to split", part)

    if context:
        words = _CONTEXT_IN_OUTPUT.sub("", words, count=1)
    if not first and not part[:1].isupper():
        words = words[:1].lower() + words[1:]
    if not last and words.endswith(".") and not part.endswith("."):
        words = words[:-1] + (part[-1] if part[-1] in _CLAUSE_MARKS else "")

    return words


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
        return verbalize_in_parts(text, self._translate)

    def _translate(self, text: str) -> tuple[str, bool]:
        """Return the words of one model pass, and whether the model ended the output before the length limit."""
        pieces = [piece if piece in self._vocab else UNKNOWN_TOKEN for piece in self._sp.encode(text, out_type=str)]
        result = self._translator.translate_batch(
            [[LANGUAGE_TOKEN, *pieces, END_TOKEN]],
            target_prefix=[[LANGUAGE_TOKEN]],
            beam_size=1,
            num_hypotheses=1,
        )
        hypothesis = result[0].hypotheses[0]
        tokens = [
            token
            for token in hypothesis[1:]
            if token not in SPECIAL_TOKENS and not (token.startswith("__") and token.endswith("__"))
        ]
        return self._sp.decode(tokens).strip(), len(hypothesis) < MAX_OUTPUT_TOKENS
