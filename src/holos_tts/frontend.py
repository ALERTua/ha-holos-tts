"""Turn a chunk of sentences into the phoneme string of the acoustic model: verbalizer, stress model, IPA."""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING
from unicodedata import normalize

from .text import needs_verbalization, recover_stress, strip_stress

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

LOG = logging.getLogger(__name__)

_DASHES = re.compile(r"[᠆‐‑‒–—―⁻₋−⸺⸻]")
_END_PUNCTUATION = ".?!:-"


class Phonemizer:
    """Stress model (stanza) and IPA converter of the HolosTTS and StyleTTS2 demos."""

    def __init__(self, threads: int = 0, *, cache_dir: Path | None = None) -> None:
        # stanza and torch are slow to import, so the worker process imports them only here
        from ipa_uk import ipa  # noqa: PLC0415
        from ukrainian_word_stress import Stressifier  # noqa: PLC0415

        from . import stanza_cache  # noqa: PLC0415

        if threads:
            import torch  # noqa: PLC0415 - stanza runs on torch

            torch.set_num_threads(threads)

        if cache_dir is not None:
            stanza_cache.install(cache_dir)

        LOG.info("Loading the stress model")
        self._ipa = ipa
        self._stressify = Stressifier()
        self._stressify("Тест.")

    def __call__(self, text: str) -> str:
        """Return the IPA string of ``text``, which already went through the verbalizer."""
        text = text.strip().replace('"', "")
        if not text:
            return ""

        text = normalize("NFKC", text)
        text = _DASHES.sub("-", text)
        if text[-1] not in _END_PUNCTUATION:
            text += "."

        text = text.replace(" - ", ": ")
        return self._ipa(self._stressify(text))


def verbalize_sentence(sentence: str, verbalizer: Callable[[str], str] | None) -> str:
    """Rewrite numbers and symbols of one sentence as words and keep the stress marks of the user."""
    if verbalizer is None or not needs_verbalization(sentence):
        return sentence

    verbalized = verbalizer(strip_stress(sentence)).strip()
    if not verbalized:
        LOG.warning(
            "The verbalizer returned nothing for %r, using the sentence as it is",
            sentence,
        )
        return sentence

    LOG.debug("Verbalized %r -> %r", sentence, verbalized)
    return recover_stress(sentence, verbalized)
