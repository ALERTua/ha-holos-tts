"""
Worker process: it holds the verbalizer, the stress model and the acoustic model.

The main process starts the worker and talks to it through a pipe. The heavy libraries (torch, stanza, ONNX Runtime,
CTranslate2) live only here, so the main process stays small. After idle time the worker drops the models but keeps the
imported libraries (about 0.4 GB). Only a stopped worker gives all of its memory back to the system.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

import psutil

from .constants import MODEL_PARTS
from .engine import HolosEngine, load_voices, order_voices
from .frontend import Phonemizer, verbalize_sentence
from .memory import model_loading, release_free_memory
from .text import needs_verbalization
from .verbalizer import Verbalizer

if TYPE_CHECKING:
    import numpy as np

    from .config import Settings

LOG = logging.getLogger(__name__)
# the property that loads each model, by the name of the model in ``status``
_PART_PROPERTIES = dict(zip(MODEL_PARTS, ("phonemizer", "engine", "verbalizer"), strict=True))


class Worker:
    """Load each model on first use and run the requests of the main process."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._voices: dict[str, np.ndarray] | None = None
        self._phonemizer: Phonemizer | None = None
        self._engine: HolosEngine | None = None
        self._verbalizer: Verbalizer | None = None
        self._verbalizer_used = 0.0

    @property
    def voices(self) -> dict[str, np.ndarray]:
        """Voice vectors by name."""
        if self._voices is None:
            self._voices = load_voices(self.settings.voices_dir)

        return self._voices

    def voice_names(self) -> list[str]:
        """Voice names with the default voice first."""
        return order_voices(set(self.voices), self.settings.default_voice)

    @property
    def phonemizer(self) -> Phonemizer:
        """Stress model and IPA converter."""
        if self._phonemizer is None:
            with model_loading():
                self._phonemizer = Phonemizer(self.settings.threads, cache_dir=self.settings.cache_dir)

        return self._phonemizer

    @property
    def engine(self) -> HolosEngine:
        """Acoustic model."""
        if self._engine is None:
            with model_loading():
                self._engine = HolosEngine(
                    self.voices,
                    self.settings.device,
                    threads=self.settings.threads,
                    cache_dir=self.settings.cache_dir,
                )

        return self._engine

    @property
    def verbalizer(self) -> Verbalizer | None:
        """Verbalizer, or None when the settings turn it off."""
        if not self.settings.verbalize:
            return None

        if self._verbalizer is None:
            with model_loading():
                self._verbalizer = Verbalizer(
                    self.settings.verbalizer_device,
                    self.settings.threads,
                    self.settings.cache_dir,
                )

        self._verbalizer_used = time.monotonic()
        return self._verbalizer

    def load(self) -> None:
        """Load each model now instead of on the first request."""
        _ = self.phonemizer, self.engine, self.verbalizer

    def load_part(self, name: str) -> None:
        """Load one model now: "stress", "engine" or "verbalizer". The verbalizer loads only when the settings ask."""
        if name not in _PART_PROPERTIES:
            msg = f"Unknown model {name!r}"
            raise ValueError(msg)

        getattr(self, _PART_PROPERTIES[name])

    def unload_idle_verbalizer(self) -> None:
        """Drop the verbalizer when nobody used it for the configured time."""
        timeout = self.settings.verbalizer_unload_after_seconds
        if self._verbalizer is None or not timeout:
            return

        if time.monotonic() - self._verbalizer_used >= timeout:
            LOG.info("Unloading the verbalizer after %s s without use", timeout)
            self._verbalizer = None
            release_free_memory()

    def unload(self) -> None:
        """Drop the three models. The voice vectors stay, and the next request loads the models again."""
        self._phonemizer = self._engine = self._verbalizer = None
        release_free_memory()

    def synthesize(self, sentences: list[str], voice: str, speed: float) -> np.ndarray:
        """Return the audio of one chunk of sentences."""
        if voice not in self.voices:
            msg = f"Unknown voice {voice!r}"
            raise KeyError(msg)

        # the verbalizer loads only for a chunk that has something to rewrite
        verbalizer = self.verbalizer if any(needs_verbalization(sentence) for sentence in sentences) else None
        text = " ".join(verbalize_sentence(sentence, verbalizer) for sentence in sentences)
        phonemes = self.phonemizer(text)
        LOG.debug("Phonemes of %r: %s", text, phonemes)
        if not phonemes:
            import numpy as np  # noqa: PLC0415

            return np.zeros(0, dtype=np.float32)

        return self.engine.synthesize(phonemes, voice, speed)

    def status(self) -> dict[str, Any]:
        """Which models are in memory, and the memory of this process."""
        return {
            "verbalizer": self._verbalizer is not None,
            "stress": self._phonemizer is not None,
            "engine": self._engine is not None,
            "rss_mb": round(psutil.Process().memory_info().rss / 2**20),
        }

    def handle(self, message: tuple[Any, ...]) -> Any:
        """Run one request of the main process."""
        command, *args = message
        if command == "voices":
            return self.voice_names()

        if command == "load":
            self.load()
            return self.status()

        if command == "load_part":
            self.load_part(*args)
            return self.status()

        if command == "unload":
            LOG.info("Unloading the models")
            self.unload()
            return self.status()

        if command == "synth":
            audio = self.synthesize(*args)
            # each request leaves freed buffers of the models behind
            release_free_memory()
            return audio

        if command == "status":
            return self.status()

        msg = f"Unknown worker command {command!r}"
        raise ValueError(msg)
