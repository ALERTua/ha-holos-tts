"""
Wyoming TTS server for Home Assistant.

A ``synthesize`` event or a ``synthesize-start``/``-chunk``/``-stop`` stream comes in.
``audio-start``, ``audio-chunk`` and ``audio-stop`` go out, and a stream ends with ``synthesize-stopped``.
"""

from __future__ import annotations

import contextlib
import logging
import re
from typing import TYPE_CHECKING

from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.error import Error
from wyoming.info import Attribution, Describe, Info, TtsProgram, TtsVoice
from wyoming.server import AsyncEventHandler
from wyoming.tts import (
    Synthesize,
    SynthesizeChunk,
    SynthesizeStart,
    SynthesizeStop,
    SynthesizeStopped,
)

from .audio import fit_level, to_pcm16
from .constants import LANGUAGE, MODEL_URL, PROGRAM_NAME, SAMPLE_RATE
from .text import SENTENCE_END_MARKS

if TYPE_CHECKING:
    import asyncio

    import numpy as np
    from wyoming.event import Event

    from .synthesizer import Synthesizer

LOG = logging.getLogger(__name__)
SAMPLE_WIDTH = 2
CHANNELS = 1
SAMPLES_PER_CHUNK = 2048
# In a text stream, everything up to the last end of sentence is ready to speak
_LAST_SENTENCE_END = re.compile(rf"[{SENTENCE_END_MARKS}](?=\s)(?!.*[{SENTENCE_END_MARKS}]\s)", re.DOTALL)
ATTRIBUTION = Attribution(name="patriotyk", url=MODEL_URL)


async def build_info(synthesizer: Synthesizer, version: str) -> Info:
    """Describe the program and its voices for Home Assistant."""
    voices = [
        TtsVoice(
            name=name,
            description=name,
            attribution=ATTRIBUTION,
            installed=True,
            version=None,
            languages=[LANGUAGE],
        )
        for name in await synthesizer.voices()
    ]
    program = TtsProgram(
        name=PROGRAM_NAME,
        description="Ukrainian HolosTTS with verbalizer and stress model",
        attribution=ATTRIBUTION,
        installed=True,
        version=version,
        voices=voices,
        supports_synthesize_streaming=True,
    )
    return Info(tts=[program])


def split_ready_text(buffer: str) -> tuple[str, str]:
    """Split a stream buffer into the complete sentences and the unfinished rest."""
    match = _LAST_SENTENCE_END.search(buffer)
    if match is None:
        return "", buffer

    return buffer[: match.end()], buffer[match.end() :]


class TtsEventHandler(AsyncEventHandler):
    """One Home Assistant connection."""

    def __init__(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        *,
        synthesizer: Synthesizer,
        version: str,
    ) -> None:
        super().__init__(reader, writer)
        self.synthesizer = synthesizer
        self.version = version
        self._streaming = False
        self._stream_voice: str | None = None
        self._stream_buffer = ""
        self._audio_started = False

    async def handle_event(self, event: Event) -> bool:
        """Answer one event of the client."""
        try:
            return await self._handle(event)
        except Exception as e:
            if self.writer.is_closing():
                LOG.info("The Wyoming client disconnected during %s", event.type)
                return False

            LOG.exception("Failed to handle the Wyoming event %s", event.type)
            with contextlib.suppress(ConnectionError, RuntimeError, TypeError):
                await self.write_event(Error(text=str(e), code=type(e).__name__).event())

            return False

    async def _handle(self, event: Event) -> bool:  # noqa: PLR0911 - one return for each event type
        if Describe.is_type(event.type):
            await self.write_event((await build_info(self.synthesizer, self.version)).event())
            return True

        if SynthesizeStart.is_type(event.type):
            start = SynthesizeStart.from_event(event)
            self._streaming = True
            self._stream_voice = start.voice.name if start.voice else None
            self._stream_buffer = ""
            self._audio_started = False
            # the models load while the client writes the text, so the first sentence does not wait for all of them
            self.synthesizer.request_warm_up()
            return True

        if SynthesizeChunk.is_type(event.type):
            self._stream_buffer += SynthesizeChunk.from_event(event).text
            ready, self._stream_buffer = split_ready_text(self._stream_buffer)
            if ready.strip():
                await self._speak(ready, self._stream_voice)

            return True

        if SynthesizeStop.is_type(event.type):
            if self._stream_buffer.strip():
                await self._speak(self._stream_buffer, self._stream_voice)

            await self._finish_audio()
            await self.write_event(SynthesizeStopped().event())
            self._streaming = False
            return True

        if Synthesize.is_type(event.type):
            if self._streaming:
                # inside a stream the client repeats the whole text only for older servers
                return True

            synthesize = Synthesize.from_event(event)
            self._audio_started = False
            await self._speak(synthesize.text, synthesize.voice.name if synthesize.voice else None)
            await self._finish_audio()
            return True

        return True

    async def _speak(self, text: str, voice: str | None) -> None:
        speed = self.synthesizer.settings.default_speed
        async with contextlib.aclosing(self.synthesizer.synthesize(text, voice, speed)) as chunks:
            async for audio in chunks:
                await self._send_audio(audio)

    async def _send_audio(self, audio: np.ndarray) -> None:
        if not self._audio_started:
            await self.write_event(AudioStart(rate=SAMPLE_RATE, width=SAMPLE_WIDTH, channels=CHANNELS).event())
            self._audio_started = True

        pcm = to_pcm16(fit_level(audio))
        step = SAMPLES_PER_CHUNK * SAMPLE_WIDTH * CHANNELS
        for offset in range(0, len(pcm), step):
            await self.write_event(
                AudioChunk(
                    audio=pcm[offset : offset + step],
                    rate=SAMPLE_RATE,
                    width=SAMPLE_WIDTH,
                    channels=CHANNELS,
                ).event()
            )

    async def _finish_audio(self) -> None:
        if not self._audio_started:
            # Home Assistant expects a start before the stop, also for a text without speech
            await self.write_event(AudioStart(rate=SAMPLE_RATE, width=SAMPLE_WIDTH, channels=CHANNELS).event())

        await self.write_event(AudioStop().event())
        self._audio_started = False
