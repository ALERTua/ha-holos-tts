"""Gradio web page of the server. ``create_app`` mounts it at ``/web`` when ``WEB_UI`` is on."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import gradio as gr
import numpy as np

from .audio import fit_level, to_pcm16
from .constants import MAX_SPEED, MIN_SPEED, SAMPLE_RATE
from .synthesizer import SynthesisError

if TYPE_CHECKING:
    from fastapi import FastAPI

    from .synthesizer import Synthesizer

SPEED_STEP = 0.05
WEB_PATH = "/web"
# every Speak makes a WAV file in the temp folder of gradio; the page deletes the files older than an hour, hourly
CACHE_CHECK_SECONDS = 3600
CACHE_MAX_AGE_SECONDS = 3600


async def speak(synthesizer: Synthesizer, text: str, voice: str | None, *, speed: float) -> tuple[int, np.ndarray]:
    """Return the sample rate and the 16-bit samples of ``text``, made the same way as for the HTTP API."""
    if not text.strip():
        msg = "Text is empty."
        raise gr.Error(msg)

    parts: list[np.ndarray] = []
    try:
        async with contextlib.aclosing(synthesizer.synthesize(text, voice, speed)) as chunks:
            parts.extend([chunk async for chunk in chunks])
    except SynthesisError as e:
        raise gr.Error(str(e)) from e

    audio = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
    # Gradio would scale a float array to full scale, so the page converts it the same way as the WAV file of the API
    return SAMPLE_RATE, np.frombuffer(to_pcm16(fit_level(audio)), dtype="<i2")


async def verbalize(synthesizer: Synthesizer, text: str) -> str:
    """Return ``text`` with numbers, dates, units and acronyms written as words."""
    try:
        return await synthesizer.verbalize(text)
    except SynthesisError as e:
        raise gr.Error(str(e)) from e


async def load_voices(synthesizer: Synthesizer) -> gr.Dropdown:
    """Return the voice list with the default voice selected. The page asks for it when it opens."""
    try:
        voices = await synthesizer.voices()
        selected = await synthesizer.resolve_voice(None)
    except SynthesisError as e:
        raise gr.Error(str(e)) from e

    return gr.Dropdown(choices=voices, value=selected)


def create_ui(synthesizer: Synthesizer) -> gr.Blocks:
    """Build the page around ``synthesizer``."""
    default_voice = synthesizer.settings.default_voice

    async def handle_speak(text: str, voice: str | None, speed: float) -> tuple[int, np.ndarray]:
        return await speak(synthesizer, text, voice, speed=speed)

    async def handle_verbalize(text: str) -> str:
        return await verbalize(synthesizer, text)

    async def handle_load() -> gr.Dropdown:
        return await load_voices(synthesizer)

    with gr.Blocks(title="HolosTTS", delete_cache=(CACHE_CHECK_SECONDS, CACHE_MAX_AGE_SECONDS)) as ui:
        text = gr.Textbox(label="Text", lines=5, placeholder="Ukrainian text. Mark a stress with + after the vowel.")
        # API clients skip the page load that fills the list, so the dropdown accepts any name
        voice = gr.Dropdown(label="Voice", choices=[default_voice], value=default_voice, allow_custom_value=True)
        speed = gr.Slider(
            label="Speed",
            minimum=MIN_SPEED,
            maximum=MAX_SPEED,
            step=SPEED_STEP,
            value=synthesizer.settings.default_speed,
        )
        with gr.Row():
            speak_button = gr.Button("Speak", variant="primary")
            verbalize_button = gr.Button("Verbalize")
        audio = gr.Audio(label="Speech", type="numpy", interactive=False)
        speak_button.click(handle_speak, inputs=[text, voice, speed], outputs=audio, api_name="speak")
        # the words replace the text, so the user can edit them before "Speak"
        verbalize_button.click(handle_verbalize, inputs=text, outputs=text, api_name="verbalize")
        ui.load(handle_load, outputs=voice, api_name=False)

    return ui


def mount(app: FastAPI, synthesizer: Synthesizer) -> None:
    """Serve the page of ``synthesizer`` at ``WEB_PATH`` of ``app``."""
    gr.mount_gradio_app(app, create_ui(synthesizer), path=WEB_PATH)
