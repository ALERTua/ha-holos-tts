"""
OpenAI-compatible speech API, for the openai_tts integration of Home Assistant and for Music Assistant.

https://platform.openai.com/docs/api-reference/audio/createSpeech
"""

from __future__ import annotations

import contextlib
import logging
from typing import TYPE_CHECKING, Annotated, Any, Literal

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict, Field

from .audio import FORMATS, UnsupportedFormatError, encode
from .constants import MAX_SPEED, MIN_SPEED, PROGRAM_NAME, SAMPLE_RATE
from .synthesizer import SynthesisError

if TYPE_CHECKING:
    from .synthesizer import Synthesizer

LOG = logging.getLogger(__name__)
MODEL_NAME = "holos"


class SpeechRequest(BaseModel):
    """Body of ``POST /v1/audio/speech``. The server ignores fields that it does not know."""

    model_config = ConfigDict(extra="ignore")

    input: Annotated[
        str,
        Field(description="Ukrainian text. Mark a stress with + after the vowel or ` before it."),
    ]
    model: str | None = Field(None, description="Accepted for compatibility. The server has one model.")
    voice: str | None = Field(
        None,
        description="A name from GET /v1/audio/voices. Unknown names use the default.",
    )
    response_format: str = Field("mp3", description=f"One of: {', '.join(FORMATS)}.")
    speed: float | None = Field(
        None,
        ge=0.25,
        le=4.0,
        description=f"1.0 is normal speed. The model uses {MIN_SPEED} to {MAX_SPEED}.",
    )


class VoicesResponse(BaseModel):
    """Body of ``GET /v1/audio/voices``."""

    voices: list[str]


class WarmUpResponse(BaseModel):
    """Body of ``POST /v1/warmup``."""

    status: Literal["started", "running"]


def create_app(synthesizer: Synthesizer) -> FastAPI:
    """Build the HTTP application around ``synthesizer``."""
    app = FastAPI(
        title=PROGRAM_NAME,
        description="Ukrainian HolosTTS speech with an OpenAI-compatible API",
    )

    @app.post("/v1/audio/speech", response_class=Response)
    async def create_speech(body: SpeechRequest) -> Response:
        if not body.input.strip():
            raise HTTPException(status_code=400, detail="input is empty")

        if body.response_format not in FORMATS:
            raise HTTPException(
                status_code=400,
                detail=f"response_format {body.response_format!r} is not supported. Use one of: {', '.join(FORMATS)}.",
            )

        speed = body.speed if body.speed is not None else synthesizer.settings.default_speed
        speed = min(max(speed, MIN_SPEED), MAX_SPEED)
        parts: list[np.ndarray] = []
        try:
            async with contextlib.aclosing(synthesizer.synthesize(body.input, body.voice, speed)) as chunks:
                parts.extend([chunk async for chunk in chunks])
        except SynthesisError as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

        audio = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
        try:
            content, media_type = encode(audio, SAMPLE_RATE, body.response_format)
        except UnsupportedFormatError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

        return Response(content=content, media_type=media_type)

    @app.get("/v1/audio/voices")
    async def list_voices() -> VoicesResponse:
        try:
            return VoicesResponse(voices=await synthesizer.voices())
        except SynthesisError as e:
            raise HTTPException(status_code=503, detail=str(e)) from e

    @app.post("/v1/warmup", status_code=202)
    async def warm_up() -> WarmUpResponse:
        # async, so the warm-up task starts in the event loop
        return WarmUpResponse(status="started" if synthesizer.request_warm_up() else "running")

    @app.get("/v1/models")
    async def list_models() -> dict[str, Any]:
        return {
            "object": "list",
            "data": [{"id": MODEL_NAME, "object": "model", "owned_by": "patriotyk"}],
        }

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {"status": "ok", **await synthesizer.status()}

    if synthesizer.settings.web_ui:
        # the import is lazy, so a server without the web page does not load gradio
        from .web_ui import mount  # noqa: PLC0415

        mount(app, synthesizer)

    return app
