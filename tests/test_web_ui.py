import asyncio
import contextlib
import json
import subprocess
import sys

import gradio as gr
import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from holos_tts.openai_api import create_app
from holos_tts.synthesizer import Synthesizer
from holos_tts.web_ui import load_voices, speak, verbalize
from tests import fake_worker
from tests.conftest import make_settings


def test_gradio_is_not_imported_when_the_web_ui_is_off():
    code = (
        "import sys\n"
        "from holos_tts.config import Settings\n"
        "from holos_tts.openai_api import create_app\n"
        "from holos_tts.synthesizer import Synthesizer\n"
        "create_app(Synthesizer(Settings.from_env({}), 20))\n"
        "print('gradio' in sys.modules)\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.strip() == "False"


@pytest.fixture
async def web_synthesizer():
    synth = Synthesizer(make_settings(preload=False, web_ui=True), chunk_chars=20, worker_target=fake_worker.run)
    yield synth
    with contextlib.suppress(Exception):
        await synth.stop()


async def test_web_page_is_served_at_web_and_the_api_stays(web_synthesizer):
    app = create_app(web_synthesizer)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test", follow_redirects=True) as client:
        page = await client.get("/web/")
        assert page.status_code == 200
        assert page.headers["content-type"].startswith("text/html")
        assert "gradio" in page.text.lower()
        assert (await client.get("/v1/audio/voices")).json() == {"voices": fake_worker.VOICES}


async def test_web_page_is_not_served_when_the_web_ui_is_off(synthesizer):
    transport = httpx.ASGITransport(app=create_app(synthesizer))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        assert (await client.get("/web/")).status_code == 404


async def test_speak_returns_the_rate_and_the_samples(synthesizer):
    rate, audio = await speak(synthesizer, "Привіт.", "Speaker_0", speed=1.0)
    assert rate == 24000
    assert audio.dtype == np.int16
    assert len(audio) == len("Привіт.")


async def test_speak_scales_a_loud_sound_down(synthesizer):
    _rate, audio = await speak(synthesizer, "LOUD", None, speed=1.0)
    assert audio[:2].tolist() == [32767, 16383]


async def test_speak_shows_a_worker_error_as_a_gradio_error(synthesizer):
    with pytest.raises(gr.Error, match="bad text"):
        await speak(synthesizer, "FAIL", None, speed=1.0)


async def test_speak_refuses_an_empty_text(synthesizer):
    with pytest.raises(gr.Error, match="empty"):
        await speak(synthesizer, "  ", None, speed=1.0)


async def test_voice_list_has_the_default_voice_selected(synthesizer):
    dropdown = await load_voices(synthesizer)
    assert [value for _label, value in dropdown.choices] == fake_worker.VOICES
    assert dropdown.value == fake_worker.VOICES[0]


def test_speak_endpoint_takes_a_voice_that_the_page_has_not_listed():
    synth = Synthesizer(make_settings(preload=False, web_ui=True), chunk_chars=20, worker_target=fake_worker.run)
    with TestClient(create_app(synth)) as client:
        call = client.post("/web/gradio_api/call/speak", json={"data": ["Привіт.", "Speaker_0", 1.0]})
        assert call.status_code == 200
        result = client.get(f"/web/gradio_api/call/speak/{call.json()['event_id']}")
        assert "event: complete" in result.text
        asyncio.run(synth.stop())


async def test_verbalize_returns_the_text_with_words(synthesizer):
    assert await verbalize(synthesizer, "Лишилось 7.") == "Лишилось сім."


async def test_verbalize_shows_a_worker_error_as_a_gradio_error(synthesizer):
    with pytest.raises(gr.Error, match="bad text"):
        await verbalize(synthesizer, "FAIL")


def test_verbalize_endpoint_completes_with_the_new_text():
    synth = Synthesizer(make_settings(preload=False, web_ui=True), chunk_chars=20, worker_target=fake_worker.run)
    with TestClient(create_app(synth)) as client:
        call = client.post("/web/gradio_api/call/verbalize", json={"data": ["Лишилось 7."]})
        assert call.status_code == 200
        result = client.get(f"/web/gradio_api/call/verbalize/{call.json()['event_id']}")
        assert "event: complete" in result.text
        assert json.loads(result.text.split("data: ", 1)[1]) == ["Лишилось сім."]
        asyncio.run(synth.stop())
