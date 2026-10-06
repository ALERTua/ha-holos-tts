import asyncio
import io
import json
import random
import re
import time
import wave
from typing import Any

import httpx
import numpy as np
import pytest
import soundfile as sf
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.client import AsyncTcpClient
from wyoming.info import Describe, Info
from wyoming.server import AsyncTcpServer
from wyoming.tts import (
    Synthesize,
    SynthesizeChunk,
    SynthesizeStart,
    SynthesizeStop,
    SynthesizeStopped,
    SynthesizeVoice,
)

from holos_tts.openai_api import create_app
from holos_tts.text import SENTENCE_END_MARKS, group_sentences, prepare_sentences
from holos_tts.wyoming_server import TtsEventHandler, split_ready_text
from tests import fake_worker


@pytest.fixture
async def http(synthesizer):
    transport = httpx.ASGITransport(app=create_app(synthesizer))
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


async def test_speech_returns_mp3_by_default(http):
    response = await http.post(
        "/v1/audio/speech",
        json={"model": "tts-1", "input": "Привіт, світе.", "voice": "x"},
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/mpeg"
    _audio, rate = sf.read(io.BytesIO(response.content))
    assert rate == 24000


async def test_speech_wav_has_the_samples_of_the_worker(http):
    response = await http.post(
        "/v1/audio/speech",
        json={"input": "Привіт.", "response_format": "wav", "speed": 1.0},
    )
    with wave.open(io.BytesIO(response.content)) as wav:
        assert wav.getnframes() == len("Привіт.")


async def test_speech_refuses_empty_input_and_unknown_format(http):
    assert (await http.post("/v1/audio/speech", json={"input": " "})).status_code == 400
    assert (await http.post("/v1/audio/speech", json={"input": "Так.", "response_format": "aac"})).status_code == 400


async def asgi_speech_call(app, text, *, leave_when):
    """Call the speech endpoint as a server does and return its status. The client leaves when ``leave_when`` is set."""
    body = json.dumps({"input": text, "response_format": "pcm"}).encode()
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": "POST",
        "path": "/v1/audio/speech",
        "raw_path": b"/v1/audio/speech",
        "query_string": b"",
        "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())],
        "client": ("127.0.0.1", 1),
        "server": ("test", 80),
        "scheme": "http",
        "root_path": "",
    }
    sent_body = False
    messages = []

    async def receive() -> dict[str, Any]:
        nonlocal sent_body
        if not sent_body:
            sent_body = True
            return {"type": "http.request", "body": body, "more_body": False}

        if leave_when.is_set():
            return {"type": "http.disconnect"}

        await asyncio.sleep(3600)  # a server waits here until the client leaves; the check of the app cancels it
        return {}

    async def send(message: dict[str, Any]) -> None:
        messages.append(message)

    await app(scope, receive, send)
    return next(message["status"] for message in messages if message["type"] == "http.response.start")


def spy_on_chunks(synthesizer, client_left=None):
    """Return the list of the chunks that go to the worker. The first chunk sets ``client_left``."""
    sent = []
    request = synthesizer._request

    async def spy(command: str, *args: Any) -> Any:
        if command == "synth":
            sent.append(args[0])
            if client_left is not None:
                client_left.set()

        return await request(command, *args)

    synthesizer._request = spy
    return sent


async def test_speech_stops_sending_chunks_to_the_worker_when_the_client_leaves(synthesizer):
    text = " ".join(["Раз."] * 30)
    chunks = group_sentences(prepare_sentences(text), synthesizer.chunk_chars)
    assert len(chunks) > 3
    client_left = asyncio.Event()
    sent = spy_on_chunks(synthesizer, client_left)
    status = await asgi_speech_call(create_app(synthesizer), text, leave_when=client_left)
    assert status == 499
    # the client left while the first chunk ran, and that chunk may finish
    assert sent == chunks[:1]


async def test_speech_sends_all_chunks_to_the_worker_while_the_client_stays(synthesizer):
    text = " ".join(["Раз."] * 30)
    sent = spy_on_chunks(synthesizer)
    status = await asgi_speech_call(create_app(synthesizer), text, leave_when=asyncio.Event())
    assert status == 200
    assert sent == group_sentences(prepare_sentences(text), synthesizer.chunk_chars)


async def test_speech_reports_a_worker_error(http):
    response = await http.post("/v1/audio/speech", json={"input": "FAIL"})
    assert response.status_code == 500
    assert "bad text" in response.json()["detail"]


async def test_voices_and_health(http):
    assert (await http.get("/v1/audio/voices")).json() == {"voices": fake_worker.VOICES}
    health = (await http.get("/health")).json()
    assert health["status"] == "ok"


@pytest.mark.parametrize(
    ("buffer", "ready", "rest"),
    [
        ("Перше. Друге", "Перше.", " Друге"),
        ("Перше. Друге! Тре", "Перше. Друге!", " Тре"),
        ("Без кінця", "", "Без кінця"),
        ("Кінець.", "", "Кінець."),
        ("Один.\nДва? Три", "Один.\nДва?", " Три"),
        ("А… Б: в", "А… Б:", " в"),
        ("Крапка.", "", "Крапка."),
        ("", "", ""),
    ],
)
def test_split_ready_text(buffer, ready, rest):
    assert split_ready_text(buffer) == (ready, rest)


# the regex that split_ready_text used before it became linear
_OLD_LAST_SENTENCE_END = re.compile(rf"[{SENTENCE_END_MARKS}](?=\s)(?!.*[{SENTENCE_END_MARKS}]\s)", re.DOTALL)


def test_split_ready_text_gives_the_results_of_the_old_regex_on_random_texts():
    rng = random.Random(7)  # noqa: S311
    alphabet = [*SENTENCE_END_MARKS, "а", "б", ",", " ", " ", "\n", "\t", " ", " ", "\x1f", "-"]
    for _ in range(20000):
        buffer = "".join(rng.choices(alphabet, k=rng.randint(0, 14)))
        match = _OLD_LAST_SENTENCE_END.search(buffer)
        expected = ("", buffer) if match is None else (buffer[: match.end()], buffer[match.end() :])
        assert split_ready_text(buffer) == expected, repr(buffer)


def test_split_ready_text_of_a_long_buffer_takes_linear_time():
    # many sentence ends and a long tail without marks took minutes with the old regex
    head = "Так. " * 15_000
    started = time.perf_counter()
    assert split_ready_text(head + "а" * 125_000) == (head[:-1], " " + "а" * 125_000)
    assert time.perf_counter() - started < 2


@pytest.fixture
async def wyoming(synthesizer):
    server = AsyncTcpServer("127.0.0.1", 0)
    await server.start(lambda reader, writer: TtsEventHandler(reader, writer, synthesizer=synthesizer, version="1.2.3"))
    port = server._server.sockets[0].getsockname()[1]
    yield port
    await server.stop()


async def read_until(client, stop_type):
    events = []
    while True:
        event = await asyncio.wait_for(client.read_event(), 30)
        events.append(event)
        if stop_type.is_type(event.type):
            return events


async def test_wyoming_describe(wyoming):
    async with AsyncTcpClient("127.0.0.1", wyoming) as client:
        await client.write_event(Describe().event())
        info = Info.from_event(await client.read_event())

    program = info.tts[0]
    assert program.version == "1.2.3"
    assert program.supports_synthesize_streaming
    assert [voice.name for voice in program.voices] == fake_worker.VOICES
    assert program.voices[0].languages == ["uk"]


async def test_wyoming_synthesize(wyoming):
    async with AsyncTcpClient("127.0.0.1", wyoming) as client:
        await client.write_event(Synthesize(text="Привіт.", voice=SynthesizeVoice(name="Speaker_0")).event())
        events = await read_until(client, AudioStop)

    assert AudioStart.is_type(events[0].type)
    pcm = b"".join(AudioChunk.from_event(e).audio for e in events if AudioChunk.is_type(e.type))
    assert len(pcm) == 2 * len("Привіт.")


async def test_wyoming_stream_speaks_each_finished_sentence(wyoming):
    async with AsyncTcpClient("127.0.0.1", wyoming) as client:
        await client.write_event(SynthesizeStart().event())
        for part in ("Перше ", "речення. Дру", "ге речення"):
            await client.write_event(SynthesizeChunk(text=part).event())

        # Home Assistant repeats the whole text inside the stream for older servers; it must not be spoken twice
        await client.write_event(Synthesize(text="Перше речення. Друге речення").event())
        await client.write_event(SynthesizeStop().event())
        events = await read_until(client, SynthesizeStopped)

    types = [e.type for e in events]
    assert types.count("audio-start") == 1
    assert types.count("audio-stop") == 1
    pcm = b"".join(AudioChunk.from_event(e).audio for e in events if AudioChunk.is_type(e.type))
    assert len(pcm) == 2 * (len("Перше речення.") + len("Друге речення"))


async def test_wyoming_error_event(wyoming):
    async with AsyncTcpClient("127.0.0.1", wyoming) as client:
        await client.write_event(Synthesize(text="FAIL").event())
        event = await asyncio.wait_for(client.read_event(), 30)

    assert event.type == "error"
    assert "bad text" in event.data["text"]


async def test_speed_is_limited_to_the_model_range(http):
    response = await http.post("/v1/audio/speech", json={"input": "Так.", "response_format": "pcm", "speed": 3.5})
    # the fake worker returns samples equal to speed / 10
    assert np.frombuffer(response.content, dtype="<i2")[0] == int(0.2 * 32767)


async def test_wyoming_scales_loud_audio_down(wyoming):
    async with AsyncTcpClient("127.0.0.1", wyoming) as client:
        await client.write_event(Synthesize(text="LOUD.").event())
        events = await read_until(client, AudioStop)

    pcm = b"".join(AudioChunk.from_event(e).audio for e in events if AudioChunk.is_type(e.type))
    assert np.frombuffer(pcm, dtype="<i2")[:2].tolist() == [32767, 16383]


async def test_wyoming_empty_text_still_starts_and_stops_audio(wyoming):
    async with AsyncTcpClient("127.0.0.1", wyoming) as client:
        await client.write_event(Synthesize(text="   ").event())
        events = await read_until(client, AudioStop)

    assert [e.type for e in events] == ["audio-start", "audio-stop"]


async def test_wyoming_plain_synthesize_after_a_stream_is_spoken(wyoming):
    async with AsyncTcpClient("127.0.0.1", wyoming) as client:
        await client.write_event(SynthesizeStart().event())
        await client.write_event(SynthesizeChunk(text="Потік.").event())
        await client.write_event(SynthesizeStop().event())
        await read_until(client, SynthesizeStopped)
        await client.write_event(Synthesize(text="Окремо.").event())
        events = await read_until(client, AudioStop)

    pcm = b"".join(AudioChunk.from_event(e).audio for e in events if AudioChunk.is_type(e.type))
    assert len(pcm) == 2 * len("Окремо.")


async def test_wyoming_stream_start_requests_one_warm_up(wyoming, synthesizer, monkeypatch):
    calls = []
    monkeypatch.setattr(synthesizer, "request_warm_up", lambda: calls.append("warm up") or True)
    async with AsyncTcpClient("127.0.0.1", wyoming) as client:
        await client.write_event(SynthesizeStart().event())
        await client.write_event(SynthesizeChunk(text="Потік.").event())
        await client.write_event(SynthesizeStop().event())
        await read_until(client, SynthesizeStopped)
        await client.write_event(Synthesize(text="Окремо.").event())
        await read_until(client, AudioStop)

    assert calls == ["warm up"]


async def test_warmup_answers_at_once_while_the_worker_is_busy(http, synthesizer):
    # the held lock stops each load of the warm-up, so an answer here proves that the endpoint does not wait for them
    async with synthesizer._lock:
        first = await asyncio.wait_for(http.post("/v1/warmup"), 5)
        second = await asyncio.wait_for(http.post("/v1/warmup"), 5)

    assert (first.status_code, first.json()) == (202, {"status": "started"})
    assert (second.status_code, second.json()) == (202, {"status": "running"})
