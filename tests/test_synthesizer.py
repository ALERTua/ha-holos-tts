import asyncio
import contextlib
import gc
import logging
import random

import pytest

from holos_tts import synthesizer as synthesizer_module
from holos_tts.synthesizer import SynthesisError, Synthesizer
from tests import fake_worker
from tests.conftest import make_settings

# the time for a request to reach the worker; two of them end before a "SLOW" request does
REQUEST_PAUSE_SECONDS = fake_worker.SLOW_SECONDS / 5


async def collect(synth, text, voice=None, speed=1.0):
    async with contextlib.aclosing(synth.synthesize(text, voice, speed)) as chunks:
        return [chunk async for chunk in chunks]


async def test_voices_come_from_the_worker(synthesizer):
    assert await synthesizer.voices() == fake_worker.VOICES


async def test_unknown_voice_falls_back_to_the_default_of_the_settings():
    # the default of the settings is not the first voice of the worker, so the test sees which one wins
    synth = Synthesizer(
        make_settings(default_voice=fake_worker.VOICES[1]), chunk_chars=20, worker_target=fake_worker.run
    )
    try:
        assert await synth.resolve_voice("alloy") == fake_worker.VOICES[1]
        assert await synth.resolve_voice(None) == fake_worker.VOICES[1]
        assert await synth.resolve_voice(fake_worker.VOICES[0]) == fake_worker.VOICES[0]
    finally:
        await synth.stop()


async def test_slow_reader_does_not_block_other_requests(synthesizer):
    chunks = synthesizer.synthesize("Перше довге речення тут. Друге довге речення тут.", None, 1.0)
    try:
        await anext(chunks)
        # the first request now waits for its reader; a second request must get through
        assert await asyncio.wait_for(collect(synthesizer, "Інший запит."), 10)
    finally:
        await chunks.aclose()


async def test_cancelled_request_does_not_leave_its_reply_in_the_pipe(synthesizer):
    await synthesizer.voices()
    task = asyncio.create_task(collect(synthesizer, "SLOW.", speed=1.0))
    await asyncio.sleep(REQUEST_PAUSE_SECONDS)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

    chunks = await collect(synthesizer, "Наступний запит.", speed=2.0)
    # the fake worker returns speed / 10, so 0.2 is the reply of this request, 0.1 the reply of the cancelled one
    assert chunks[0][0] == pytest.approx(0.2)


async def test_hung_worker_times_out_and_restarts(synthesizer, monkeypatch):
    # the short limit covers only the hung request: neither the start of the first worker nor the start of the next one
    await synthesizer.voices()
    reply_timeouts = synthesizer_module.REPLY_TIMEOUT_SECONDS
    monkeypatch.setattr(synthesizer_module, "REPLY_TIMEOUT_SECONDS", {"synth": 0.3})
    with pytest.raises(SynthesisError, match="stopped unexpectedly"):
        await collect(synthesizer, "HANG.")

    monkeypatch.setattr(synthesizer_module, "REPLY_TIMEOUT_SECONDS", reply_timeouts)
    assert not synthesizer.worker_running
    assert await collect(synthesizer, "Після зависання.")


async def test_status_does_not_wait_for_a_busy_worker(synthesizer):
    await synthesizer.voices()
    task = asyncio.create_task(collect(synthesizer, "SLOW."))
    await asyncio.sleep(REQUEST_PAUSE_SECONDS)
    assert await asyncio.wait_for(synthesizer.status(), 0.2) == {"worker": "busy"}
    await task


async def test_synthesize_yields_one_array_per_chunk(synthesizer):
    chunks = await collect(synthesizer, "Перше речення тут. Друге речення тут. Третє.", speed=1.5)
    # a chunk takes the next sentence while it is not longer than 20 characters;
    # the fake worker returns one sample per character of the chunk
    assert [len(chunk) for chunk in chunks] == [
        len("Перше речення тут. Друге речення тут."),
        len("Третє."),
    ]
    assert chunks[0][0] == pytest.approx(0.15)


async def test_worker_error_becomes_a_synthesis_error(synthesizer):
    with pytest.raises(SynthesisError, match="bad text"):
        await collect(synthesizer, "FAIL")

    assert await collect(synthesizer, "Знову працює.")


async def test_worker_crash_starts_a_new_worker(synthesizer):
    await synthesizer.voices()
    first = (await synthesizer.status())["pid"]
    with pytest.raises(SynthesisError, match="stopped unexpectedly"):
        await collect(synthesizer, "CRASH")

    assert not synthesizer.worker_running
    assert await collect(synthesizer, "Після збою.")
    assert (await synthesizer.status())["pid"] != first


async def test_status_does_not_start_a_stopped_worker(synthesizer):
    assert await synthesizer.status() == {"worker": "stopped"}
    assert not synthesizer.worker_running


CHECK_SECONDS = 0.05
LONG_IDLE_SECONDS = 100


@pytest.fixture
async def idle_synth(monkeypatch):
    """A started synthesizer with a loaded fake worker and a fast idle check."""
    monkeypatch.setattr(synthesizer_module, "IDLE_CHECK_SECONDS", CHECK_SECONDS)
    synth = Synthesizer(
        make_settings(preload=True, unload_after_seconds=1),
        chunk_chars=20,
        worker_target=fake_worker.run,
    )
    await synth.start()
    await synth._warm_up_task
    yield synth
    await synth.stop()


def make_idle(synth):
    """Act as if the synthesizer got no request for a long time."""
    synth._last_used -= LONG_IDLE_SECONDS


async def wait_for_unloads(synth, count):
    for _ in range(100):
        await asyncio.sleep(CHECK_SECONDS)
        if (await synth.status()).get("unloads") == count:
            return

    pytest.fail(f"The worker did not get {count} unload request(s)")


async def test_idle_worker_unloads_its_models_once_and_keeps_running(idle_synth):
    first = await idle_synth.status()
    assert first["engine"] is True
    make_idle(idle_synth)
    await wait_for_unloads(idle_synth, 1)

    # more checks go by, and nothing happened since the unload
    await asyncio.sleep(CHECK_SECONDS * 6)
    after = await idle_synth.status()
    assert after["unloads"] == 1
    assert after["engine"] is False
    assert idle_synth.worker_running
    assert after["pid"] == first["pid"]


async def test_request_after_an_unload_allows_the_next_unload(idle_synth):
    make_idle(idle_synth)
    await wait_for_unloads(idle_synth, 1)
    pid = (await idle_synth.status())["pid"]
    assert await collect(idle_synth, "Знову.")
    make_idle(idle_synth)
    await wait_for_unloads(idle_synth, 2)
    assert (await idle_synth.status())["pid"] == pid


async def test_no_unload_while_a_request_holds_the_lock(idle_synth):
    make_idle(idle_synth)
    # the synthesis holds the lock longer than one check; its end sets the time of the last request
    assert await collect(idle_synth, "SLOW.")
    await asyncio.sleep(CHECK_SECONDS * 6)
    assert (await idle_synth.status())["unloads"] == 0


async def test_request_that_ends_while_the_idle_check_waits_for_the_lock_is_not_followed_by_an_unload(monkeypatch):
    # a check at each turn of the event loop sees the free lock before the waiting request takes it
    monkeypatch.setattr(synthesizer_module, "IDLE_CHECK_SECONDS", 0)
    synth = Synthesizer(
        make_settings(preload=False, unload_after_seconds=1),
        chunk_chars=20,
        worker_target=fake_worker.run,
    )
    try:
        await synth.start()
        # a holder that does not change the time of the last request, such as a health check
        await synth._lock.acquire()
        make_idle(synth)
        request = asyncio.create_task(collect(synth, "Запит."))
        for _ in range(3):
            await asyncio.sleep(0)

        synth._lock.release()
        assert await request
        await asyncio.sleep(CHECK_SECONDS * 6)
        assert (await synth.status())["unloads"] == 0
    finally:
        await synth.stop()


async def test_failed_unload_is_not_repeated_and_does_not_stop_the_idle_timer(monkeypatch, caplog):
    monkeypatch.setattr(synthesizer_module, "IDLE_CHECK_SECONDS", CHECK_SECONDS)
    synth = Synthesizer(
        make_settings(preload=False, unload_after_seconds=1),
        chunk_chars=20,
        worker_target=fake_worker.refusing_unload_worker,
    )
    try:
        await synth.start()
        make_idle(synth)
        with caplog.at_level(logging.ERROR, logger=synthesizer_module.__name__):
            for _ in range(100):
                await asyncio.sleep(CHECK_SECONDS)
                if caplog.records:
                    break

        await asyncio.sleep(CHECK_SECONDS * 6)
        assert (await synth.status())["unloads"] == 1
        assert not synth._idle_task.done()
        assert synth.worker_running
    finally:
        await synth.stop()


async def test_stop_stops_the_worker_process(idle_synth):
    assert idle_synth.worker_running
    process = idle_synth._process
    await idle_synth.stop()
    assert not idle_synth.worker_running
    assert not process.is_alive()


async def test_second_cancel_makes_the_next_request_use_a_new_worker(synthesizer):
    await synthesizer.voices()
    first = (await synthesizer.status())["pid"]
    task = asyncio.create_task(collect(synthesizer, "SLOW.", speed=1.0))
    await asyncio.sleep(REQUEST_PAUSE_SECONDS)
    task.cancel()
    await asyncio.sleep(REQUEST_PAUSE_SECONDS)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task

    chunks = await collect(synthesizer, "Наступний запит.", speed=2.0)
    assert chunks[0][0] == pytest.approx(0.2)
    assert (await synthesizer.status())["pid"] != first


@pytest.fixture
async def loop_errors():
    """Contexts that reach the exception handler of the event loop, such as "Task exception was never retrieved"."""
    loop = asyncio.get_running_loop()
    errors = []
    previous = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: errors.append(context))
    yield errors
    loop.set_exception_handler(previous)


async def cancel_twice(task, pause):
    await asyncio.sleep(pause)
    task.cancel()
    await asyncio.sleep(pause)
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


async def test_requests_after_a_second_cancel_get_their_own_replies(synthesizer, loop_errors):
    await synthesizer.voices()
    for _ in range(3):
        await cancel_twice(asyncio.create_task(collect(synthesizer, "SLOW.", speed=1.0)), REQUEST_PAUSE_SECONDS)
        # the thread of the cancelled request still waits for its slow reply while these requests start
        results = await asyncio.gather(
            collect(synthesizer, "Наступний.", speed=2.0),
            collect(synthesizer, "Ще один.", speed=3.0),
        )
        # the fake worker returns speed / 10, so each request sees whose reply it got
        assert [chunks[0][0] for chunks in results] == pytest.approx([0.2, 0.3])

    gc.collect()
    assert [context["message"] for context in loop_errors] == []


async def test_random_second_cancels_do_not_mix_up_replies(synthesizer, loop_errors, caplog):
    caplog.set_level(logging.WARNING, logger=synthesizer_module.__name__)
    await synthesizer.voices()
    rng = random.Random(7)  # noqa: S311
    wrong = []
    for round_ in range(12):
        speeds = [round_ + index / 10 for index in range(1, rng.randint(2, 4))]
        tasks = [asyncio.create_task(collect(synthesizer, "JITTER.", speed=speed)) for speed in speeds]
        await cancel_twice(rng.choice(tasks), rng.uniform(0, 0.02))
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for speed, result in zip(speeds, results, strict=True):
            # only the cancelled request may end without audio
            if isinstance(result, BaseException):
                assert isinstance(result, asyncio.CancelledError), result
            elif result[0][0] != pytest.approx(speed / 10):
                wrong.append((speed, float(result[0][0] * 10)))

    assert wrong == []
    gc.collect()
    assert [context["message"] for context in loop_errors] == []
    # the rounds above must reach the case under test: a second cancel while the worker still runs the request
    restarts = [record for record in caplog.records if "unknown state" in record.getMessage()]
    assert len(restarts) >= 3


@pytest.fixture
async def warm_synth():
    synth = Synthesizer(make_settings(preload=False), chunk_chars=20, worker_target=fake_worker.warm_up_worker)
    await synth.voices()
    yield synth
    await synth.stop()


async def worker_log(synth):
    return (await synth.status())["log"]


async def test_warm_up_loads_each_model_in_order(warm_synth):
    assert warm_synth.request_warm_up() is True
    await warm_synth._warm_up_task
    assert await worker_log(warm_synth) == [
        "voices",
        "load_part stress",
        "load_part engine",
        "load_part verbalizer",
    ]


async def test_warm_up_skips_a_verbalizer_that_has_its_own_idle_timeout():
    settings = make_settings(preload=False, verbalizer_unload_after_seconds=45)
    synth = Synthesizer(settings, chunk_chars=20, worker_target=fake_worker.warm_up_worker)
    try:
        await synth.voices()
        assert synth.request_warm_up() is True
        await synth._warm_up_task
        assert await worker_log(synth) == ["voices", "load_part stress", "load_part engine"]
    finally:
        await synth.stop()


async def test_only_one_warm_up_runs_at_a_time(warm_synth):
    assert warm_synth.request_warm_up() is True
    assert warm_synth.request_warm_up() is False
    await warm_synth._warm_up_task
    assert warm_synth.request_warm_up() is True
    await warm_synth._warm_up_task
    assert (await worker_log(warm_synth)).count("load_part stress") == 2


async def test_synthesis_runs_between_two_warm_up_steps(warm_synth):
    warm_synth.request_warm_up()
    # the warm-up takes the free lock on its first step, so the synthesis below waits behind that step only
    await asyncio.sleep(0)
    assert warm_synth._lock.locked()
    assert await collect(warm_synth, "Так.")
    await warm_synth._warm_up_task
    assert await worker_log(warm_synth) == [
        "voices",
        "load_part stress",
        "synth",
        "load_part engine",
        "load_part verbalizer",
    ]


async def test_cancelled_warm_up_leaves_the_pipe_usable(warm_synth):
    warm_synth.request_warm_up()
    await asyncio.sleep(fake_worker.LOAD_SECONDS / 3)
    warm_synth._warm_up_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await warm_synth._warm_up_task

    chunks = await collect(warm_synth, "Так.", speed=2.0)
    # the reply of the cancelled load is a dict, so an array with speed / 10 is the reply of this synthesis
    assert chunks[0][0] == pytest.approx(0.2)
    assert await worker_log(warm_synth) == ["voices", "load_part stress", "synth"]


async def test_stop_cancels_a_running_warm_up(warm_synth):
    warm_synth.request_warm_up()
    await asyncio.sleep(fake_worker.LOAD_SECONDS / 3)
    task = warm_synth._warm_up_task
    await warm_synth.stop()
    assert task.cancelled()
    assert not warm_synth.worker_running


async def test_failed_warm_up_step_is_logged_and_stops_the_warm_up(synthesizer, caplog, monkeypatch):
    # the fake worker answers a model name that it does not know with an error
    monkeypatch.setattr(synthesizer_module, "WARM_UP_PARTS", ("unknown", "stress"))
    await synthesizer.voices()
    synthesizer.request_warm_up()
    with caplog.at_level(logging.ERROR, logger=synthesizer_module.__name__):
        await synthesizer._warm_up_task

    assert synthesizer._warm_up_task.exception() is None
    assert [record.getMessage() for record in caplog.records] == ["Failed to load the unknown model in advance"]
    assert await collect(synthesizer, "Після помилки.")


async def test_start_with_preload_returns_before_the_models_are_loaded():
    synth = Synthesizer(make_settings(preload=True), chunk_chars=20, worker_target=fake_worker.warm_up_worker)
    try:
        await synth.start()
        assert not synth._warm_up_task.done()
        await synth._warm_up_task
        assert await worker_log(synth) == [
            "voices",
            "load_part stress",
            "load_part engine",
            "load_part verbalizer",
        ]
    finally:
        await synth.stop()


async def test_synthesis_right_after_start_waits_only_for_the_model_that_loads_now():
    synth = Synthesizer(make_settings(preload=True), chunk_chars=20, worker_target=fake_worker.warm_up_worker)
    try:
        await synth.start()
        # the warm-up takes the free lock for its first step, as it does when a request comes a moment later
        await asyncio.sleep(0)
        assert await collect(synth, "Так.")
        await synth._warm_up_task
        assert await worker_log(synth) == [
            "voices",
            "load_part stress",
            "synth",
            "load_part engine",
            "load_part verbalizer",
        ]
    finally:
        await synth.stop()
