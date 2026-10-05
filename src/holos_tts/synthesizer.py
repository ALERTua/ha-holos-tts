"""
Main-process side of the worker: start it, send it requests one at a time, unload its models after idle time.

An idle worker drops its models and gives their memory back to the system, but it keeps the imported libraries
(about 0.4 GB). The next request loads the models again. A stopped worker gives all of its memory back.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import multiprocessing
import time
from typing import TYPE_CHECKING, Any

from . import worker_entry
from .constants import MODEL_PARTS
from .text import group_sentences, prepare_sentences

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator, Callable
    from multiprocessing.connection import Connection
    from multiprocessing.process import BaseProcess

    import numpy as np

    from .config import Settings

LOG = logging.getLogger(__name__)
IDLE_CHECK_SECONDS = 5.0
STOP_TIMEOUT_SECONDS = 10.0
# a hung worker must not hold the lock forever; "voices", "load" and "load_part" can download, so they have no limit
REPLY_TIMEOUT_SECONDS = {"synth": 600.0, "status": 10.0, "unload": 10.0}
WARM_UP_PARTS = MODEL_PARTS


class SynthesisError(RuntimeError):
    """The worker could not make the audio."""


class Synthesizer:
    """Async interface to the worker process."""

    def __init__(
        self,
        settings: Settings,
        chunk_chars: int,
        worker_target: Callable[[Connection, Settings], None] = worker_entry.run,
    ) -> None:
        self.settings = settings
        self.chunk_chars = chunk_chars
        self._worker_target = worker_target
        self._context = multiprocessing.get_context("spawn")
        self._process: BaseProcess | None = None
        self._conn: Connection | None = None
        self._lock = asyncio.Lock()
        self._voices: list[str] | None = None
        self._last_used = time.monotonic()
        self._idle_task: asyncio.Task[None] | None = None
        self._warm_up_task: asyncio.Task[None] | None = None
        # the exchange of a cancelled request that did not end well; its thread may still read the pipe
        self._abandoned: asyncio.Future[tuple[str, Any]] | None = None
        # set after an idle unload, cleared by any request, so an idle worker gets one "unload" and not one per check
        self._unloaded = False

    @property
    def worker_running(self) -> bool:
        """Tell if a worker process runs now."""
        return self._process is not None and self._process.is_alive()

    async def start(self) -> None:
        """Read the voice list and, when the settings ask for it, start the warm-up."""
        if self.settings.unload_after_seconds:
            self._idle_task = asyncio.create_task(self._unload_when_idle(), name="unload idle worker")

        await self.voices()
        if self.settings.preload:
            # one model at a time, so a request right after the start waits only for the model that loads now
            self.request_warm_up()

    async def stop(self) -> None:
        """Stop the idle timer, the warm-up and the worker."""
        for task in (self._idle_task, self._warm_up_task):
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

        async with self._lock:
            await self._end_abandoned_exchange()
            await asyncio.to_thread(self._stop_worker)

    def request_warm_up(self) -> bool:
        """Start to load the models in the background. Return False when a warm-up already runs."""
        if self._warm_up_task is not None and not self._warm_up_task.done():
            return False

        self._warm_up_task = asyncio.create_task(self._warm_up(), name="warm up worker")
        return True

    async def voices(self) -> list[str]:
        """Voice names, with the default voice first. The list stays in memory after the worker stops."""
        if self._voices is None:
            self._voices = await self._request("voices")

        return self._voices

    async def resolve_voice(self, name: str | None) -> str:
        """Return ``name`` when the model has it, else the default voice."""
        voices = await self.voices()
        if name in voices:
            return name

        fallback = self.settings.default_voice if self.settings.default_voice in voices else voices[0]
        if name:
            LOG.warning(
                "Unknown voice %r, using %r. Known voices: %s",
                name,
                fallback,
                ", ".join(voices),
            )

        return fallback

    async def status(self) -> dict[str, Any]:
        """State of the worker and its models. This call does not start a stopped worker."""
        if not self.worker_running:
            return {"worker": "stopped"}

        if self._lock.locked():
            # a health check must not wait for a long synthesis
            return {"worker": "busy"}

        async with self._lock:
            status = await self._send("status")

        return {"worker": "running", **status}

    async def synthesize(self, text: str, voice: str | None, speed: float) -> AsyncGenerator[np.ndarray, None]:
        """Yield the audio of ``text`` chunk by chunk, so that a caller can send the first chunk early."""
        chunks = group_sentences(prepare_sentences(text), self.chunk_chars)
        resolved = await self.resolve_voice(voice)
        LOG.info(
            "Synthesizing %d chunk(s) with voice %r at speed %s: %r",
            len(chunks),
            resolved,
            speed,
            text,
        )
        for chunk in chunks:
            # the lock covers one chunk, so a client that reads its audio slowly does not stop other requests
            audio = await self._request("synth", chunk, resolved, speed)
            yield audio

    async def verbalize(self, text: str) -> str:
        """Return ``text`` with numbers, dates, units and acronyms written as words."""
        sentences = prepare_sentences(text)
        if not sentences:
            return ""

        return await self._request("verbalize", sentences)

    async def _warm_up(self) -> None:
        # one request for each model, so a synthesis that waits for the lock runs between two loads
        status = None
        # a verbalizer with its own idle timeout loads only for a text that needs it
        skip = {"verbalizer"} if self.settings.verbalizer_unload_after_seconds else set()
        for part in (part for part in WARM_UP_PARTS if part not in skip):
            try:
                status = await self._request("load_part", part)
            except Exception:
                # a warm-up is only a hint: the next synthesis loads the models again and reports the error
                LOG.exception("Failed to load the %s model in advance", part)
                return

        LOG.info("Models are loaded in advance: %s", status)

    async def _request(self, command: str, *args: Any) -> Any:
        async with self._lock:
            try:
                return await self._send(command, *args)
            finally:
                self._last_used = time.monotonic()
                # the request may have loaded the models again
                self._unloaded = False

    async def _send(self, command: str, *args: Any) -> Any:
        """
        Send one request and wait for its reply. The caller holds the lock.

        A cancelled caller still waits for the reply, else the next request would read this reply from the pipe.
        """
        await self._end_abandoned_exchange()
        exchange = asyncio.ensure_future(asyncio.to_thread(self._exchange, (command, *args)))
        try:
            status, value = await asyncio.shield(exchange)
        except asyncio.CancelledError:
            # a second cancel stops this wait; the next request then ends the exchange before it uses the worker
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.wait({exchange})

            if not exchange.done() or exchange.cancelled() or exchange.exception() is not None:
                self._abandoned = exchange
                exchange.add_done_callback(_retrieve_exception)

            raise
        except Exception as e:
            # after any failure of the pipe its state is unknown, so the next request gets a new worker
            LOG.exception("The worker process failed while it ran %r", command)
            await asyncio.to_thread(self._stop_worker)
            msg = "The worker process stopped unexpectedly. Read the log of the container."
            raise SynthesisError(msg) from e

        if status != "ok":
            raise SynthesisError(value)

        return value

    async def _end_abandoned_exchange(self) -> None:
        """
        Wait until the thread of an abandoned exchange ends, then stop its worker. The caller holds the lock.

        A thread that reads a closed pipe can take the reply of the next request, so the pipe closes after the thread.
        """
        abandoned = self._abandoned
        if abandoned is None:
            return

        LOG.warning("A cancelled request left the worker in an unknown state, stopping the worker")
        # the worker gets as much time to finish the cancelled request as a stop gives it
        while not (await asyncio.wait({abandoned}, timeout=STOP_TIMEOUT_SECONDS))[0]:
            LOG.warning("The worker did not finish the cancelled request in %s s, killing it", STOP_TIMEOUT_SECONDS)
            # the blocking read of the abandoned exchange ends only when its worker ends
            if self._process is not None:
                self._process.kill()

        await asyncio.to_thread(self._stop_worker)
        # cleared only now, so a caller cancelled above leaves the same work to the next request
        self._abandoned = None

    def _exchange(self, message: tuple[Any, ...]) -> tuple[str, Any]:
        conn = self._ensure_worker()
        conn.send(message)
        timeout = REPLY_TIMEOUT_SECONDS.get(message[0])
        if timeout is not None and not conn.poll(timeout):
            msg = f"The worker did not answer {message[0]!r} in {timeout} s"
            raise TimeoutError(msg)

        return conn.recv()

    def _ensure_worker(self) -> Connection:
        if self.worker_running and self._conn is not None:
            return self._conn

        self._stop_worker()
        LOG.info("Starting the worker process")
        parent_conn, child_conn = self._context.Pipe()
        process = self._context.Process(
            target=self._worker_target,
            args=(child_conn, self.settings),
            name="holos-tts-worker",
            daemon=True,
        )
        process.start()
        child_conn.close()
        self._process, self._conn = process, parent_conn
        return parent_conn

    def _stop_worker(self) -> None:
        process, conn = self._process, self._conn
        self._process = self._conn = None
        if conn is not None:
            with contextlib.suppress(EOFError, OSError):
                conn.send(("exit",))
            conn.close()

        if process is not None:
            process.join(STOP_TIMEOUT_SECONDS)
            if process.is_alive():
                LOG.warning("The worker did not stop in %s s, killing it", STOP_TIMEOUT_SECONDS)
                process.kill()
                process.join()

    async def _unload_when_idle(self) -> None:
        timeout = self.settings.unload_after_seconds
        while True:
            await asyncio.sleep(IDLE_CHECK_SECONDS)
            if self._lock.locked() or not self._is_idle(timeout):
                continue

            async with self._lock:
                # a request may have run while this check waited for the lock
                if not self._is_idle(timeout):
                    continue

                LOG.info("Unloading the models after %s s without requests to free their memory", timeout)
                # set before the exchange, so a worker that answers with an error does not get a request at each check
                self._unloaded = True
                try:
                    status = await self._send("unload")
                except SynthesisError:
                    LOG.exception("Failed to unload the models")
                else:
                    LOG.info("The models are unloaded: %s", status)

    def _is_idle(self, timeout: float) -> bool:
        """Tell if a running worker that still has its models got no request for ``timeout`` seconds."""
        return self.worker_running and not self._unloaded and time.monotonic() - self._last_used >= timeout


def _retrieve_exception(future: asyncio.Future[Any]) -> None:
    # nobody awaits an abandoned exchange, and its error is expected after the worker stops
    if not future.cancelled():
        future.exception()
