"""Start the OpenAI-compatible HTTP API and the Wyoming server in one process."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import signal
from functools import partial
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING

import uvicorn
from wyoming.server import AsyncTcpServer

from .config import Settings, SettingsError
from .constants import CHUNK_CHARS, PROGRAM_NAME
from .openai_api import create_app
from .synthesizer import Synthesizer
from .wyoming_server import TtsEventHandler

if TYPE_CHECKING:
    from collections.abc import Iterator

LOG = logging.getLogger(PROGRAM_NAME)


class _HttpServer(uvicorn.Server):
    """Uvicorn server that leaves the stop signals to ``main``, which stops both servers."""

    @contextlib.contextmanager
    def capture_signals(self) -> Iterator[None]:
        yield


def _version() -> str:
    try:
        return version(PROGRAM_NAME)
    except PackageNotFoundError:
        return "0"


async def _warm_up(synthesizer: Synthesizer) -> None:
    try:
        await synthesizer.start()
    except Exception:
        # the servers keep running: the next request starts a new worker and shows the error to the client
        LOG.exception("Failed to read the voice list at start")


async def main(settings: Settings) -> None:
    """Run both servers until a stop signal."""
    if not settings.http_port and not settings.wyoming_port:
        msg = "HTTP_PORT and WYOMING_PORT are both 0, so the server has nothing to serve."
        raise SettingsError(msg)

    synthesizer = Synthesizer(settings, CHUNK_CHARS)
    warm_up = asyncio.create_task(_warm_up(synthesizer), name="warm up")
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError):
            loop.add_signal_handler(sig, stop.set)

    wyoming = None
    if settings.wyoming_port:
        wyoming = AsyncTcpServer(settings.wyoming_host, settings.wyoming_port)
        await wyoming.start(partial(TtsEventHandler, synthesizer=synthesizer, version=_version()))
        LOG.info(
            "Wyoming server listens on %s:%s",
            settings.wyoming_host,
            settings.wyoming_port,
        )

    http = None
    if settings.http_port:
        config = uvicorn.Config(
            create_app(synthesizer),
            host=settings.http_host,
            port=settings.http_port,
            log_level=settings.log_level.lower(),
            access_log=False,
        )
        http = _HttpServer(config)
        http_task = asyncio.create_task(http.serve(), name="http")
        http_task.add_done_callback(lambda _: stop.set())
        LOG.info(
            "OpenAI-compatible API listens on http://%s:%s",
            settings.http_host,
            settings.http_port,
        )

    try:
        await stop.wait()
    finally:
        LOG.info("Stopping")
        if http is not None:
            http.should_exit = True
            await http_task

        if wyoming is not None:
            await wyoming.stop()

        warm_up.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await warm_up

        await synthesizer.stop()


def run() -> None:
    """Console entry point."""
    try:
        settings = Settings.from_env()
    except SettingsError as e:
        raise SystemExit(str(e)) from e

    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # one line for each HTTP request of the model downloads hides the useful lines
    logging.getLogger("httpx").setLevel(max(logging.WARNING, logging.getLogger().level))
    LOG.info("Settings: %s", settings)
    asyncio.run(main(settings))


if __name__ == "__main__":
    run()
