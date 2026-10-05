"""
Entry point of the worker process.

The main process imports this module to name the target of the process. This module imports the heavy worker code
only inside the new process, so the main process stays small.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from multiprocessing.connection import Connection

    from .config import Settings

LOG = logging.getLogger(__name__)
POLL_SECONDS = 5.0


def run(conn: Connection, settings: Settings) -> None:
    """Answer the requests of the main process until it sends "exit" or closes the pipe."""
    logging.basicConfig(
        level=settings.log_level,
        format="%(asctime)s %(levelname)s worker %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(max(logging.WARNING, logging.getLogger().level))
    from .memory import LOAD_MMAP_THRESHOLD, set_mmap_threshold  # noqa: PLC0415

    # the imports and the model loads below must not leave freed buffers in the heap
    set_mmap_threshold(LOAD_MMAP_THRESHOLD)
    from .worker import Worker  # noqa: PLC0415

    worker = Worker(settings)
    while True:
        try:
            if not conn.poll(POLL_SECONDS):
                worker.unload_idle_verbalizer()
                continue

            message = conn.recv()
        except (EOFError, OSError):
            LOG.info("The main process closed the pipe, the worker stops")
            return

        if message[0] == "exit":
            return

        try:
            result = ("ok", worker.handle(message))
        except Exception as e:
            LOG.exception("The worker failed to run %r", message[0])
            result = ("error", f"{type(e).__name__}: {e}")

        try:
            conn.send(result)
        except (EOFError, OSError):
            return
