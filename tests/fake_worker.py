"""A worker process without models, for the tests of the synthesizer and the servers."""

import os
import random
import time

import numpy as np

from holos_tts.constants import MODEL_PARTS

VOICES = ["Гаська Шиян", "Speaker_0"]
# a "SLOW" request must outlast two pauses of a test, so a test can cancel it twice while the worker still runs it
SLOW_SECONDS = 0.25


def run(conn, settings):
    loaded = False
    unloads = 0
    while True:
        try:
            message = conn.recv()
        except EOFError:
            return

        command, *args = message
        if command == "exit":
            return

        if command == "voices":
            result = ("ok", VOICES)
        elif command == "load":
            loaded = True
            result = ("ok", {"engine": loaded})
        elif command == "load_part":
            # the real worker refuses a model that it does not know
            if args[0] in MODEL_PARTS:
                loaded = True
                result = ("ok", {"engine": loaded})
            else:
                result = ("error", f"ValueError: Unknown model {args[0]!r}")
        elif command == "unload":
            loaded = False
            unloads += 1
            result = ("ok", {"engine": loaded, "pid": os.getpid()})
        elif command == "status":
            result = ("ok", {"engine": loaded, "pid": os.getpid(), "unloads": unloads})
        elif command == "synth":
            sentences, _voice, speed = args
            result = synthesize(" ".join(sentences), speed)
        elif command == "verbalize":
            result = verbalize(args[0])
        else:
            result = ("error", f"unknown command {command}")

        conn.send(result)


def synthesize(text, speed):
    """Act on the markers in ``text``, else return one sample of ``speed / 10`` per character."""
    if "CRASH" in text:
        os._exit(1)

    if "SLOW" in text:
        time.sleep(SLOW_SECONDS)

    if "JITTER" in text:
        time.sleep(random.uniform(0, 0.03))  # noqa: S311

    if "HANG" in text:
        time.sleep(30)

    if "FAIL" in text:
        return ("error", "ValueError: bad text")

    if "LOUD" in text:
        return ("ok", np.tile(np.array([2.0, 1.0], dtype=np.float32), 4))

    # one sample per character, so a test can see which chunk came back
    return ("ok", np.full(len(text), speed / 10, dtype=np.float32))


def verbalize(sentences):
    """Return the sentences joined by spaces, with each digit 7 written as a word. A "FAIL" sentence is an error."""
    text = " ".join(sentences)
    if "FAIL" in text:
        return ("error", "ValueError: bad text")

    return ("ok", text.replace("7", "сім"))


def refusing_unload_worker(conn, settings):
    """A fake worker that answers each ``unload`` with an error and counts them."""
    unloads = 0
    while True:
        try:
            command, *_args = conn.recv()
        except EOFError:
            return

        if command == "exit":
            return

        if command == "unload":
            unloads += 1
            result = ("error", "RuntimeError: cannot unload")
        elif command == "status":
            result = ("ok", {"unloads": unloads})
        elif command == "voices":
            result = ("ok", VOICES)
        else:
            result = ("error", f"unknown command {command}")

        conn.send(result)


LOAD_SECONDS = 0.1


def warm_up_worker(conn, settings):
    """A fake worker that knows ``load_part`` and tells the order of the commands that it got."""
    log = []
    while True:
        try:
            command, *args = conn.recv()
        except EOFError:
            return

        if command == "exit":
            return

        if command == "status":
            conn.send(("ok", {"log": log}))
            continue

        log.append(f"{command} {args[0]}" if command == "load_part" else command)
        if command == "voices":
            result = ("ok", VOICES)
        elif command == "load_part":
            time.sleep(LOAD_SECONDS)
            result = ("ok", {args[0]: True})
        elif command == "synth":
            _sentences, _voice, speed = args
            result = ("ok", np.full(3, speed / 10, dtype=np.float32))
        else:
            result = ("error", f"unknown command {command}")

        conn.send(result)
