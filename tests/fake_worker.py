"""A worker process without models, for the tests of the synthesizer and the servers."""

import os
import random
import time

import numpy as np

from holos_tts.constants import MODEL_PARTS

VOICES = ["Гаська Шиян", "Speaker_0"]


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
        else:
            result = ("error", f"unknown command {command}")

        conn.send(result)


def synthesize(text, speed):
    """Act on the markers in ``text``, else return one sample of ``speed / 10`` per character."""
    if "CRASH" in text:
        os._exit(1)

    if "SLOW" in text:
        time.sleep(0.5)

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
