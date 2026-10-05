import contextlib
import dataclasses

import pytest

from holos_tts.config import Settings
from holos_tts.synthesizer import Synthesizer
from tests import fake_worker


def make_settings(**overrides):
    return dataclasses.replace(Settings(), **overrides)


@pytest.fixture
async def synthesizer():
    synth = Synthesizer(make_settings(preload=False), chunk_chars=20, worker_target=fake_worker.run)
    yield synth
    with contextlib.suppress(Exception):
        await synth.stop()
