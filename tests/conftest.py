import contextlib
import dataclasses

import pytest

from holos_tts import synthesizer as synthesizer_module
from holos_tts.config import Settings
from holos_tts.synthesizer import Synthesizer
from tests import fake_worker


def make_settings(**overrides):
    return dataclasses.replace(Settings(), **overrides)


@pytest.fixture(autouse=True)
def short_stop_timeout(monkeypatch):
    """A hung fake worker gets killed after 0.5 s, not after the 10 s of production."""
    monkeypatch.setattr(synthesizer_module, "STOP_TIMEOUT_SECONDS", 0.5)


@pytest.fixture
async def synthesizer():
    synth = Synthesizer(make_settings(preload=False), chunk_chars=20, worker_target=fake_worker.run)
    yield synth
    with contextlib.suppress(Exception):
        await synth.stop()
