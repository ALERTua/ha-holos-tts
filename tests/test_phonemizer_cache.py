import numpy as np
import pytest
import ukrainian_word_stress
from stanza.models.common.pretrain import Pretrain

from holos_tts import worker
from holos_tts.frontend import Phonemizer
from tests.conftest import make_settings


class FakeStressifier:
    """Stands for the stress model: remembers whether ``Pretrain.load`` was wrapped when the model loaded."""

    wrapped_at_load: list[bool] = []

    def __init__(self):
        FakeStressifier.wrapped_at_load.append(hasattr(Pretrain.load, "__wrapped__"))

    def __call__(self, text):
        return text


@pytest.fixture
def fake_stress(monkeypatch):
    FakeStressifier.wrapped_at_load = []
    monkeypatch.setattr(ukrainian_word_stress, "Stressifier", FakeStressifier)
    # the test restores the loader after Phonemizer wrapped it
    monkeypatch.setattr(Pretrain, "load", Pretrain.load)


def test_cache_dir_installs_the_copy_before_the_stress_model_loads(fake_stress, tmp_path):
    Phonemizer(cache_dir=tmp_path)

    assert FakeStressifier.wrapped_at_load == [True]


def test_without_a_cache_dir_the_pretrain_loader_stays_as_it_is(fake_stress):
    Phonemizer()

    assert FakeStressifier.wrapped_at_load == [False]


class FakePhonemizer:
    calls: list[tuple] = []

    def __init__(self, *args, **kwargs):
        FakePhonemizer.calls.append((args, kwargs))


def test_worker_gives_the_cache_dir_to_the_phonemizer(monkeypatch, tmp_path):
    FakePhonemizer.calls = []
    monkeypatch.setattr(worker, "Phonemizer", FakePhonemizer)
    settings = make_settings(data_dir=tmp_path, threads=3)
    w = worker.Worker(settings)
    w._voices = {"Гаська Шиян": np.zeros(256, dtype=np.float32)}

    _ = w.phonemizer

    assert FakePhonemizer.calls == [((3,), {"cache_dir": tmp_path / "cache"})]
