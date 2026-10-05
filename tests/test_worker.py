from pathlib import Path

import numpy as np
import pytest

from holos_tts import memory, worker
from tests.conftest import make_settings


class FakeVerbalizer:
    calls: list[str] = []

    def __init__(self, *_):
        pass

    def __call__(self, text):
        FakeVerbalizer.calls.append(text)
        return text.replace("7", "сьомій")


class FakeEngine:
    def synthesize(self, phonemes, voice, speed):
        return np.full(len(phonemes), speed, dtype=np.float32)


@pytest.fixture
def fake_worker(monkeypatch):
    FakeVerbalizer.calls = []
    monkeypatch.setattr(worker, "Verbalizer", FakeVerbalizer)
    w = worker.Worker(make_settings())
    w._voices = {"Гаська Шиян": np.zeros(256, dtype=np.float32)}
    w._phonemizer = lambda text: text
    w._engine = FakeEngine()
    return w


def test_verbalizer_loads_only_for_text_with_numbers(fake_worker):
    fake_worker.synthesize(["Двері відчинені."], "Гаська Шиян", 1.0)
    assert fake_worker.status()["verbalizer"] is False
    fake_worker.synthesize(["Зустріч о 7.", "Без цифр."], "Гаська Шиян", 1.0)
    assert FakeVerbalizer.calls == ["Зустріч о 7."]


def test_unknown_voice_is_an_error(fake_worker):
    with pytest.raises(KeyError, match="alloy"):
        fake_worker.synthesize(["Так."], "alloy", 1.0)


def test_idle_verbalizer_is_unloaded(fake_worker):
    fake_worker.settings = make_settings(verbalizer_unload_after_seconds=1)
    fake_worker.synthesize(["О 7."], "Гаська Шиян", 1.0)
    fake_worker._verbalizer_used -= 5
    fake_worker.unload_idle_verbalizer()
    assert fake_worker.status()["verbalizer"] is False


def test_model_loading_restores_the_run_threshold_after_an_error(monkeypatch):
    seen = []
    monkeypatch.setattr(memory, "set_mmap_threshold", seen.append)
    with pytest.raises(RuntimeError), memory.model_loading():
        raise RuntimeError

    assert seen == [memory.LOAD_MMAP_THRESHOLD, memory.RUN_MMAP_THRESHOLD]


class Recorder:
    """Stand-in for a model class that records which models the worker built."""

    built: list[str] = []
    cache_dirs: dict[str, Path | None] = {}

    def __init__(self, *args, **kwargs):
        Recorder.built.append(type(self).__name__)


class FakePhonemizer(Recorder):
    pass


class FakeHolosEngine(Recorder):
    def __init__(self, voices, device="cpu", *, threads=0, cache_dir=None):
        super().__init__()
        Recorder.cache_dirs["engine"] = cache_dir


class FakeRecordedVerbalizer(Recorder):
    def __init__(self, device="cpu", threads=0, cache_dir=None):
        super().__init__()
        Recorder.cache_dirs["verbalizer"] = cache_dir


@pytest.fixture
def empty_worker(monkeypatch):
    Recorder.built = []
    Recorder.cache_dirs = {}
    monkeypatch.setattr(worker, "Phonemizer", FakePhonemizer)
    monkeypatch.setattr(worker, "HolosEngine", FakeHolosEngine)
    monkeypatch.setattr(worker, "Verbalizer", FakeRecordedVerbalizer)

    def make(**overrides) -> worker.Worker:
        w = worker.Worker(make_settings(**overrides))
        w._voices = {"Гаська Шиян": np.zeros(256, dtype=np.float32)}
        return w

    return make


@pytest.mark.parametrize(
    ("name", "built"),
    [("stress", "FakePhonemizer"), ("engine", "FakeHolosEngine"), ("verbalizer", "FakeRecordedVerbalizer")],
)
def test_load_part_loads_only_the_named_model(empty_worker, name, built):
    status = empty_worker().handle(("load_part", name))
    assert Recorder.built == [built]
    assert {part: status[part] for part in ("stress", "engine", "verbalizer")} == {
        part: part == name for part in ("stress", "engine", "verbalizer")
    }


def test_engine_and_verbalizer_get_the_cache_dir_of_the_settings(empty_worker, tmp_path):
    w = empty_worker(data_dir=tmp_path / "test-data")

    w.handle(("load",))

    expected = tmp_path / "test-data" / "cache"
    assert w.settings.cache_dir == expected
    assert Recorder.cache_dirs == {"engine": expected, "verbalizer": expected}


def test_unload_drops_the_three_models_and_the_next_load_builds_them_again(empty_worker, monkeypatch):
    freed = []
    monkeypatch.setattr(worker, "release_free_memory", lambda: freed.append("freed"))
    w = empty_worker()
    voices = w._voices
    models = ("stress", "engine", "verbalizer")
    assert all(w.handle(("load",))[part] for part in models)
    assert freed == []

    status = w.handle(("unload",))
    assert [status[part] for part in models] == [False, False, False]
    assert freed == ["freed"]
    # the voice vectors are small, so they stay
    assert w._voices is voices

    assert all(w.handle(("load",))[part] for part in models)
    assert Recorder.built == ["FakePhonemizer", "FakeHolosEngine", "FakeRecordedVerbalizer"] * 2


def test_load_part_skips_a_verbalizer_that_the_settings_turn_off(empty_worker):
    status = empty_worker(verbalize=False).handle(("load_part", "verbalizer"))
    assert Recorder.built == []
    assert status["verbalizer"] is False


def test_load_part_refuses_an_unknown_model(empty_worker):
    with pytest.raises(ValueError, match="voices"):
        empty_worker().handle(("load_part", "voices"))


def test_verbalize_sends_only_a_text_with_numbers_to_the_verbalizer(fake_worker):
    assert fake_worker.verbalize(["Зустріч о 7.", "Без цифр."]) == "Зустріч о сьомій. Без цифр."
    assert FakeVerbalizer.calls == ["Зустріч о 7."]


def test_verbalize_command_does_not_load_the_verbalizer_for_a_text_without_numbers(fake_worker):
    assert fake_worker.handle(("verbalize", ["Двері відчинені.", "Так."])) == "Двері відчинені. Так."
    assert fake_worker.status()["verbalizer"] is False
    assert FakeVerbalizer.calls == []


def test_verbalize_command_returns_the_text_as_it_is_when_the_settings_turn_the_verbalizer_off(empty_worker):
    w = empty_worker(verbalize=False)
    assert w.handle(("verbalize", ["Зустріч о 7."])) == "Зустріч о 7."
    assert Recorder.built == []


def test_verbalize_command_releases_the_free_memory(fake_worker, monkeypatch):
    calls = []
    monkeypatch.setattr(worker, "release_free_memory", lambda: calls.append(1))
    fake_worker.handle(("verbalize", ["О 7."]))
    assert calls == [1]
