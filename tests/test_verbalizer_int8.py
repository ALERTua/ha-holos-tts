import logging
import types
from typing import NoReturn

import pytest

from holos_tts import verbalizer


class FakeTranslator:
    loads: list[tuple[str, dict]] = []
    broken: set[str] = set()

    def __init__(self, model_dir, **options):
        FakeTranslator.loads.append((model_dir, options))
        if model_dir in FakeTranslator.broken:
            msg = f"cannot read {model_dir}"
            raise RuntimeError(msg)


@pytest.fixture
def env(tmp_path, monkeypatch):
    FakeTranslator.loads = []
    FakeTranslator.broken = set()
    model_dir = tmp_path / "hub" / "model"
    model_dir.mkdir(parents=True)
    conversions = []

    def fake_hf_hub_download(repo, filename, revision) -> str:
        path = tmp_path / "hub" / filename
        path.write_text('{"▁so": 0}', encoding="utf-8")
        return str(path)

    def fake_convert(source_dir, target_dir) -> None:
        conversions.append((source_dir, target_dir))
        target_dir.mkdir(parents=True)

    monkeypatch.setattr(verbalizer, "snapshot_download", lambda *_, **__: str(model_dir))
    monkeypatch.setattr(verbalizer, "hf_hub_download", fake_hf_hub_download)
    monkeypatch.setattr(verbalizer, "sentencepiece", types.SimpleNamespace(SentencePieceProcessor=lambda **_: None))
    monkeypatch.setattr(verbalizer.ctranslate2, "Translator", FakeTranslator)
    monkeypatch.setattr(verbalizer, "convert_to_int8", fake_convert)
    # the test machine has no GPU; this one supports float16
    monkeypatch.setattr(
        verbalizer.ctranslate2,
        "get_supported_compute_types",
        lambda _device: {"float32", "int8", "int8_float32", "int8_float16", "float16"},
    )
    cache_dir = tmp_path / "cache"
    int8_dir = cache_dir / "verbalizer-int8" / verbalizer.MODEL_REVISION[:12]
    return types.SimpleNamespace(model_dir=model_dir, cache_dir=cache_dir, int8_dir=int8_dir, conversions=conversions)


def test_converts_once_and_loads_the_int8_copy(env):
    verbalizer.Verbalizer("cuda", 3, env.cache_dir)
    verbalizer.Verbalizer("cuda", 3, env.cache_dir)

    assert env.conversions == [(env.model_dir, env.int8_dir)]
    options = {"device": "cuda", "compute_type": "int8_float16", "intra_threads": 3}
    assert FakeTranslator.loads == [(str(env.int8_dir), options), (str(env.int8_dir), options)]


@pytest.mark.parametrize(
    ("supported", "expected"),
    [
        ({"float32", "int8", "int8_float32"}, "int8_float32"),
        ({"float32"}, "float32"),
        (set(), "float32"),
    ],
)
def test_a_gpu_without_float16_gets_the_next_compute_type(env, monkeypatch, supported, expected):
    monkeypatch.setattr(verbalizer.ctranslate2, "get_supported_compute_types", lambda _device: supported)

    verbalizer.Verbalizer("cuda", 2, None)

    assert FakeTranslator.loads == [
        (str(env.model_dir), {"device": "cuda", "compute_type": expected, "intra_threads": 2})
    ]


def test_the_cpu_keeps_int8_without_asking_the_gpu(env, monkeypatch):
    def no_gpu_query(_device: str) -> NoReturn:
        msg = "the CPU path must not ask for the GPU types"
        raise AssertionError(msg)

    monkeypatch.setattr(verbalizer.ctranslate2, "get_supported_compute_types", no_gpu_query)

    verbalizer.Verbalizer("cpu", 2, None)

    assert FakeTranslator.loads == [(str(env.model_dir), {"device": "cpu", "compute_type": "int8", "intra_threads": 2})]


def test_conversion_removes_the_copies_of_other_revisions_but_not_a_running_conversion(env):
    root = env.int8_dir.parent
    running = [root / f"{env.int8_dir.name}.{pid}.partial" for pid in (4242, 4343)]
    other_revisions = [root / "0123456789ab", root / "0123456789ab.4242.partial", root / "0123456789ab.partial"]
    # the fixed name of the previous version is not a running conversion any more
    old_name = root / f"{env.int8_dir.name}.partial"
    for folder in (*other_revisions, *running, old_name):
        folder.mkdir(parents=True)

    verbalizer.Verbalizer("cpu", 2, env.cache_dir)

    assert sorted(root.iterdir()) == sorted([env.int8_dir, *running])


def test_failed_conversion_keeps_the_copies_of_other_revisions(env, monkeypatch):
    old = env.int8_dir.parent / "0123456789ab"
    old.mkdir(parents=True)

    def failing_convert(source_dir, target_dir) -> None:
        msg = "unknown binary version"
        raise ValueError(msg)

    monkeypatch.setattr(verbalizer, "convert_to_int8", failing_convert)

    verbalizer.Verbalizer("cpu", 2, env.cache_dir)

    assert old.is_dir()


def test_without_a_cache_loads_the_original_model(env):
    verbalizer.Verbalizer("cpu", 2)

    assert env.conversions == []
    assert FakeTranslator.loads == [(str(env.model_dir), {"device": "cpu", "compute_type": "int8", "intra_threads": 2})]


def test_falls_back_when_the_conversion_fails(env, monkeypatch, caplog):
    def failing_convert(source_dir, target_dir) -> None:
        target_dir.mkdir(parents=True)
        msg = "unknown binary version"
        raise ValueError(msg)

    monkeypatch.setattr(verbalizer, "convert_to_int8", failing_convert)

    with caplog.at_level(logging.WARNING, logger=verbalizer.__name__):
        verbalizer.Verbalizer("cpu", 2, env.cache_dir)

    assert [path for path, _ in FakeTranslator.loads] == [str(env.model_dir)]
    assert not env.int8_dir.exists()
    assert "the original model loads instead" in caplog.text


def test_falls_back_and_removes_a_copy_that_does_not_load(env):
    env.int8_dir.mkdir(parents=True)
    (env.int8_dir / "model.bin").write_bytes(b"broken")
    FakeTranslator.broken = {str(env.int8_dir)}

    verbalizer.Verbalizer("cpu", 2, env.cache_dir)

    assert env.conversions == []
    assert [path for path, _ in FakeTranslator.loads] == [str(env.int8_dir), str(env.model_dir)]
    assert not env.int8_dir.exists()
