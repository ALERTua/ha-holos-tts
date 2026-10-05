from pathlib import Path

import pytest

from holos_tts import config
from holos_tts.config import Settings, SettingsError


@pytest.fixture(autouse=True)
def cpu_max(tmp_path, monkeypatch):
    """The file with the CPU limit. It does not exist until a test writes it."""
    path = tmp_path / "cpu.max"
    monkeypatch.setattr(config, "CGROUP_CPU_MAX", path)
    return path


def test_defaults():
    settings = Settings.from_env({})
    assert settings.default_voice == "Speaker_67"
    assert settings.threads == 0
    assert settings.http_port == 8000
    assert settings.wyoming_port == 10200
    assert settings.device == "cpu"
    assert settings.verbalizer_device == "cpu"
    assert settings.verbalize is True
    assert settings.web_ui is False
    assert settings.voices_dir == Path("/data/voices")


def test_values_from_env():
    settings = Settings.from_env(
        {
            "HTTP_PORT": "0",
            "WYOMING_PORT": "10300",
            "DEVICE": "CUDA",
            "DEFAULT_VOICE": "Speaker_0",
            "DEFAULT_SPEED": "1.15",
            "UNLOAD_AFTER_SECONDS": "60",
            "WEB_UI": "1",
            "DATA_DIR": "/srv/tts",
        }
    )
    assert settings.http_port == 0
    assert settings.wyoming_port == 10300
    assert settings.device == "cuda"
    assert settings.verbalizer_device == "cuda"
    assert settings.default_voice == "Speaker_0"
    assert settings.default_speed == pytest.approx(1.15)
    assert settings.unload_after_seconds == 60
    assert settings.web_ui is True
    assert settings.voices_dir == Path("/srv/tts/voices")


def test_verbalizer_device_can_differ_from_device():
    settings = Settings.from_env({"DEVICE": "cuda", "VERBALIZER_DEVICE": "cpu"})
    assert settings.verbalizer_device == "cpu"


@pytest.mark.parametrize(
    ("env", "verbalize"),
    [({"AUTO_USE_VERBALIZER": "0"}, False), ({"VERBALIZE": "off"}, False)],
)
def test_verbalizer_switch(env, verbalize):
    assert Settings.from_env(env).verbalize is verbalize


@pytest.mark.parametrize(
    ("content", "threads"),
    [
        ("400000 100000\n", 4),
        ("150000 100000\n", 2),
        ("1000 100000\n", 1),
        ("100000 100000\n", 1),
        ("0 100000\n", 0),
        ("max 100000\n", 0),
        ("", 0),
        ("garbage", 0),
        ("400000\n", 0),
        ("400000 0\n", 0),
        ("-1 100000\n", 0),
        ("4e5 100000 extra\n", 0),
    ],
)
@pytest.mark.parametrize("env", [{}, {"THREADS": ""}, {"THREADS": "0"}])
def test_threads_from_cpu_limit(cpu_max, env, content, threads):
    cpu_max.write_text(content, encoding="ascii")
    assert Settings.from_env(env).threads == threads


@pytest.mark.parametrize("env", [{}, {"THREADS": "0"}])
def test_threads_without_cpu_limit_file(env):
    assert Settings.from_env(env).threads == 0


def test_explicit_threads_win_over_cpu_limit(cpu_max):
    cpu_max.write_text("400000 100000\n", encoding="ascii")
    assert Settings.from_env({"THREADS": "2"}).threads == 2


@pytest.mark.parametrize(
    "env",
    [
        {"HTTP_PORT": "eighty"},
        {"HTTP_PORT": "-1"},
        {"HTTP_PORT": "８０"},
        {"DEVICE": "tpu"},
        {"DEFAULT_SPEED": "3"},
        {"DEFAULT_SPEED": "fast"},
        {"PRELOAD": "maybe"},
        {"WEB_UI": "maybe"},
    ],
)
def test_bad_values_are_refused(env):
    with pytest.raises(SettingsError):
        Settings.from_env(env)
