import logging
import os
import sys
from pathlib import Path
from typing import Never

import pytest
import uvicorn.config

from holos_tts import config
from holos_tts.config import Settings, SettingsError

HOST_CPUS = 4
# read before the autouse fixture replaces it
REAL_SCHED_GETAFFINITY = config.SCHED_GETAFFINITY


@pytest.fixture(autouse=True)
def cpu_max(tmp_path, monkeypatch):
    """The file with the CPU limit. It does not exist until a test writes it."""
    path = tmp_path / "cpu.max"
    monkeypatch.setattr(config, "CGROUP_CPU_MAX", path)
    return path


@pytest.fixture(autouse=True)
def affinity(monkeypatch):
    """The CPUs that the process may run on. The set holds all host CPUs until a test changes it."""
    cpus = set(range(HOST_CPUS))

    def own_cpus(pid) -> set[int]:
        assert pid == 0, "pid 0 is the server process, another pid has its own set"
        return cpus

    monkeypatch.setattr(os, "cpu_count", lambda: HOST_CPUS)
    monkeypatch.setattr(config, "SCHED_GETAFFINITY", own_cpus)
    return cpus


def _pin(affinity, cpus) -> None:
    affinity.clear()
    affinity.update(cpus)


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


@pytest.mark.parametrize("empty", ["", "  "])
def test_empty_device_values_give_the_defaults(empty):
    settings = Settings.from_env({"DEVICE": empty, "VERBALIZER_DEVICE": empty})
    assert (settings.device, settings.verbalizer_device) == ("cpu", "cpu")


def test_empty_verbalizer_device_follows_the_device():
    assert Settings.from_env({"DEVICE": "cuda", "VERBALIZER_DEVICE": ""}).verbalizer_device == "cuda"


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


@pytest.mark.parametrize("env", [{}, {"THREADS": ""}, {"THREADS": "0"}])
def test_threads_from_cpu_pinning(affinity, env):
    _pin(affinity, {1, 2, 3})
    assert Settings.from_env(env).threads == 3


@pytest.mark.parametrize(
    ("content", "cpus", "threads"),
    [
        ("400000 100000\n", {1, 2, 3}, 3),
        ("150000 100000\n", {1, 2, 3}, 2),
        ("100000 100000\n", {2, 3}, 1),
        ("max 100000\n", {2, 3}, 2),
        ("garbage", {0}, 1),
        ("200000 100000\n", set(range(HOST_CPUS)), 2),
    ],
)
def test_threads_take_the_smaller_of_cpu_limit_and_cpu_pinning(cpu_max, affinity, content, cpus, threads):
    cpu_max.write_text(content, encoding="ascii")
    _pin(affinity, cpus)
    assert Settings.from_env({}).threads == threads


def test_explicit_threads_win_over_cpu_pinning(affinity):
    _pin(affinity, {1})
    assert Settings.from_env({"THREADS": "2"}).threads == 2


def test_threads_from_the_affinity_set_when_host_cpu_count_is_unknown(monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: None)
    assert Settings.from_env({}).threads == HOST_CPUS


def test_affinity_source_is_the_function_of_the_os():
    assert REAL_SCHED_GETAFFINITY is getattr(os, "sched_getaffinity", None)
    assert (REAL_SCHED_GETAFFINITY is None) == (sys.platform != "linux")


def test_threads_without_affinity_support_use_only_cpu_limit(cpu_max, monkeypatch):
    monkeypatch.setattr(config, "SCHED_GETAFFINITY", None)
    assert Settings.from_env({}).threads == 0
    cpu_max.write_text("300000 100000\n", encoding="ascii")
    assert Settings.from_env({}).threads == 3


def test_threads_ignore_an_affinity_error(cpu_max, monkeypatch):
    def refuse(_pid) -> Never:
        raise OSError

    monkeypatch.setattr(config, "SCHED_GETAFFINITY", refuse)
    assert Settings.from_env({}).threads == 0
    cpu_max.write_text("300000 100000\n", encoding="ascii")
    assert Settings.from_env({}).threads == 3


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


@pytest.mark.parametrize(
    ("raw", "level"),
    [
        (None, "INFO"),
        ("", "INFO"),
        ("  ", "INFO"),
        ("debug", "DEBUG"),
        (" Warning ", "WARNING"),
        ("ERROR", "ERROR"),
        ("critical", "CRITICAL"),
        ("WARN", "WARNING"),
        ("fatal", "CRITICAL"),
    ],
)
def test_log_level_is_normalized_to_a_name_that_logging_and_uvicorn_accept(raw, level):
    settings = Settings.from_env({} if raw is None else {"LOG_LEVEL": raw})
    assert settings.log_level == level
    assert settings.log_level in logging.getLevelNamesMapping()
    assert settings.log_level.lower() in uvicorn.config.LOG_LEVELS


@pytest.mark.parametrize("raw", ["verbose", "TRACE", "NOTSET", "10", "warning!"])
def test_unknown_log_level_is_refused_with_the_variable_name_and_the_allowed_values(raw):
    with pytest.raises(SettingsError) as error:
        Settings.from_env({"LOG_LEVEL": raw})

    assert f"LOG_LEVEL={raw!r}" in str(error.value)
    assert "DEBUG, INFO, WARNING, ERROR, CRITICAL" in str(error.value)
