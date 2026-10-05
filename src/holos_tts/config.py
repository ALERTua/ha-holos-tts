"""Settings that the server reads from environment variables."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from .constants import MAX_SPEED, MIN_SPEED

if TYPE_CHECKING:
    from collections.abc import Mapping

TRUE_VALUES = ("1", "true", "yes", "on", "y", "t")
FALSE_VALUES = ("0", "false", "no", "off", "n", "f", "")
DEVICES = ("cpu", "cuda")
CGROUP_CPU_MAX = Path("/sys/fs/cgroup/cpu.max")


class SettingsError(ValueError):
    """An environment variable holds a value that the server cannot use."""


def _bool(env: Mapping[str, str], name: str, *, default: bool) -> bool:
    raw = env.get(name)
    if raw is None:
        return default

    value = raw.strip().lower()
    if value in TRUE_VALUES:
        return True

    if value in FALSE_VALUES:
        return False

    msg = f"{name}={raw!r} is not a boolean. Use 1 or 0."
    raise SettingsError(msg)


def _int(env: Mapping[str, str], name: str, *, default: int, minimum: int = 0) -> int:
    raw = env.get(name)
    if raw is None or not raw.strip():
        return default

    value = raw.strip()
    if not (value.isascii() and value.lstrip("-").isdigit()):
        msg = f"{name}={raw!r} is not a whole number."
        raise SettingsError(msg)

    number = int(value)
    if number < minimum:
        msg = f"{name}={raw!r} is less than {minimum}."
        raise SettingsError(msg)

    return number


def _float(env: Mapping[str, str], name: str, *, default: float, low: float, high: float) -> float:
    raw = env.get(name)
    if raw is None or not raw.strip():
        return default

    try:
        number = float(raw.strip())
    except ValueError as e:
        msg = f"{name}={raw!r} is not a number."
        raise SettingsError(msg) from e

    if not low <= number <= high:
        msg = f"{name}={raw!r} is outside the range {low}..{high}."
        raise SettingsError(msg)

    return number


def _cpu_limit_threads() -> int:
    """Whole CPUs of the container limit (cgroup v2), or 0 when there is no limit or the file is unusable."""
    try:
        quota, period = CGROUP_CPU_MAX.read_text(encoding="ascii").split()
    except (OSError, ValueError):
        return 0

    if not (quota.isdigit() and period.isdigit() and int(period) > 0):
        return 0

    return math.ceil(int(quota) / int(period))


def _choice(env: Mapping[str, str], name: str, *, default: str, choices: tuple[str, ...]) -> str:
    value = env.get(name, default).strip().lower() or default
    if value not in choices:
        msg = f"{name}={value!r} is not one of {', '.join(choices)}."
        raise SettingsError(msg)

    return value


@dataclass(frozen=True)
class Settings:
    """Server settings. Each field has the environment variable of the same name in upper case."""

    http_host: str = "0.0.0.0"  # noqa: S104 - the server runs in a container and must be reachable
    http_port: int = 8000
    wyoming_host: str = "0.0.0.0"  # noqa: S104
    wyoming_port: int = 10200
    data_dir: Path = field(default_factory=lambda: Path("/data"))
    device: str = "cpu"
    verbalizer_device: str = "cpu"
    default_voice: str = "Speaker_67"
    default_speed: float = 1.0
    verbalize: bool = True
    preload: bool = True
    web_ui: bool = False
    unload_after_seconds: int = 0
    verbalizer_unload_after_seconds: int = 0
    threads: int = 0
    log_level: str = "INFO"

    @property
    def voices_dir(self) -> Path:
        """Folder with extra voice files that the user adds."""
        return self.data_dir / "voices"

    @property
    def cache_dir(self) -> Path:
        """Folder for files that the server derives from the downloaded models and can make again."""
        return self.data_dir / "cache"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        """Read the settings from the environment, with the defaults of this class."""
        env = os.environ if env is None else env
        defaults = cls()
        device = _choice(env, "DEVICE", default=defaults.device, choices=DEVICES)
        threads = _int(env, "THREADS", default=defaults.threads)
        return cls(
            http_host=env.get("HTTP_HOST", defaults.http_host),
            http_port=_int(env, "HTTP_PORT", default=defaults.http_port),
            wyoming_host=env.get("WYOMING_HOST", defaults.wyoming_host),
            wyoming_port=_int(env, "WYOMING_PORT", default=defaults.wyoming_port),
            data_dir=Path(env.get("DATA_DIR", str(defaults.data_dir))),
            device=device,
            verbalizer_device=_choice(env, "VERBALIZER_DEVICE", default=device, choices=DEVICES),
            default_voice=env.get("DEFAULT_VOICE", defaults.default_voice).strip() or defaults.default_voice,
            default_speed=_float(env, "DEFAULT_SPEED", default=defaults.default_speed, low=MIN_SPEED, high=MAX_SPEED),
            # AUTO_USE_VERBALIZER is the name that the older styletts2 API used for the same switch
            verbalize=_bool(
                env,
                "VERBALIZE",
                default=_bool(env, "AUTO_USE_VERBALIZER", default=defaults.verbalize),
            ),
            preload=_bool(env, "PRELOAD", default=defaults.preload),
            web_ui=_bool(env, "WEB_UI", default=defaults.web_ui),
            unload_after_seconds=_int(env, "UNLOAD_AFTER_SECONDS", default=defaults.unload_after_seconds),
            verbalizer_unload_after_seconds=_int(
                env,
                "VERBALIZER_UNLOAD_AFTER_SECONDS",
                default=defaults.verbalizer_unload_after_seconds,
            ),
            # the libraries start a thread per host core, so a container limit must set the number
            threads=threads or _cpu_limit_threads(),
            log_level=env.get("LOG_LEVEL", defaults.log_level).strip().upper() or defaults.log_level,
        )
