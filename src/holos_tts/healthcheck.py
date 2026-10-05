"""Container health check: the HTTP API answers, or the Wyoming port accepts a connection when HTTP is off."""

from __future__ import annotations

import socket
import sys
import urllib.request

from .config import Settings

TIMEOUT_SECONDS = 3


def main() -> int:
    """Return 0 when the server answers."""
    settings = Settings.from_env()
    try:
        if settings.http_port:
            with urllib.request.urlopen(f"http://127.0.0.1:{settings.http_port}/health", timeout=TIMEOUT_SECONDS):
                return 0

        with socket.create_connection(("127.0.0.1", settings.wyoming_port), timeout=TIMEOUT_SECONDS):
            return 0
    except OSError as e:
        print(f"unhealthy: {e}", file=sys.stderr)  # noqa: T201
        return 1


if __name__ == "__main__":
    sys.exit(main())
