import contextlib
import http.server
import socket
import threading

import pytest

from holos_tts import healthcheck


class HealthHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200 if self.path == "/health" else 404)
        self.end_headers()

    def log_message(self, *args):
        pass


@pytest.fixture
def http_port():
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), HealthHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield server.server_address[1]
    server.shutdown()
    server.server_close()
    thread.join()


@pytest.fixture
def listening_port():
    with socket.create_server(("127.0.0.1", 0)) as server:
        yield server.getsockname()[1]


@pytest.fixture
def closed_port():
    with contextlib.closing(socket.socket()) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def test_http_api_that_answers_is_healthy(monkeypatch, http_port):
    monkeypatch.setenv("HTTP_PORT", str(http_port))
    assert healthcheck.main() == 0


def test_http_api_that_does_not_answer_is_unhealthy(monkeypatch, closed_port, capsys):
    monkeypatch.setenv("HTTP_PORT", str(closed_port))
    assert healthcheck.main() == 1
    assert "unhealthy" in capsys.readouterr().err


def test_wyoming_port_is_checked_when_http_is_off(monkeypatch, listening_port):
    monkeypatch.setenv("HTTP_PORT", "0")
    monkeypatch.setenv("WYOMING_PORT", str(listening_port))
    assert healthcheck.main() == 0


def test_closed_wyoming_port_is_unhealthy_when_http_is_off(monkeypatch, closed_port, capsys):
    monkeypatch.setenv("HTTP_PORT", "0")
    monkeypatch.setenv("WYOMING_PORT", str(closed_port))
    assert healthcheck.main() == 1
    assert "unhealthy" in capsys.readouterr().err
