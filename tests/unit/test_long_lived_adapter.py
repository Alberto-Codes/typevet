"""Unit tests: long-lived adapter behaviour on one session ([#204][i204]).

A stub llama.cpp router runs on a loopback socket (``127.0.0.1``, port chosen
by the kernel). The tests use a real socket so that the ``httpx`` connection
pool sees a real restart: the stub closes every open connection and stops
listening. No request leaves the host.

A call while the stub is down raises ``TransportError``. An error status on
``/tokenize`` raises ``BackendHttpError`` ([#298][i298]). A 200 ``/tokenize``
body that is not JSON, or that has no ``tokens`` list of integers, raises
``GenerationError`` ([#310][i310]).

Examples:
    ```bash
    uv run pytest -q tests/unit/test_long_lived_adapter.py
    ```

See Also:
    - [typevet.adapters.outbound.llama_cpp.gemma_native_vision_factory][]: Factory
    - [typevet.adapters.outbound.llama_cpp.scoring][]: Scoring adapter

[i204]: https://github.com/Alberto-Codes/typevet/issues/204
[i298]: https://github.com/Alberto-Codes/typevet/issues/298
[i310]: https://github.com/Alberto-Codes/typevet/issues/310
"""

from __future__ import annotations

import json
import socket
import threading
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from typevet.adapters.inbound.settings import load_llama_settings
from typevet.adapters.outbound.llama_cpp.gemma_native_vision_factory import (
    GemmaNativeVisionSession,
    open_gemma_native_vision_judgment,
)
from typevet.domain.errors import BackendHttpError, GenerationError, TransportError
from typevet.domain.judgment_answers import NoulAnswer
from typevet.domain.judgment_questions import Noul
from typevet.domain.media import ImageInput

pytestmark = pytest.mark.unit

_MODEL = "gemma-4-long-lived"
_GEMMA4_RENDERED = "<|turn>user\nhello<turn|>\n<|turn>model\n"
_MARKER = "<__media__>"
_PNG = ImageInput(data=b"\x89PNG\r\n\x1a\nstub", mime_type="image/png")
_QUESTIONS = {"filled": Noul(instructions="Is the image filled with one colour?")}
_CALLS = 200
_RESTART_AT = 100


@dataclass
class _StubState:
    """Counters that survive a stub restart.

    Attributes:
        connections (int): TCP connections the stub accepted.
        completions (int): ``/completion`` requests the stub answered.
        props_calls (int): ``/props`` requests the stub answered.
        tokenize_status (int): HTTP status the stub sends for ``/tokenize``.
        tokenize_raw (tuple[str, bytes] | None): Content type and raw body of
            a 200 reply that replaces the ``/tokenize`` JSON when set.
        open_sockets (list[socket.socket]): Accepted sockets, closed on stop.
    """

    connections: int = 0
    completions: int = 0
    props_calls: int = 0
    tokenize_status: int = 200
    tokenize_raw: tuple[str, bytes] | None = None
    open_sockets: list[socket.socket] = field(default_factory=list)


def _route(state: _StubState, path: str, body: dict[str, Any]) -> dict[str, Any]:
    """Return the JSON body for one stub route.

    Returns:
        Canned llama.cpp JSON for ``path``.
    """
    if path == "/apply-template":
        return {"prompt": _GEMMA4_RENDERED}
    if path == "/props":
        state.props_calls += 1
        return {"modalities": {"vision": True}, "media_marker": _MARKER}
    if path == "/tokenize":
        token_id = 100 + sum(map(ord, body["content"])) % 50
        return {"tokens": [token_id]}
    state.completions += 1
    top = [{"id": i, "logprob": -0.5 - i * 0.001} for i in range(100, 150)]
    return {
        "completion_probabilities": [{"top_logprobs": top}],
        "tokens_evaluated": 300,
    }


def _handler(state: _StubState) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        disable_nagle_algorithm = True
        wbufsize = -1

        def _reply(self, body: dict[str, Any], status: int = 200) -> None:
            self._send(json.dumps(body).encode(), "application/json", status)

        def _send(self, payload: bytes, content_type: str, status: int) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:
            self._reply(_route(state, self.path.split("?")[0], {}))

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length) or b"{}")
            if self.path == "/tokenize" and state.tokenize_raw is not None:
                self._send(state.tokenize_raw[1], state.tokenize_raw[0], 200)
                return
            if self.path == "/tokenize" and state.tokenize_status != 200:
                self._reply({"error": "tokenize failed"}, state.tokenize_status)
                return
            self._reply(_route(state, self.path, body))

        def log_message(self, format: str, *args: Any) -> None:
            del format, args

    return Handler


class _StubRouter:
    """Loopback llama.cpp stub that can stop and start on one port.

    Attributes:
        state (_StubState): Counters shared across restarts.
        port (int): Loopback port; stays the same across restarts.
    """

    def __init__(self) -> None:
        self.state = _StubState()
        self.port = 0
        self._server: ThreadingHTTPServer | None = None

    @property
    def base_url(self) -> str:
        """Return the stub base URL."""
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> None:
        """Listen on ``port`` (a free port on the first start)."""
        state = self.state

        class Server(ThreadingHTTPServer):
            daemon_threads = True

            def get_request(self) -> tuple[socket.socket, Any]:
                conn, addr = super().get_request()
                state.connections += 1
                state.open_sockets.append(conn)
                return conn, addr

        self._server = Server(("127.0.0.1", self.port), _handler(state))
        self.port = self._server.server_address[1]
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def stop(self) -> None:
        """Stop listening and close every accepted connection."""
        assert self._server is not None
        self._server.shutdown()
        self._server.server_close()
        for conn in self.state.open_sockets:
            with suppress(OSError):
                conn.shutdown(socket.SHUT_RDWR)
            conn.close()
        self.state.open_sockets.clear()
        self._server = None


@pytest.fixture
def stub() -> Iterator[_StubRouter]:
    """Start one stub router and stop it after the test."""
    router = _StubRouter()
    router.start()
    yield router
    if router._server is not None:
        router.stop()


@contextmanager
def _session(stub: _StubRouter) -> Iterator[GemmaNativeVisionSession]:
    settings = load_llama_settings(
        {
            "TYPEVET_LLAMA__BASE_URL": stub.base_url,
            "TYPEVET_LLAMA__TIMEOUT": "5",
            "TYPEVET_LLAMA__MULTIMODAL_MODEL": _MODEL,
        }
    )
    with open_gemma_native_vision_judgment(settings=settings) as session:
        yield session


def _judge_valid(session: GemmaNativeVisionSession) -> None:
    response = session.port.judge("Look.", _QUESTIONS, _MODEL, media=(_PNG,))
    answer = response.answers["filled"]
    assert isinstance(answer, NoulAnswer)
    assert 0.0 <= answer.noul <= 1.0
    assert response.usage.input_tokens == 300


def test_two_hundred_sequential_calls_stay_valid(stub: _StubRouter) -> None:
    """200 calls on one session stay valid and reuse one pooled connection."""
    with _session(stub) as session:
        for _ in range(_CALLS):
            _judge_valid(session)
    assert stub.state.completions == _CALLS
    assert stub.state.props_calls == 1
    assert stub.state.connections == 1


def test_restart_mid_run_gives_typed_outcome_then_recovers(
    stub: _StubRouter,
) -> None:
    """A restart gives one typed error or success; the next call recovers."""
    with _session(stub) as session:
        for _ in range(_RESTART_AT):
            _judge_valid(session)
        stub.stop()
        stub.start()
        with suppress(TransportError):
            _judge_valid(session)
        for _ in range(_CALLS - _RESTART_AT - 1):
            _judge_valid(session)
    assert stub.state.completions >= _CALLS - 1
    assert stub.state.connections == 2


def test_call_while_stub_is_down_raises_transport_error(stub: _StubRouter) -> None:
    """A call while the stub is down raises ``TransportError``; restart recovers."""
    with _session(stub) as session:
        _judge_valid(session)
        stub.stop()
        with pytest.raises(TransportError) as info:
            _judge_valid(session)
        assert type(info.value) is TransportError
        stub.start()
        _judge_valid(session)


def test_tokenize_error_status_raises_backend_http_error(stub: _StubRouter) -> None:
    """An error status on ``/tokenize`` raises ``BackendHttpError``."""
    stub.state.tokenize_status = 500
    with _session(stub) as session, pytest.raises(BackendHttpError) as info:
        session.port.judge("Look.", _QUESTIONS, _MODEL, media=(_PNG,))
    assert info.value.status_code == 500
    assert stub.state.completions == 0


@pytest.mark.parametrize(
    "raw",
    [
        ("text/plain", b"oops"),
        ("application/json", b'{"no_tokens": []}'),
        ("application/json", b'{"tokens": "abc"}'),
        ("application/json", b'{"tokens": [1, "x"]}'),
        ("application/json", b"[1, 2]"),
    ],
)
def test_malformed_tokenize_body_raises_generation_error(
    stub: _StubRouter, raw: tuple[str, bytes]
) -> None:
    """A 200 ``/tokenize`` body with no usable ``tokens`` raises ``GenerationError``."""
    stub.state.tokenize_raw = raw
    with _session(stub) as session, pytest.raises(GenerationError) as info:
        session.port.judge("Look.", _QUESTIONS, _MODEL, media=(_PNG,))
    assert type(info.value) is GenerationError
    assert "llama.cpp" in str(info.value)
    assert stub.state.completions == 0
