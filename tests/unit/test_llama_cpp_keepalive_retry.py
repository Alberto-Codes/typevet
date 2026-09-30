"""Unit tests: one retry when llama.cpp closes a reused connection ([#305][i305]).

A stub llama.cpp router runs on a loopback socket (``127.0.0.1``, port chosen
by the kernel). On request, the stub reads the next request on a reused
keep-alive connection and then closes that connection without a response.
This is the race from [#301][i301]: ``httpx`` raises ``RemoteProtocolError``
("Server disconnected without sending a response").

The idempotent scoring calls (``/tokenize``, ``/completion`` with
``n_predict: 0``, ``/props``) retry once on a fresh connection. A second
disconnect, a malformed response, an error status and a timeout do not
retry. No request leaves the host.

Examples:
    ```bash
    uv run pytest -q tests/unit/test_llama_cpp_keepalive_retry.py
    ```

See Also:
    - [typevet.adapters.outbound.llama_cpp.http_mapping][]: Retry helper
    - [typevet.adapters.outbound.llama_cpp.scoring][]: Scoring adapter
    - [typevet.adapters.outbound.llama_cpp.gemma_native_vision_factory][]: Factory

[i301]: https://github.com/Alberto-Codes/typevet/issues/301
[i305]: https://github.com/Alberto-Codes/typevet/issues/305
"""

from __future__ import annotations

import json
import socket
import threading
import time
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from typevet.adapters.inbound.settings import load_llama_settings
from typevet.adapters.outbound.llama_cpp.gemma_native_vision_factory import (
    GemmaNativeVisionSession,
    open_gemma_native_vision_judgment,
)
from typevet.adapters.outbound.llama_cpp.scoring import (
    LlamaCppCandidateScoringAdapter,
)
from typevet.domain.candidate_scoring_request import (
    CandidateScoringRequest,
    CandidateTokenSpec,
)
from typevet.domain.errors import BackendHttpError, TransportError
from typevet.domain.judgment_answers import NoulAnswer
from typevet.domain.judgment_questions import Noul
from typevet.domain.media import ImageInput

pytestmark = pytest.mark.unit

_MODEL = "gemma-4-keepalive"
_GEMMA4_RENDERED = "<|turn>user\nhello<turn|>\n<|turn>model\n"
_MARKER = "<__media__>"
_PNG = ImageInput(data=b"\x89PNG\r\n\x1a\nstub", mime_type="image/png")
_QUESTIONS = {"filled": Noul(instructions="Is the image filled with one colour?")}
_TIMEOUT = 0.5
_SCORING_PATHS = ("/tokenize", "/completion")


@dataclass
class _StubState:
    """Counters and per-path faults for the stub router.

    Attributes:
        connections (int): TCP connections the stub accepted.
        requests (Counter[str]): Requests the stub read, by path.
        drops (Counter[str]): Reused-connection requests to close without a
            response, by path.
        drop_fresh (bool): Also close a fresh connection when ``drops`` is armed.
        garbage (Counter[str]): Requests to answer with malformed bytes, by path.
        status (dict[str, int]): Error status to send, by path.
        delay (dict[str, float]): Seconds to wait before a reply, by path.
    """

    connections: int = 0
    requests: Counter[str] = field(default_factory=Counter)
    drops: Counter[str] = field(default_factory=Counter)
    drop_fresh: bool = False
    garbage: Counter[str] = field(default_factory=Counter)
    status: dict[str, int] = field(default_factory=dict)
    delay: dict[str, float] = field(default_factory=dict)


def _route(path: str, body: dict[str, Any]) -> dict[str, Any]:
    """Return the JSON body for one stub route.

    Returns:
        Canned llama.cpp JSON for ``path``.
    """
    if path == "/apply-template":
        return {"prompt": _GEMMA4_RENDERED}
    if path == "/props":
        return {"modalities": {"vision": True}, "media_marker": _MARKER}
    if path == "/tokenize":
        return {"tokens": [100 + sum(map(ord, body["content"])) % 50]}
    top = [{"id": i, "logprob": -0.5 - i * 0.001} for i in range(100, 150)]
    return {"completion_probabilities": [{"top_logprobs": top}], "tokens_evaluated": 3}


def _handler(state: _StubState) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        disable_nagle_algorithm = True
        wbufsize = -1
        served = 0

        def _dispatch(self, body: dict[str, Any]) -> None:
            path = self.path.split("?")[0]
            state.requests[path] += 1
            reused = self.served > 0
            self.served += 1
            if (reused or state.drop_fresh) and state.drops[path] > 0:
                state.drops[path] -= 1
                self.close_connection = True
                return
            if state.garbage[path] > 0:
                state.garbage[path] -= 1
                self.wfile.write(b"NOT-HTTP garbage\r\n\r\n")
                self.close_connection = True
                return
            time.sleep(state.delay.get(path, 0.0))
            status = state.status.get(path, 200)
            reply = {"error": "stub"} if status != 200 else _route(path, body)
            payload = json.dumps(reply).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self) -> None:
            self._dispatch({})

        def do_POST(self) -> None:
            length = int(self.headers.get("Content-Length", "0"))
            self._dispatch(json.loads(self.rfile.read(length) or b"{}"))

        def log_message(self, format: str, *args: Any) -> None:
            del format, args

    return Handler


class _StubRouter:
    """Loopback llama.cpp stub with per-path connection faults.

    Attributes:
        state (_StubState): Counters and armed faults.
        base_url (str): Stub root URL.
    """

    def __init__(self) -> None:
        self.state = _StubState()
        state = self.state

        class Server(ThreadingHTTPServer):
            daemon_threads = True

            def get_request(self) -> tuple[socket.socket, Any]:
                conn, addr = super().get_request()
                state.connections += 1
                return conn, addr

        self._server = Server(("127.0.0.1", 0), _handler(state))
        self.base_url = f"http://127.0.0.1:{self._server.server_address[1]}"
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def stop(self) -> None:
        """Stop listening."""
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture
def stub() -> Iterator[_StubRouter]:
    """Start one stub router and stop it after the test."""
    router = _StubRouter()
    yield router
    router.stop()


@contextmanager
def _session(stub: _StubRouter) -> Iterator[GemmaNativeVisionSession]:
    settings = load_llama_settings(
        {
            "TYPEVET_LLAMA__BASE_URL": stub.base_url,
            "TYPEVET_LLAMA__TIMEOUT": str(_TIMEOUT),
            "TYPEVET_LLAMA__MULTIMODAL_MODEL": _MODEL,
        }
    )
    with open_gemma_native_vision_judgment(settings=settings) as session:
        yield session


def _judge(session: GemmaNativeVisionSession) -> None:
    response = session.port.judge("Look.", _QUESTIONS, _MODEL, media=(_PNG,))
    assert isinstance(response.answers["filled"], NoulAnswer)


def _warm(stub: _StubRouter, session: GemmaNativeVisionSession) -> Counter[str]:
    """Run one judgment and return the per-path request count it used.

    Returns:
        Requests one clean ``judge`` call sends, by path.
    """
    before = Counter(stub.state.requests)
    _judge(session)
    used = Counter(stub.state.requests)
    used.subtract(before)
    return used


@pytest.mark.parametrize("path", _SCORING_PATHS)
def test_judge_retries_once_after_reused_connection_closes(
    stub: _StubRouter, path: str
) -> None:
    """One close on a reused connection retries once and ``judge`` succeeds."""
    with _session(stub) as session:
        per_call = _warm(stub, session)
        assert per_call[path] >= 1
        start = stub.state.requests[path]
        stub.state.drops[path] = 1
        _judge(session)
    assert stub.state.drops[path] == 0
    assert stub.state.requests[path] - start == per_call[path] + 1
    assert stub.state.connections == 2


@pytest.mark.parametrize("path", _SCORING_PATHS)
def test_second_disconnect_raises_transport_error(stub: _StubRouter, path: str) -> None:
    """A disconnect on the retry raises ``TransportError`` after two requests."""
    with _session(stub) as session:
        _warm(stub, session)
        start = stub.state.requests[path]
        stub.state.drops[path] = 3
        stub.state.drop_fresh = True
        with pytest.raises(TransportError) as info:
            _judge(session)
    assert type(info.value) is TransportError
    assert "Server disconnected" in str(info.value)
    assert stub.state.requests[path] - start == 2
    assert stub.state.drops[path] == 1


def _scoring_request(*, media: bool) -> CandidateScoringRequest:
    return CandidateScoringRequest(
        model=_MODEL,
        prefix=f"{_MARKER}Answer:" if media else "Answer:",
        candidates=(CandidateTokenSpec("a", (101,)),),
        media=(_PNG,) if media else (),
    )


def test_props_probe_retries_once_after_reused_connection_closes(
    stub: _StubRouter,
) -> None:
    """The ``/props`` probe retries once on a fresh connection."""
    with LlamaCppCandidateScoringAdapter(
        base_url=stub.base_url, timeout=_TIMEOUT, n_vocab=64
    ) as adapter:
        adapter.score_candidates(_scoring_request(media=False))
        stub.state.drops["/props"] = 1
        result = adapter.score_candidates(_scoring_request(media=True))
    assert result.model == _MODEL
    assert stub.state.requests["/props"] == 2
    assert stub.state.connections == 2


def test_props_probe_second_disconnect_raises_transport_error(
    stub: _StubRouter,
) -> None:
    """A second ``/props`` disconnect raises ``TransportError``."""
    with LlamaCppCandidateScoringAdapter(
        base_url=stub.base_url, timeout=_TIMEOUT, n_vocab=64
    ) as adapter:
        adapter.score_candidates(_scoring_request(media=False))
        stub.state.drops["/props"] = 3
        stub.state.drop_fresh = True
        with pytest.raises(TransportError):
            adapter.score_candidates(_scoring_request(media=True))
    assert stub.state.requests["/props"] == 2


@pytest.mark.parametrize("path", _SCORING_PATHS)
def test_malformed_response_does_not_retry(stub: _StubRouter, path: str) -> None:
    """Bytes that are not HTTP raise ``TransportError`` with no retry."""
    with _session(stub) as session:
        _warm(stub, session)
        start = stub.state.requests[path]
        stub.state.garbage[path] = 1
        with pytest.raises(TransportError):
            _judge(session)
    assert stub.state.requests[path] - start == 1


@pytest.mark.parametrize("path", _SCORING_PATHS)
def test_error_status_does_not_retry(stub: _StubRouter, path: str) -> None:
    """An error status raises ``BackendHttpError`` with no retry."""
    with _session(stub) as session:
        _warm(stub, session)
        start = stub.state.requests[path]
        stub.state.status[path] = 503
        with pytest.raises(BackendHttpError) as info:
            _judge(session)
    assert info.value.status_code == 503
    assert stub.state.requests[path] - start == 1


@pytest.mark.parametrize("path", _SCORING_PATHS)
def test_timeout_does_not_retry(stub: _StubRouter, path: str) -> None:
    """A read timeout raises ``TransportError`` with no retry."""
    with _session(stub) as session:
        _warm(stub, session)
        start = stub.state.requests[path]
        stub.state.delay[path] = _TIMEOUT * 3
        with pytest.raises(TransportError) as info:
            _judge(session)
    assert "timed out" in str(info.value)
    assert stub.state.requests[path] - start == 1
