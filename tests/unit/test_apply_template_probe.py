"""Unit tests: typed errors from the ``/apply-template`` probe ([#311][i311]).

A stub llama.cpp router runs on a loopback socket (``127.0.0.1``, port chosen
by the kernel). ``/props`` reports vision. The stub then sends a fault on
``/apply-template``, which the factory calls when it opens a session. No
request leaves the host.

A close before a response raises ``TransportError`` after one retry. An error
status raises ``BackendHttpError``. A 200 body that is not JSON, or that has no
``prompt`` string, raises ``GenerationError``. A served Gemma 3 template raises
``ValueError`` when the caller requires Gemma 4.

Examples:
    ```bash
    uv run pytest -q tests/unit/test_apply_template_probe.py
    ```

See Also:
    - [typevet.adapters.outbound.llama_cpp.gemma_native_vision_factory][]: Factory
    - [typevet.adapters.outbound.llama_cpp.http_mapping][]: Error mapping

[i311]: https://github.com/Alberto-Codes/typevet/issues/311
"""

from __future__ import annotations

import json
import threading
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from typevet.adapters.inbound.settings import load_llama_settings
from typevet.adapters.outbound.llama_cpp.gemma_native_vision_factory import (
    open_gemma_native_vision_judgment,
)
from typevet.domain.errors import BackendHttpError, GenerationError, TransportError

pytestmark = pytest.mark.unit

_MODEL = "gemma-4-apply-template"
_PATH = "/apply-template"
_GEMMA4_RENDERED = "<|turn>user\nhello<turn|>\n<|turn>model\n"
_GEMMA3_RENDERED = "<start_of_turn>user\nhello<end_of_turn>\n<start_of_turn>model\n"
_PROPS = {"modalities": {"vision": True}, "media_marker": "<__media__>"}


@dataclass
class _StubState:
    """Fault settings and counters for ``/apply-template``.

    Attributes:
        requests (Counter[str]): Requests the stub read, by path.
        close (bool): Close the connection with no response on the probe.
        status (int): HTTP status the stub sends for the probe.
        raw (tuple[str, bytes] | None): Content type and raw body of a 200
            probe reply that replaces the JSON when set.
    """

    requests: Counter[str] = field(default_factory=Counter)
    close: bool = False
    status: int = 200
    raw: tuple[str, bytes] | None = None


def _handler(state: _StubState) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send(self, status: int, content_type: str, payload: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _dispatch(self) -> None:
            path = self.path.split("?")[0]
            state.requests[path] += 1
            if path != _PATH:
                self._send(200, "application/json", json.dumps(_PROPS).encode())
            elif state.close:
                self.close_connection = True
            elif state.raw is not None:
                self._send(200, *state.raw)
            else:
                reply = {"prompt": _GEMMA4_RENDERED}
                if state.status != 200:
                    reply = {"error": "stub"}
                self._send(state.status, "application/json", json.dumps(reply).encode())

        def do_GET(self) -> None:
            self._dispatch()

        def do_POST(self) -> None:
            self.rfile.read(int(self.headers.get("Content-Length", "0")))
            self._dispatch()

        def log_message(self, format: str, *args: Any) -> None:
            del format, args

    return Handler


@pytest.fixture
def stub() -> Iterator[tuple[_StubState, str]]:
    """Start one stub router and stop it after the test."""
    state = _StubState()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _handler(state))
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield state, f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def _open(base_url: str, *, require_gemma4: bool = True) -> None:
    settings = load_llama_settings(
        {
            "TYPEVET_LLAMA__BASE_URL": base_url,
            "TYPEVET_LLAMA__TIMEOUT": "5",
            "TYPEVET_LLAMA__MULTIMODAL_MODEL": _MODEL,
        }
    )
    with open_gemma_native_vision_judgment(
        settings=settings, require_gemma4=require_gemma4
    ):
        pass


def test_healthy_probe_opens_session(stub: tuple[_StubState, str]) -> None:
    """A 200 ``prompt`` string opens the session with one probe request."""
    state, base_url = stub
    _open(base_url)
    assert state.requests[_PATH] == 1


def test_close_before_response_raises_transport_error(
    stub: tuple[_StubState, str],
) -> None:
    """A close with no response raises ``TransportError`` after one retry."""
    state, base_url = stub
    state.close = True
    with pytest.raises(TransportError) as info:
        _open(base_url)
    assert type(info.value) is TransportError
    assert state.requests[_PATH] == 2


def test_error_status_raises_backend_http_error(stub: tuple[_StubState, str]) -> None:
    """An error status on ``/apply-template`` raises ``BackendHttpError``."""
    state, base_url = stub
    state.status = 500
    with pytest.raises(BackendHttpError) as info:
        _open(base_url)
    assert info.value.status_code == 500
    assert state.requests[_PATH] == 1


@pytest.mark.parametrize(
    "raw",
    [
        ("text/plain", b"oops", "llama.cpp returned non-JSON HTTP body"),
        (
            "application/json",
            b'{"no_prompt": "x"}',
            "llama.cpp /apply-template response missing a prompt string",
        ),
        (
            "application/json",
            b'{"prompt": 7}',
            "llama.cpp /apply-template response missing a prompt string",
        ),
        (
            "application/json",
            b'["prompt"]',
            "llama.cpp /apply-template response missing a prompt string",
        ),
    ],
)
def test_malformed_body_raises_generation_error(
    stub: tuple[_StubState, str], raw: tuple[str, bytes, str]
) -> None:
    """A 200 body with no ``prompt`` string raises ``GenerationError``."""
    state, base_url = stub
    content_type, body, prefix = raw
    state.raw = (content_type, body)
    with pytest.raises(GenerationError) as info:
        _open(base_url)
    assert type(info.value) is GenerationError
    assert str(info.value).startswith(prefix)
    assert "llama.cpp" in str(info.value)


def test_gemma3_template_raises_when_gemma4_is_required(
    stub: tuple[_StubState, str],
) -> None:
    """A served Gemma 3 template fails the Gemma 4 requirement."""
    state, base_url = stub
    state.raw = ("application/json", json.dumps({"prompt": _GEMMA3_RENDERED}).encode())
    with pytest.raises(ValueError, match=r"^expected NATIVE_GEMMA4_TURN, got") as info:
        _open(base_url)
    assert str(info.value).endswith("native_gemma3_turn")
    assert state.requests[_PATH] == 1


def test_gemma3_template_opens_when_gemma4_is_not_required(
    stub: tuple[_StubState, str],
) -> None:
    """The same Gemma 3 template opens a session without the requirement."""
    state, base_url = stub
    state.raw = ("application/json", json.dumps({"prompt": _GEMMA3_RENDERED}).encode())
    _open(base_url, require_gemma4=False)
    assert state.requests[_PATH] == 1
