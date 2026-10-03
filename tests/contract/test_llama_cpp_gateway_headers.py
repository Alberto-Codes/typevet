"""Contract tests: llama.cpp clients send the key and gateway headers (#410).

The composition-root clients from ``generation_adapter``,
``async_llama_http_client`` and ``open_judgment`` run the real llama.cpp
adapters over ``httpx.MockTransport``. The mock plays a router behind a
gateway: it records each request and answers each llama.cpp route. The tests
prove the default Bearer key, a bare key in a custom header, the extra
headers, the user agent, no added header when nothing is set, and that no
key leaves through an error chain.

[i410]: https://github.com/Alberto-Codes/typevet/issues/410
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from typevet.adapters.inbound.backend_settings import generation_adapter, open_judgment
from typevet.adapters.inbound.settings import (
    async_llama_http_client,
    load_llama_settings,
)
from typevet.adapters.outbound.llama_cpp.generation_async import (
    AsyncLlamaCppGenerationAdapter,
)
from typevet.domain.errors import GenerationError
from typevet.domain.judgment_questions import Noul
from typevet.domain.models import GenerationRequest

pytestmark = pytest.mark.contract

_BASE = "http://router.example"
_KEY = "sk-LLAMA-GATEWAY-SENTINEL-410"
_TENANT = "acme-tenant-SENTINEL-410"
_AGENT = "typevet-test/410"
_SCHEMA = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"],
    "additionalProperties": False,
}
_REQUEST = GenerationRequest(prompt="hi", schema=_SCHEMA, model="llama-model")
_GEMMA4_RENDERED = "<|turn>user\nhello<turn|>\n<|turn>model\n"


class _Router:
    """Offline llama.cpp router that records each request.

    Attributes:
        requests (list[httpx.Request]): Requests in arrival order.
        fail_path (str | None): Path that raises a transport error which
            echoes the key, as a proxy error text can.

    Examples:
        ```python
        router = _Router()
        transport = httpx.MockTransport(router.handle)
        ```
    """

    def __init__(self, fail_path: str | None = None) -> None:
        self.requests: list[httpx.Request] = []
        self.fail_path = fail_path

    def handle(self, request: httpx.Request) -> httpx.Response:
        """Return a canned llama.cpp answer for the request path.

        Args:
            request: Request sent by a typevet client.

        Returns:
            A canned response for a known path, else HTTP 404.

        Raises:
            httpx.ConnectError: On ``fail_path``; the text holds the key.
        """
        self.requests.append(request)
        path = request.url.path
        if path == self.fail_path:
            msg = f"proxy refused credential {_KEY}"
            raise httpx.ConnectError(msg, request=request)
        if path == "/v1/chat/completions":
            reply = {"choices": [{"message": {"content": '{"ok": true}'}}]}
            return httpx.Response(200, json=reply)
        if path == "/apply-template":
            return httpx.Response(200, json={"prompt": _GEMMA4_RENDERED})
        if path == "/props":
            modalities = {"vision": True, "audio": False}
            return httpx.Response(200, json={"modalities": modalities})
        if path == "/tokenize":
            content = json.loads(request.content.decode())["content"]
            token_id = 100 + sum(map(ord, content)) % 50
            return httpx.Response(200, json={"tokens": [token_id]})
        if path == "/completion":
            top = [{"id": i, "logprob": -0.2 - (i % 5) * 0.15} for i in range(100, 180)]
            body = {"completion_probabilities": [{"top_logprobs": top}]}
            return httpx.Response(200, json=body)
        return httpx.Response(404)


def _env(**extra: str) -> dict[str, str]:
    return {"TYPEVET_BACKEND": "llama_cpp", "TYPEVET_LLAMA__BASE_URL": _BASE, **extra}


def _generate(env: dict[str, str], router: _Router) -> None:
    transport = httpx.MockTransport(router.handle)
    with generation_adapter(env, transport=transport) as port:
        port.generate(_REQUEST)


def _judge(env: dict[str, str], router: _Router) -> None:
    transport = httpx.MockTransport(router.handle)
    with open_judgment(env, transport=transport) as session:
        session.port.judge("state", {"f": Noul()}, session.model)


def _generate_async(env: dict[str, str], router: _Router) -> None:
    settings = load_llama_settings(env)

    async def run() -> None:
        transport = httpx.MockTransport(router.handle)
        async with async_llama_http_client(settings, transport=transport) as client:
            adapter = AsyncLlamaCppGenerationAdapter(
                settings.base_url, timeout=settings.timeout, client=client
            )
            await adapter.generate(_REQUEST)

    asyncio.run(run())


Drive = Callable[[dict[str, str], _Router], None]
_CLIENTS = pytest.mark.parametrize(
    "drive",
    [_generate, _judge, _generate_async],
    ids=["generation_adapter", "open_judgment", "async_llama_http_client"],
)


def _sent(drive: Drive, env: dict[str, str]) -> list[httpx.Request]:
    router = _Router()
    drive(env, router)
    assert router.requests
    return router.requests


@_CLIENTS
def test_bearer_key_by_default(drive: Drive) -> None:
    for request in _sent(drive, _env(TYPEVET_LLAMA__API_KEY=_KEY)):
        assert request.headers.get("Authorization") == f"Bearer {_KEY}"


@_CLIENTS
def test_bare_key_in_custom_auth_header(drive: Drive) -> None:
    env = _env(
        TYPEVET_LLAMA__API_KEY=_KEY,
        TYPEVET_LLAMA__AUTH_HEADER="X-Api-Key",
        TYPEVET_LLAMA__AUTH_SCHEME="",
    )
    for request in _sent(drive, env):
        assert request.headers.get("X-Api-Key") == _KEY
        assert "Authorization" not in request.headers


@_CLIENTS
def test_extra_headers_sent(drive: Drive) -> None:
    env = _env(TYPEVET_LLAMA__HEADERS=json.dumps({"X-Tenant": _TENANT}))
    for request in _sent(drive, env):
        assert request.headers.get("X-Tenant") == _TENANT


@_CLIENTS
def test_user_agent_sent(drive: Drive) -> None:
    for request in _sent(drive, _env(TYPEVET_LLAMA__USER_AGENT=_AGENT)):
        assert request.headers.get("User-Agent") == _AGENT


@_CLIENTS
def test_no_added_header_when_unset(drive: Drive) -> None:
    for request in _sent(drive, _env()):
        assert "Authorization" not in request.headers
        assert "X-Api-Key" not in request.headers
        assert "X-Tenant" not in request.headers
        assert request.headers["User-Agent"].startswith("python-httpx/")


def _chain(exc: BaseException) -> list[BaseException]:
    links: list[BaseException] = []
    pending: list[BaseException | None] = [exc]
    while pending:
        link = pending.pop()
        if link is None or any(link is seen for seen in links):
            continue
        links.append(link)
        pending += [link.__cause__, link.__context__]
    return links


def _leaks(link: BaseException) -> list[str]:
    texts = [str(link), repr(link), repr(link.args), repr(vars(link))]
    request: Any = getattr(link, "_request", None)
    if isinstance(request, httpx.Request):
        texts.append(repr(dict(request.headers)))
    return [text for text in texts if _KEY in text]


@pytest.mark.parametrize(
    ("drive", "fail_path"),
    [
        (_generate, "/v1/chat/completions"),
        (_judge, "/props"),
        (_judge, "/completion"),
    ],
    ids=["generate", "judgment-open", "judge"],
)
def test_transport_error_chain_never_holds_the_key(
    drive: Drive, fail_path: str
) -> None:
    router = _Router(fail_path=fail_path)
    with pytest.raises(GenerationError) as info:
        drive(_env(TYPEVET_LLAMA__API_KEY=_KEY), router)
    assert any(request.url.path == fail_path for request in router.requests)
    for link in _chain(info.value):
        assert _leaks(link) == [], type(link).__name__


@pytest.mark.parametrize(
    ("drive", "fail_path"),
    [(_generate, "/v1/chat/completions"), (_judge, "/props")],
    ids=["generate", "judgment-open"],
)
def test_keyless_transport_error_keeps_its_cause(drive: Drive, fail_path: str) -> None:
    with pytest.raises(GenerationError) as info:
        drive(_env(), _Router(fail_path=fail_path))
    assert isinstance(info.value.__cause__, httpx.ConnectError)


def test_generation_adapter_close_closes_its_client() -> None:
    transport = httpx.MockTransport(_Router().handle)
    adapter = generation_adapter(_env(), transport=transport)
    client = adapter._client
    assert isinstance(client, httpx.Client)
    assert not client.is_closed
    adapter.close()
    assert client.is_closed
