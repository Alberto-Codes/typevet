"""Contract tests: a llama.cpp gateway log line matches a judgment (#411).

The composition-root clients from ``generation_adapter``,
``async_llama_http_client`` and ``open_judgment`` run the real llama.cpp
adapters over ``httpx.MockTransport``. The mock plays a router behind a
gateway and keeps each request. With ``TYPEVET_LLAMA__REQUEST_ID_HEADER``
set, each request carries a fresh UUID4 hex id, and
``JudgmentResponse.request_ids`` maps each question to the id of its
``/completion`` request. Without it, no request carries the header and
``request_ids`` stays empty. The llama.cpp clients get the request hook only,
not the vLLM response guard.

[i411]: https://github.com/Alberto-Codes/typevet/issues/411
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable

import httpx
import pytest

from typevet.adapters.inbound.backend_settings import generation_adapter, open_judgment
from typevet.adapters.inbound.settings import (
    async_llama_http_client,
    llama_http_client,
    load_llama_settings,
)
from typevet.adapters.outbound.llama_cpp.generation_async import (
    AsyncLlamaCppGenerationAdapter,
)
from typevet.domain.judgment_questions import Noul
from typevet.domain.judgment_response import JudgmentResponse
from typevet.domain.models import GenerationRequest

pytestmark = pytest.mark.contract

_BASE = "http://router.example"
_HEADER = "X-Request-Id"
_SCHEMA = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"],
    "additionalProperties": False,
}
_REQUEST = GenerationRequest(prompt="hi", schema=_SCHEMA, model="llama-model")
_GEMMA4_RENDERED = "<|turn>user\nhello<turn|>\n<|turn>model\n"
_QUESTIONS = {
    "first": Noul(instructions="Is it ok?"),
    "second": Noul(instructions="Is it polite?"),
}
_HEX = re.compile(r"[0-9a-f]{32}")
_TOP = [{"id": i, "logprob": -0.2 - (i % 5) * 0.15} for i in range(100, 180)]
_CANNED: dict[str, object] = {
    "/v1/chat/completions": {"choices": [{"message": {"content": '{"ok": true}'}}]},
    "/apply-template": {"prompt": _GEMMA4_RENDERED},
    "/props": {"modalities": {"vision": True, "audio": False}},
    "/completion": {"completion_probabilities": [{"top_logprobs": _TOP}]},
}


class _Router:
    """Offline llama.cpp router that keeps each request.

    Attributes:
        requests (list[httpx.Request]): Requests in arrival order.
        redirect (bool): Answer ``/v1/chat/completions`` with HTTP 302.
    """

    def __init__(self, *, redirect: bool = False) -> None:
        self.requests: list[httpx.Request] = []
        self.redirect = redirect

    def handle(self, request: httpx.Request) -> httpx.Response:
        """Return a canned llama.cpp answer for the request path.

        Args:
            request: Request sent by a typevet client.

        Returns:
            A canned response for a known path, else HTTP 404.
        """
        self.requests.append(request)
        path = request.url.path
        if path == "/v1/chat/completions" and self.redirect:
            return httpx.Response(302, headers={"Location": "https://login/"})
        if path == "/tokenize":
            content = json.loads(request.content.decode())["content"]
            token_id = 100 + sum(map(ord, content)) % 50
            return httpx.Response(200, json={"tokens": [token_id]})
        body = _CANNED.get(path)
        if body is None:
            return httpx.Response(404)
        return httpx.Response(200, json=body)

    def ids(self, path: str | None = None) -> list[str | None]:
        """Return the request-id header of each kept request on ``path``.

        Args:
            path: Request path, or ``None`` for every request.

        Returns:
            The header values in arrival order; ``None`` where absent.
        """
        return [
            request.headers.get(_HEADER)
            for request in self.requests
            if path is None or request.url.path == path
        ]


def _env(*, request_id: bool = True) -> dict[str, str]:
    env = {"TYPEVET_BACKEND": "llama_cpp", "TYPEVET_LLAMA__BASE_URL": _BASE}
    if request_id:
        env["TYPEVET_LLAMA__REQUEST_ID_HEADER"] = _HEADER
    return env


def _judge(env: dict[str, str], router: _Router) -> JudgmentResponse:
    transport = httpx.MockTransport(router.handle)
    with open_judgment(env, transport=transport) as session:
        return session.port.judge("state", _QUESTIONS, session.model)


def _generate_twice(env: dict[str, str], router: _Router) -> None:
    transport = httpx.MockTransport(router.handle)
    with generation_adapter(env, transport=transport) as port:
        port.generate(_REQUEST)
        port.generate(_REQUEST)


def _generate_async_twice(env: dict[str, str], router: _Router) -> None:
    settings = load_llama_settings(env)

    async def run() -> None:
        transport = httpx.MockTransport(router.handle)
        async with async_llama_http_client(settings, transport=transport) as client:
            adapter = AsyncLlamaCppGenerationAdapter(
                settings.base_url, timeout=settings.timeout, client=client
            )
            await adapter.generate(_REQUEST)
            await adapter.generate(_REQUEST)

    asyncio.run(run())


Drive = Callable[[dict[str, str], _Router], object]
_GENERATION = pytest.mark.parametrize(
    "drive", [_generate_twice, _generate_async_twice], ids=["sync", "async"]
)


def test_judgment_records_the_completion_request_id_of_each_question() -> None:
    router = _Router()
    response = _judge(_env(), router)
    sent = router.ids("/completion")
    assert response.request_ids == dict(zip(_QUESTIONS, sent, strict=True))
    assert all(_HEX.fullmatch(value or "") for value in sent)
    assert len(set(sent)) == len(sent)


def test_every_judgment_request_carries_a_fresh_id() -> None:
    router = _Router()
    _judge(_env(), router)
    sent = router.ids()
    assert {request.url.path for request in router.requests} >= {
        "/props",
        "/apply-template",
        "/tokenize",
        "/completion",
    }
    assert all(_HEX.fullmatch(value or "") for value in sent)
    assert len(set(sent)) == len(sent)


@_GENERATION
def test_fresh_hex_id_per_generation_call(drive: Drive) -> None:
    router = _Router()
    drive(_env(), router)
    sent = router.ids("/v1/chat/completions")
    assert len(sent) == 2
    for value in sent:
        assert value is not None, f"missing {_HEADER} header"
        assert _HEX.fullmatch(value)
    assert sent[0] != sent[1]


@_GENERATION
def test_unset_header_sends_no_id_on_generation(drive: Drive) -> None:
    router = _Router()
    drive(_env(request_id=False), router)
    assert router.requests
    assert router.ids() == [None] * len(router.requests)


def test_unset_header_sends_no_id_and_records_none() -> None:
    router = _Router()
    response = _judge(_env(request_id=False), router)
    assert router.requests
    assert router.ids() == [None] * len(router.requests)
    assert response.request_ids == {}


def test_caller_set_id_is_kept() -> None:
    settings = load_llama_settings(_env())
    router = _Router()
    transport = httpx.MockTransport(router.handle)
    with llama_http_client(settings, transport=transport) as client:
        client.post("/completion", json={}, headers={_HEADER: "caller-id-411"})
    assert router.ids() == ["caller-id-411"]


def test_llama_clients_get_the_request_hook_only() -> None:
    settings = load_llama_settings(_env())
    with llama_http_client(settings) as client:
        assert len(client.event_hooks["request"]) == 1
        assert client.event_hooks["response"] == []
    hooks = async_llama_http_client(settings).event_hooks
    assert len(hooks["request"]) == 1
    assert hooks["response"] == []


def test_redirect_is_not_refused_by_a_vllm_guard() -> None:
    settings = load_llama_settings(_env())
    router = _Router(redirect=True)
    transport = httpx.MockTransport(router.handle)
    with llama_http_client(settings, transport=transport) as client:
        response = client.post("/v1/chat/completions", json={})
    assert response.status_code == 302
    assert _HEX.fullmatch(response.request.headers[_HEADER])
