"""Contract tests: a gateway log line matches a judgment or an error (#356).

The composition-root clients from ``generation_adapter``,
``async_vllm_generation_adapter`` and ``open_judgment`` run the real vLLM
adapters over ``httpx.MockTransport``. The mock plays the gateway and keeps
each request. With ``TYPEVET_VLLM__REQUEST_ID_HEADER`` set, the id in
``JudgmentResponse.request_ids`` and in ``request_id`` on an error must equal
the id that the gateway saw. Without it, nothing records an id.

[i356]: https://github.com/Alberto-Codes/typevet/issues/356
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from typevet.adapters.inbound.backend_settings import (
    async_vllm_generation_adapter,
    generation_adapter,
    open_judgment,
)
from typevet.adapters.inbound.gateway_headers import sync_event_hooks
from typevet.adapters.outbound.vllm.http_mapping import map_transport_error
from typevet.adapters.outbound.vllm.judgment_factory import open_vllm_judgment
from typevet.domain.candidate_scoring_request import CandidateScoringRequest
from typevet.domain.candidate_scoring_response import CandidateScoringResult
from typevet.domain.errors import BackendHttpError, GenerationError, TransportError
from typevet.domain.judgment_questions import Noul
from typevet.domain.judgment_response import JudgmentResponse
from typevet.domain.models import GenerationRequest
from typevet.ports.scoring import CandidateScoringPort

pytestmark = pytest.mark.contract

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "vllm"
_BASE = "https://gw.example.com/vllm"
_MODEL = "served-model"
_KEY = "sk-GATEWAY-SENTINEL-356"
_HEADER = "X-Request-Id"
_SCORING_PATH = "/vllm/v1/chat/completions"
_SCHEMA = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"],
    "additionalProperties": False,
}
_REPLY = {"choices": [{"message": {"content": '{"ok": true}'}}]}
_QUESTIONS = {
    "first": Noul(instructions="Is it ok?"),
    "second": Noul(instructions="Is it polite?"),
}
_HEX = re.compile(r"[0-9a-f]{32}")
Handler = Callable[[httpx.Request], httpx.Response]


def _fixture(name: str) -> dict[str, Any]:
    return json.loads((_FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def _env(*, request_id: bool = True) -> dict[str, str]:
    env = {
        "TYPEVET_BACKEND": "vllm",
        "TYPEVET_VLLM__BASE_URL": _BASE,
        "TYPEVET_VLLM__MODEL": _MODEL,
        "TYPEVET_VLLM__API_KEY": _KEY,
        "TYPEVET_VLLM__HEADERS": json.dumps({"X-Tenant": "acme"}),
    }
    if request_id:
        env["TYPEVET_VLLM__REQUEST_ID_HEADER"] = _HEADER
    return env


class _Gateway:
    """MockTransport handler that keeps each request and serves vLLM."""

    def __init__(self, fail: Handler | None = None) -> None:
        self.requests: list[httpx.Request] = []
        self._fail = fail
        self._scoring = _fixture("text_three_way")["response"]
        self._tokens = _fixture("tokenize_ordinals")["responses"]

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if self._fail is not None:
            return self._fail(request)
        body = json.loads(request.content.decode())
        if request.url.path == "/vllm/tokenize":
            return httpx.Response(200, json=self._tokens[body["prompt"]])
        reply = self._scoring if "logprobs" in body else _REPLY
        return httpx.Response(200, json=reply)

    def scoring_ids(self) -> list[str | None]:
        return [
            request.headers.get(_HEADER)
            for request in self.requests
            if request.url.path == _SCORING_PATH
        ]


def _judge(env: dict[str, str], gateway: _Gateway) -> JudgmentResponse:
    with open_judgment(env, transport=httpx.MockTransport(gateway)) as session:
        return session.port.judge("state", _QUESTIONS, _MODEL)


def _generate(env: dict[str, str], gateway: _Gateway) -> None:
    adapter = generation_adapter(env, transport=httpx.MockTransport(gateway))
    with adapter:
        adapter.generate(GenerationRequest(prompt="hi", schema=_SCHEMA, model=_MODEL))


def _generate_async(env: dict[str, str], gateway: _Gateway) -> None:
    adapter = async_vllm_generation_adapter(env, transport=httpx.MockTransport(gateway))
    request = GenerationRequest(prompt="hi", schema=_SCHEMA, model=_MODEL)

    async def run() -> None:
        try:
            await adapter.generate(request)
        finally:
            await adapter.close()

    asyncio.run(run())


Call = Callable[[dict[str, str], _Gateway], object]
_ENTRY_POINTS = pytest.mark.parametrize(
    "call", [_generate, _generate_async, _judge], ids=["sync", "async", "judge"]
)


def _raised(call: Call, env: dict[str, str], gateway: _Gateway) -> GenerationError:
    with pytest.raises(GenerationError) as info:
        call(env, gateway)
    return info.value


def _unavailable(request: httpx.Request) -> httpx.Response:
    return httpx.Response(503, json={"error": "busy"}, request=request)


def _redirect(request: httpx.Request) -> httpx.Response:
    headers = {"Location": "https://login.example.com/"}
    return httpx.Response(302, headers=headers, request=request)


def _refused(request: httpx.Request) -> httpx.Response:
    msg = "connection refused"
    raise httpx.ConnectError(msg, request=request)


def test_judgment_records_the_scoring_request_id_of_each_question() -> None:
    gateway = _Gateway()
    response = _judge(_env(), gateway)
    sent = gateway.scoring_ids()
    assert response.request_ids == dict(zip(_QUESTIONS, sent, strict=True))
    assert all(_HEX.fullmatch(str(value)) for value in sent)
    assert len(set(sent)) == len(_QUESTIONS)


def test_judgment_without_the_header_records_no_id() -> None:
    gateway = _Gateway()
    unset = _judge(_env(request_id=False), gateway)
    assert unset.request_ids == {}
    assert gateway.scoring_ids() == [None, None]
    traced = dataclasses.asdict(_judge(_env(), _Gateway()))
    plain = dataclasses.asdict(unset)
    assert set(traced) - set(plain) == set()
    assert plain.pop("request_ids") == {}
    traced.pop("request_ids")
    assert traced == plain


def _hooked_client(gateway: _Gateway, **headers: str) -> httpx.Client:
    return httpx.Client(
        base_url=_BASE,
        transport=httpx.MockTransport(gateway),
        headers=headers,
        event_hooks=sync_event_hooks(_HEADER),
    )


def test_open_vllm_judgment_keeps_a_caller_request_id() -> None:
    gateway = _Gateway()
    with (
        _hooked_client(gateway, **{_HEADER: "caller-id-356"}) as client,
        open_vllm_judgment(client=client, model=_MODEL) as session,
    ):
        response = session.port.judge("state", _QUESTIONS, _MODEL)
    assert response.request_ids == {"first": "caller-id-356", "second": "caller-id-356"}


class _Twice:
    """Scoring wrapper that scores each request twice, as a retry would."""

    def __init__(self, port: CandidateScoringPort) -> None:
        self._port = port

    def score_candidates(
        self, request: CandidateScoringRequest
    ) -> CandidateScoringResult:
        self._port.score_candidates(request)
        return self._port.score_candidates(request)


def test_open_vllm_judgment_records_the_last_request_of_a_question() -> None:
    gateway = _Gateway()
    with (
        _hooked_client(gateway) as client,
        open_vllm_judgment(
            client=client, model=_MODEL, scoring_port_wrapper=_Twice
        ) as session,
    ):
        response = session.port.judge("state", _QUESTIONS, _MODEL)
    sent = gateway.scoring_ids()
    assert len(sent) == 2 * len(_QUESTIONS)
    assert response.request_ids == {"first": sent[1], "second": sent[3]}


@_ENTRY_POINTS
@pytest.mark.parametrize("fail", [_unavailable, _redirect], ids=["503", "302"])
def test_backend_http_error_carries_the_failed_request_id(
    call: Call, fail: Handler
) -> None:
    gateway = _Gateway(fail)
    err = _raised(call, _env(), gateway)
    assert type(err) is BackendHttpError
    assert err.request_id == gateway.requests[-1].headers[_HEADER]
    assert _HEX.fullmatch(str(err.request_id))


@_ENTRY_POINTS
def test_transport_error_carries_the_failed_request_id(call: Call) -> None:
    gateway = _Gateway(_refused)
    err = _raised(call, _env(), gateway)
    assert type(err) is TransportError
    assert err.request_id == gateway.requests[-1].headers[_HEADER]
    assert _HEX.fullmatch(str(err.request_id))


@_ENTRY_POINTS
@pytest.mark.parametrize("fail", [_unavailable, _redirect, _refused])
def test_errors_without_the_header_carry_no_request_id(
    call: Call, fail: Handler
) -> None:
    gateway = _Gateway(fail)
    err = _raised(call, _env(request_id=False), gateway)
    assert isinstance(err, BackendHttpError | TransportError)
    assert err.request_id is None
    assert _HEADER not in gateway.requests[-1].headers


def test_transport_error_without_a_request_has_no_id() -> None:
    assert map_transport_error(httpx.ConnectError("refused")).request_id is None
