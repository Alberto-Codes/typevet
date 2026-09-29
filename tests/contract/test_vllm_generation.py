"""Contract tests for the vLLM structured generation adapter (#169A).

The HTTP 400 fixture ``http400_invalid_schema`` is the redacted vLLM v0.30.0
P10 response from the #168 probe. Shared ``generation_contract`` fixtures
prove parity with ``FakeGenerationAdapter`` and, for the async adapter
(#169E), with ``AsyncFakeGenerationAdapter``.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from tests.contract._generation_port_assertions import (
    run_error_contract,
    run_success_contract,
)
from tests.fixtures.generation_contract import (
    CONTRACT_SCHEMA,
    async_fake_adapter,
    async_vllm_adapter,
    generation_request,
    get_fixtures,
    sync_fake_adapter,
    sync_vllm_adapter,
)
from typevet.adapters import outbound
from typevet.adapters.outbound.vllm_generation import VllmGenerationAdapter
from typevet.domain.errors import (
    BackendHttpError,
    GenerationError,
    SchemaValidationError,
    TransportError,
)
from typevet.domain.models import GenerationRequest
from typevet.ports.generation import GenerationPort

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "vllm"
_MODEL = "served-model"


def _probe(name: str) -> dict[str, Any]:
    return json.loads((_FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


class _Recorder:
    """MockTransport handler that records requests and replies once per call."""

    def __init__(self, *, status: int = 200, payload: object = None, text: str = ""):
        self.bodies: list[dict[str, Any]] = []
        self.urls: list[str] = []
        self._status = status
        self._payload = payload
        self._text = text

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.urls.append(str(request.url))
        self.bodies.append(json.loads(request.content.decode()))
        if self._payload is None:
            return httpx.Response(self._status, text=self._text)
        return httpx.Response(self._status, json=self._payload)


def _content_reply(content: object) -> dict[str, Any]:
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


def _adapter(recorder: _Recorder) -> VllmGenerationAdapter:
    client = httpx.Client(transport=httpx.MockTransport(recorder))
    return VllmGenerationAdapter("http://vllm.test:8000/", client=client)


def _request(schema: Any = CONTRACT_SCHEMA) -> GenerationRequest:
    return GenerationRequest(prompt="seven", schema=schema, model=_MODEL)


@pytest.mark.contract
def test_vllm_generation_is_exported_and_satisfies_port() -> None:
    assert outbound.VllmGenerationAdapter is VllmGenerationAdapter
    assert "VllmGenerationAdapter" in outbound.__all__
    port: GenerationPort = VllmGenerationAdapter("http://x")
    assert callable(port.generate)


@pytest.mark.contract
def test_vllm_generation_sends_exact_body_keys() -> None:
    recorder = _Recorder(payload=_content_reply('{"answer": 7}'))

    result = _adapter(recorder).generate(_request())

    assert result.value == {"answer": 7}
    assert recorder.urls == ["http://vllm.test:8000/v1/chat/completions"]
    assert recorder.bodies == [
        {
            "model": _MODEL,
            "messages": [{"role": "user", "content": "seven"}],
            "temperature": 0,
            "structured_outputs": {"json": dict(CONTRACT_SCHEMA)},
            "add_generation_prompt": True,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    ]
    body = recorder.bodies[0]
    assert "response_format" not in body
    assert "continue_final_message" not in body
    assert not [key for key in body if key.startswith("guided_")]


@pytest.mark.contract
@pytest.mark.parametrize("fixture", get_fixtures(), ids=lambda row: row["name"])
def test_vllm_generation_agrees_with_fake_on_shared_fixtures(
    fixture: dict[str, Any],
) -> None:
    request = generation_request(fixture)
    fake = sync_fake_adapter(fixture)
    real = sync_vllm_adapter(fixture)
    runner = run_success_contract
    if fixture["expect"]["kind"] == "error":
        runner = run_error_contract
    runner(
        fixture=fixture,
        fake_generate=lambda: fake.generate(request),
        real_generate=lambda: real.generate(request),
    )


@pytest.mark.contract
@pytest.mark.parametrize("fixture", get_fixtures(), ids=lambda row: row["name"])
def test_async_vllm_generation_agrees_with_async_fake_on_shared_fixtures(
    fixture: dict[str, Any],
) -> None:
    request = generation_request(fixture)
    fake = async_fake_adapter(fixture)
    real = async_vllm_adapter(fixture)
    runner = run_success_contract
    if fixture["expect"]["kind"] == "error":
        runner = run_error_contract
    runner(
        fixture=fixture,
        fake_generate=lambda: asyncio.run(fake.generate(request)),
        real_generate=lambda: asyncio.run(real.generate(request)),
    )


@pytest.mark.contract
def test_vllm_generation_p10_invalid_schema_is_rejected_before_the_request() -> None:
    probe = _probe("http400_invalid_schema")
    recorder = _Recorder(status=probe["http_status"], payload=probe["response"])
    schema = probe["request"]["structured_outputs"]["json"]

    with pytest.raises(ValueError, match=r"^schema is not a valid JSON Schema: "):
        _adapter(recorder).generate(_request(schema))

    assert recorder.bodies == []


@pytest.mark.contract
def test_vllm_generation_p10_http_400_reply_raises_backend_http_error() -> None:
    probe = _probe("http400_invalid_schema")
    recorder = _Recorder(status=probe["http_status"], payload=probe["response"])

    with pytest.raises(BackendHttpError) as caught:
        _adapter(recorder).generate(_request())

    exc = caught.value
    assert exc.status_code == 400
    assert probe["response"]["error"]["message"] in exc.body_snippet
    assert str(exc).startswith("vLLM HTTP 400: ")
    assert len(recorder.bodies) == 1


@pytest.mark.contract
def test_vllm_generation_transport_failure_raises_transport_error() -> None:
    posts: list[httpx.Request] = []

    def fail(request: httpx.Request) -> httpx.Response:
        posts.append(request)
        raise httpx.ConnectError("refused", request=request)

    client = httpx.Client(transport=httpx.MockTransport(fail))
    adapter = VllmGenerationAdapter("http://vllm.test", client=client)

    with pytest.raises(TransportError, match="vLLM request failed"):
        adapter.generate(_request())
    assert len(posts) == 1


@pytest.mark.contract
def test_vllm_generation_http_500_raises_backend_http_error_after_one_post() -> None:
    recorder = _Recorder(status=500, text="engine crashed")

    with pytest.raises(BackendHttpError) as caught:
        _adapter(recorder).generate(_request())

    assert caught.value.status_code == 500
    assert str(caught.value) == "vLLM HTTP 500: engine crashed"
    assert len(recorder.bodies) == 1


@pytest.mark.contract
@pytest.mark.parametrize(
    ("reply", "message"),
    [
        ({"choices": []}, "missing choices"),
        ({"object": "chat.completion"}, "missing choices"),
        (_content_reply(None), "empty message content"),
        (_content_reply("   "), "empty message content"),
        (_content_reply("not json"), "not valid JSON"),
    ],
    ids=["no-choice", "no-choices-key", "null-content", "blank", "non-json"],
)
def test_vllm_generation_bad_shape_raises_generation_error(
    reply: dict[str, Any], message: str
) -> None:
    recorder = _Recorder(payload=reply)

    with pytest.raises(GenerationError, match=message) as caught:
        _adapter(recorder).generate(_request())

    assert type(caught.value) is GenerationError
    assert len(recorder.bodies) == 1


@pytest.mark.contract
def test_vllm_generation_non_json_http_body_raises_generation_error() -> None:
    recorder = _Recorder(text="<html>ok</html>")

    with pytest.raises(GenerationError, match="vLLM returned non-JSON HTTP body"):
        _adapter(recorder).generate(_request())


@pytest.mark.contract
@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("[7]", "root must be an object"),
        ('{"answer": NaN}', "non-finite"),
        ('{"answer": Infinity}', "non-finite"),
        ('{"answer": 7, "extra": 1}', "output failed schema"),
    ],
    ids=["array-root", "nan", "infinity", "schema-miss"],
)
def test_vllm_generation_invalid_value_raises_schema_validation_error(
    content: str, message: str
) -> None:
    recorder = _Recorder(payload=_content_reply(content))

    with pytest.raises(SchemaValidationError, match=message):
        _adapter(recorder).generate(_request())

    assert len(recorder.bodies) == 1


@pytest.mark.contract
def test_vllm_generation_owned_client_closes_on_exit() -> None:
    with VllmGenerationAdapter("http://vllm.test") as adapter:
        client = adapter._ensure_client()
        assert adapter._ensure_client() is client
    assert client.is_closed
    assert adapter._client is None


@pytest.mark.contract
def test_vllm_generation_injected_client_stays_open() -> None:
    client = httpx.Client()
    with VllmGenerationAdapter("http://vllm.test", client=client):
        pass
    assert not client.is_closed
    client.close()


@pytest.mark.contract
def test_vllm_generation_rejects_malformed_schema_before_any_request() -> None:
    recorder = _Recorder(payload=_content_reply('{"answer": 7}'))
    schema = {"type": "object", "properties": {"answer": {"type": "not-a-type"}}}

    with pytest.raises(ValueError, match=r"^schema is not a valid JSON Schema: "):
        _adapter(recorder).generate(_request(schema))

    assert recorder.bodies == []
