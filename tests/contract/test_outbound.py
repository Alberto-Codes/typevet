"""Contract tests for outbound fakes and HTTP adapter parsing."""

from __future__ import annotations

import json

import httpx
import pytest

from tests.fixtures.generation_contract import CONTRACT_SCHEMA as SCHEMA
from typevet.adapters.outbound import FakeGenerationAdapter, LlamaCppGenerationAdapter
from typevet.domain.errors import (
    BackendHttpError,
    SchemaValidationError,
)
from typevet.domain.models import GenerationRequest


@pytest.mark.contract
def test_fake_validates_against_schema() -> None:
    adapter = FakeGenerationAdapter(value={"answer": 42})
    result = adapter.generate(
        GenerationRequest(prompt="n?", schema=SCHEMA, model="fake")
    )
    assert result.value == {"answer": 42}


@pytest.mark.contract
def test_fake_rejects_invalid_value() -> None:
    adapter = FakeGenerationAdapter(value={"answer": "nope"})
    with pytest.raises(SchemaValidationError):
        adapter.generate(GenerationRequest(prompt="n?", schema=SCHEMA, model="fake"))


@pytest.mark.contract
def test_llama_cpp_adapter_parses_and_validates() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/v1/chat/completions")
        body = json.loads(request.content.decode())
        assert body["response_format"]["type"] == "json_schema"
        assert body["chat_template_kwargs"] == {"enable_thinking": False}
        assert body["chat_template_kwargs"]["enable_thinking"] is False
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": '{"answer": 7}',
                        }
                    }
                ]
            },
        )

    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport, base_url="http://test")
    adapter = LlamaCppGenerationAdapter(
        base_url="http://test",
        client=client,
    )
    result = adapter.generate(
        GenerationRequest(prompt="seven", schema=SCHEMA, model="gemma-test")
    )
    assert result.value == {"answer": 7}
    assert result.raw_text == '{"answer": 7}'


@pytest.mark.contract
def test_llama_cpp_adapter_fail_fast_on_schema() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"role": "assistant", "content": '{"answer": "x"}'}}
                ]
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    adapter = LlamaCppGenerationAdapter(base_url="http://test", client=client)
    with pytest.raises(SchemaValidationError):
        adapter.generate(GenerationRequest(prompt="bad", schema=SCHEMA, model="m"))


@pytest.mark.contract
def test_llama_cpp_adapter_http_error() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    adapter = LlamaCppGenerationAdapter(base_url="http://test", client=client)
    with pytest.raises(BackendHttpError, match="HTTP 500") as exc_info:
        adapter.generate(GenerationRequest(prompt="x", schema=SCHEMA, model="m"))
    assert exc_info.value.status_code == 500
    assert exc_info.value.body_snippet == "boom"


@pytest.mark.contract
def test_llama_cpp_429_keeps_default_retry_fields() -> None:
    """llama.cpp errors read no ``Retry-After`` or rate-limit header (#355)."""

    def handler(_request: httpx.Request) -> httpx.Response:
        headers = {"Retry-After": "7", "X-RateLimit-Remaining": "0"}
        return httpx.Response(429, text="busy", headers=headers)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    adapter = LlamaCppGenerationAdapter(base_url="http://test", client=client)
    with pytest.raises(BackendHttpError, match="HTTP 429") as exc_info:
        adapter.generate(GenerationRequest(prompt="x", schema=SCHEMA, model="m"))
    assert exc_info.value.retry_after_seconds is None
    assert dict(exc_info.value.rate_limit) == {}


_BAD_SCHEMA = {"type": "object", "properties": {"answer": {"type": "not-a-type"}}}


def _reply_answer(_request: httpx.Request) -> httpx.Response:
    content = '{"answer": 7}'
    return httpx.Response(
        200, json={"choices": [{"message": {"role": "assistant", "content": content}}]}
    )


@pytest.mark.contract
def test_llama_cpp_adapter_rejects_malformed_schema_before_any_request() -> None:
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return _reply_answer(request)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    adapter = LlamaCppGenerationAdapter(base_url="http://test", client=client)
    request = GenerationRequest(prompt="x", schema=_BAD_SCHEMA, model="m")
    with pytest.raises(ValueError, match=r"^schema is not a valid JSON Schema: "):
        adapter.generate(request)
    assert sent == []
