"""Contract tests for outbound fakes and HTTP adapter parsing."""

from __future__ import annotations

import json

import httpx
import pytest

from typevet.adapters.outbound import FakeGenerationAdapter, LlamaCppGenerationAdapter
from typevet.domain.errors import GenerationError, SchemaValidationError
from typevet.domain.models import GenerationRequest

SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "integer"},
    },
    "required": ["answer"],
    "additionalProperties": False,
}


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
    with pytest.raises(GenerationError, match="HTTP 500"):
        adapter.generate(GenerationRequest(prompt="x", schema=SCHEMA, model="m"))
