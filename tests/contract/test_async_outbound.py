"""Contract tests for async outbound fakes and HTTP adapter parsing."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from tests.fixtures.generation_contract import CONTRACT_SCHEMA as SCHEMA
from typevet.adapters.outbound import (
    AsyncFakeGenerationAdapter,
    AsyncLlamaCppGenerationAdapter,
)
from typevet.domain.errors import (
    BackendHttpError,
    SchemaValidationError,
)
from typevet.domain.models import GenerationRequest


@pytest.mark.contract
def test_async_fake_validates_against_schema() -> None:
    adapter = AsyncFakeGenerationAdapter(value={"answer": 42})
    result = asyncio.run(
        adapter.generate(GenerationRequest(prompt="n?", schema=SCHEMA, model="fake"))
    )
    assert result.value == {"answer": 42}


@pytest.mark.contract
def test_async_fake_rejects_invalid_value() -> None:
    adapter = AsyncFakeGenerationAdapter(value={"answer": "nope"})
    with pytest.raises(SchemaValidationError):
        asyncio.run(
            adapter.generate(
                GenerationRequest(prompt="n?", schema=SCHEMA, model="fake")
            )
        )


@pytest.mark.contract
def test_async_llama_cpp_adapter_parses_and_validates() -> None:
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
    client = httpx.AsyncClient(transport=transport, base_url="http://test")
    adapter = AsyncLlamaCppGenerationAdapter(
        base_url="http://test",
        client=client,
    )

    async def run() -> None:
        result = await adapter.generate(
            GenerationRequest(prompt="seven", schema=SCHEMA, model="gemma-test")
        )
        assert result.value == {"answer": 7}
        assert result.raw_text == '{"answer": 7}'
        await adapter.close()

    asyncio.run(run())


@pytest.mark.contract
def test_async_llama_cpp_adapter_fail_fast_on_schema() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"role": "assistant", "content": '{"answer": "x"}'}}
                ]
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    adapter = AsyncLlamaCppGenerationAdapter(base_url="http://test", client=client)
    with pytest.raises(SchemaValidationError):
        asyncio.run(
            adapter.generate(GenerationRequest(prompt="bad", schema=SCHEMA, model="m"))
        )


@pytest.mark.contract
def test_async_llama_cpp_adapter_http_error() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    adapter = AsyncLlamaCppGenerationAdapter(base_url="http://test", client=client)
    with pytest.raises(BackendHttpError, match="HTTP 500") as exc_info:
        asyncio.run(
            adapter.generate(GenerationRequest(prompt="x", schema=SCHEMA, model="m"))
        )
    assert exc_info.value.status_code == 500
    assert exc_info.value.body_snippet == "boom"


_BAD_SCHEMA = {"type": "object", "properties": {"answer": {"type": "not-a-type"}}}


def _reply_answer(_request: httpx.Request) -> httpx.Response:
    content = '{"answer": 7}'
    return httpx.Response(
        200, json={"choices": [{"message": {"role": "assistant", "content": content}}]}
    )


@pytest.mark.contract
def test_async_llama_cpp_adapter_rejects_malformed_schema_before_any_request() -> None:
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return _reply_answer(request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    adapter = AsyncLlamaCppGenerationAdapter(base_url="http://test", client=client)
    request = GenerationRequest(prompt="x", schema=_BAD_SCHEMA, model="m")
    with pytest.raises(ValueError, match=r"^schema is not a valid JSON Schema: "):
        asyncio.run(adapter.generate(request))
    assert sent == []
