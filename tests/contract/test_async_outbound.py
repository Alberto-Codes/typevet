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
from typevet.domain.errors import GenerationError, SchemaValidationError
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
    with pytest.raises(GenerationError, match="HTTP 500"):
        asyncio.run(
            adapter.generate(GenerationRequest(prompt="x", schema=SCHEMA, model="m"))
        )
