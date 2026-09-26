"""Unit coverage for async outbound adapters."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from typevet.adapters.outbound import (
    AsyncFakeGenerationAdapter,
    AsyncLlamaCppGenerationAdapter,
)
from typevet.domain.errors import GenerationError, TransportError
from typevet.domain.models import GenerationRequest

SCHEMA = {
    "type": "object",
    "properties": {"n": {"type": "integer"}},
    "required": ["n"],
    "additionalProperties": False,
}


@pytest.mark.unit
def test_async_fake_requires_a_source() -> None:
    with pytest.raises(ValueError, match="provide"):
        AsyncFakeGenerationAdapter()


@pytest.mark.unit
def test_async_fake_responder_and_fail() -> None:
    with pytest.raises(GenerationError, match="boom"):
        asyncio.run(
            AsyncFakeGenerationAdapter(fail=GenerationError("boom")).generate(
                GenerationRequest(prompt="x", schema=SCHEMA, model="m")
            )
        )
    adapter = AsyncFakeGenerationAdapter(responder=lambda _r: {"n": 3})
    assert asyncio.run(
        adapter.generate(GenerationRequest(prompt="x", schema=SCHEMA, model="m"))
    ).value == {"n": 3}


@pytest.mark.unit
def test_async_llama_context_closes_owned_client() -> None:
    async def run() -> None:
        async with AsyncLlamaCppGenerationAdapter(base_url="http://test") as adapter:
            assert adapter is not None
        await adapter.close()

    asyncio.run(run())


@pytest.mark.contract
def test_async_llama_transport_error() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    adapter = AsyncLlamaCppGenerationAdapter(base_url="http://test", client=client)
    with pytest.raises(TransportError, match="request failed"):
        asyncio.run(
            adapter.generate(GenerationRequest(prompt="x", schema=SCHEMA, model="m"))
        )


@pytest.mark.contract
def test_async_llama_non_json_body() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not-json")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    adapter = AsyncLlamaCppGenerationAdapter(base_url="http://test", client=client)
    with pytest.raises(GenerationError, match="non-JSON"):
        asyncio.run(
            adapter.generate(GenerationRequest(prompt="x", schema=SCHEMA, model="m"))
        )


@pytest.mark.contract
def test_async_llama_empty_content_and_non_object_json() -> None:
    def empty(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "  "}}]},
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(empty))
    adapter = AsyncLlamaCppGenerationAdapter(base_url="http://test", client=client)
    with pytest.raises(GenerationError, match="empty"):
        asyncio.run(
            adapter.generate(GenerationRequest(prompt="x", schema=SCHEMA, model="m"))
        )

    def array_root(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "[1]"}}]},
        )

    client2 = httpx.AsyncClient(transport=httpx.MockTransport(array_root))
    adapter2 = AsyncLlamaCppGenerationAdapter(base_url="http://test", client=client2)
    with pytest.raises(Exception, match="object"):
        asyncio.run(
            adapter2.generate(GenerationRequest(prompt="x", schema=SCHEMA, model="m"))
        )


@pytest.mark.contract
def test_async_llama_invalid_json_content() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": "{not-json"}}]
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    adapter = AsyncLlamaCppGenerationAdapter(base_url="http://test", client=client)
    with pytest.raises(GenerationError, match="not valid JSON"):
        asyncio.run(
            adapter.generate(GenerationRequest(prompt="x", schema=SCHEMA, model="m"))
        )


@pytest.mark.contract
def test_async_llama_missing_choices() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": []})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    adapter = AsyncLlamaCppGenerationAdapter(base_url="http://test", client=client)
    with pytest.raises(GenerationError, match="missing choices"):
        asyncio.run(
            adapter.generate(GenerationRequest(prompt="x", schema=SCHEMA, model="m"))
        )


@pytest.mark.unit
def test_async_llama_owned_client_lifecycle() -> None:
    async def run() -> None:
        adapter = AsyncLlamaCppGenerationAdapter(
            base_url="http://127.0.0.1:1",
            timeout=0.01,
        )
        with pytest.raises(TransportError, match="request failed"):
            await adapter.generate(
                GenerationRequest(prompt="x", schema=SCHEMA, model="m")
            )
        await adapter.close()
        await adapter.close()

    asyncio.run(run())
