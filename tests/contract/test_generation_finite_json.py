"""Contract: structured generation rejects non-finite JSON numbers."""

from __future__ import annotations

import asyncio
import json
import math
from typing import Any

import httpx
import pytest

from typevet.adapters.outbound import (
    AsyncFakeGenerationAdapter,
    AsyncLlamaCppGenerationAdapter,
    FakeGenerationAdapter,
    LlamaCppGenerationAdapter,
)
from typevet.domain.errors import GenerationError, SchemaValidationError
from typevet.domain.models import GenerationRequest

NESTED_NUMBER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "score": {"type": "number"},
        "nested": {
            "type": "object",
            "properties": {"value": {"type": "number"}},
            "required": ["value"],
            "additionalProperties": False,
        },
        "items": {"type": "array", "items": {"type": "number"}},
        "label": {"type": "string"},
    },
    "required": ["score"],
    "additionalProperties": False,
}

REQUEST = GenerationRequest(
    prompt="score",
    model="m",
    schema=NESTED_NUMBER_SCHEMA,
)


def _llama_response(content: str) -> dict[str, Any]:
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


def _mock_client(content: str) -> httpx.Client:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_llama_response(content))

    return httpx.Client(transport=httpx.MockTransport(handler), base_url="http://test")


def _mock_async_client(content: str) -> httpx.AsyncClient:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_llama_response(content))

    return httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://test"
    )


@pytest.mark.contract
@pytest.mark.parametrize(
    ("content", "fake_value"),
    [
        ('{"score": NaN}', {"score": math.nan}),
        ('{"score": Infinity}', {"score": math.inf}),
        ('{"score": -Infinity}', {"score": -math.inf}),
        ('{"score": 1e999}', {"score": math.inf}),
        (
            '{"score": 1.0, "nested": {"value": NaN}}',
            {"score": 1.0, "nested": {"value": math.nan}},
        ),
        (
            '{"score": 1.0, "items": [1, NaN]}',
            {"score": 1.0, "items": [1, math.nan]},
        ),
    ],
    ids=["nan", "infinity", "neg_infinity", "overflow", "nested_nan", "array_nan"],
)
def test_sync_llama_rejects_non_finite(
    content: str, fake_value: dict[str, Any]
) -> None:
    client = _mock_client(content)
    adapter = LlamaCppGenerationAdapter(base_url="http://test", client=client)
    with pytest.raises(GenerationError):
        adapter.generate(REQUEST)


@pytest.mark.contract
@pytest.mark.parametrize(
    ("content", "fake_value"),
    [
        ('{"score": NaN}', {"score": math.nan}),
        ('{"score": Infinity}', {"score": math.inf}),
        ('{"score": -Infinity}', {"score": -math.inf}),
        ('{"score": 1e999}', {"score": math.inf}),
        (
            '{"score": 1.0, "nested": {"value": NaN}}',
            {"score": 1.0, "nested": {"value": math.nan}},
        ),
        (
            '{"score": 1.0, "items": [1, NaN]}',
            {"score": 1.0, "items": [1, math.nan]},
        ),
    ],
    ids=["nan", "infinity", "neg_infinity", "overflow", "nested_nan", "array_nan"],
)
def test_async_llama_rejects_non_finite(
    content: str, fake_value: dict[str, Any]
) -> None:
    async def run() -> None:
        async with _mock_async_client(content) as client:
            adapter = AsyncLlamaCppGenerationAdapter(
                base_url="http://test", client=client
            )
            with pytest.raises(GenerationError):
                await adapter.generate(REQUEST)

    asyncio.run(run())


@pytest.mark.contract
@pytest.mark.parametrize(
    "fake_value",
    [
        {"score": math.nan},
        {"score": math.inf},
        {"score": -math.inf},
        {"score": 1.0, "nested": {"value": math.nan}},
        {"score": 1.0, "items": [1, math.nan]},
    ],
    ids=["nan", "infinity", "neg_infinity", "nested_nan", "array_nan"],
)
def test_sync_fake_rejects_non_finite(fake_value: dict[str, Any]) -> None:
    adapter = FakeGenerationAdapter(value=fake_value)
    with pytest.raises(SchemaValidationError):
        adapter.generate(REQUEST)


@pytest.mark.contract
@pytest.mark.parametrize(
    "fake_value",
    [
        {"score": math.nan},
        {"score": math.inf},
        {"score": -math.inf},
        {"score": 1.0, "nested": {"value": math.nan}},
        {"score": 1.0, "items": [1, math.nan]},
    ],
    ids=["nan", "infinity", "neg_infinity", "nested_nan", "array_nan"],
)
def test_async_fake_rejects_non_finite(fake_value: dict[str, Any]) -> None:
    async def run() -> None:
        adapter = AsyncFakeGenerationAdapter(value=fake_value)
        with pytest.raises(SchemaValidationError):
            await adapter.generate(REQUEST)

    asyncio.run(run())


@pytest.mark.contract
def test_sync_llama_accepts_finite_and_nan_string_label() -> None:
    content = json.dumps({"score": 2.5, "label": "NaN is not a number token here"})
    client = _mock_client(content)
    adapter = LlamaCppGenerationAdapter(base_url="http://test", client=client)
    result = adapter.generate(REQUEST)
    assert result.value["score"] == 2.5
    assert "NaN" in result.value["label"]


@pytest.mark.contract
def test_sync_fake_accepts_finite_control() -> None:
    adapter = FakeGenerationAdapter(
        value={"score": 0.0, "nested": {"value": -3.14}, "items": [1, 2]}
    )
    result = adapter.generate(REQUEST)
    assert result.value["score"] == 0.0
    assert result.value["nested"]["value"] == pytest.approx(-3.14)


@pytest.mark.contract
def test_non_finite_raises_schema_validation_not_builtin() -> None:
    """Rejection uses GenerationError family, not json.JSONDecodeError."""
    client = _mock_client('{"score": NaN}')
    adapter = LlamaCppGenerationAdapter(base_url="http://test", client=client)
    with pytest.raises(SchemaValidationError) as exc_info:
        adapter.generate(REQUEST)
    assert exc_info.value.payload is not None
