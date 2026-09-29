"""Shared GenerationPort contract fixtures (judgevet shape).

Each fixture is a **synthetic** labeled case: the author defines the request,
the mocked HTTP stimulus (when used), and the fake configuration. A green
contract test shows the offline fake and ``LlamaCppGenerationAdapter`` (or
``VllmGenerationAdapter`` and its async counterpart) driven through
``httpx.MockTransport`` agree on value or error type for that case.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, Literal

import httpx

from typevet.adapters.outbound import (
    AsyncFakeGenerationAdapter,
    AsyncLlamaCppGenerationAdapter,
    FakeGenerationAdapter,
    LlamaCppGenerationAdapter,
)
from typevet.adapters.outbound.vllm_generation import VllmGenerationAdapter
from typevet.adapters.outbound.vllm_generation_async import AsyncVllmGenerationAdapter
from typevet.domain.errors import (
    BackendHttpError,
    GenerationError,
    SchemaValidationError,
)
from typevet.domain.models import GenerationRequest

CONTRACT_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "properties": {
        "answer": {"type": "integer"},
    },
    "required": ["answer"],
    "additionalProperties": False,
}

ExpectKind = Literal["success", "error"]


def _success_fixture(
    *,
    name: str,
    prompt: str,
    model: str,
    answer: int,
) -> dict[str, Any]:
    content = json.dumps({"answer": answer})
    return {
        "name": name,
        "label": "synthetic",
        "request": {"prompt": prompt, "model": model, "schema": CONTRACT_SCHEMA},
        "http": {"status": 200, "content": content},
        "fake": {"value": {"answer": answer}},
        "expect": {
            "kind": "success",
            "value": {"answer": answer},
            "raw_text": content,
        },
    }


def _schema_error_fixture() -> dict[str, Any]:
    content = '{"answer": "x"}'
    return {
        "name": "schema_validation_fail",
        "label": "synthetic",
        "request": {
            "prompt": "bad",
            "model": "m",
            "schema": CONTRACT_SCHEMA,
        },
        "http": {"status": 200, "content": content},
        "fake": {"value": {"answer": "x"}},
        "expect": {"kind": "error", "exc_type": "SchemaValidationError"},
    }


def _http_500_fixture() -> dict[str, Any]:
    body = "boom"
    status = 500
    return {
        "name": "backend_http_500",
        "label": "synthetic",
        "request": {
            "prompt": "x",
            "model": "m",
            "schema": CONTRACT_SCHEMA,
        },
        "http": {"status": status, "text": body},
        "fake": {
            "fail_backend_http": {
                "status_code": status,
                "body_snippet": body,
            }
        },
        "expect": {
            "kind": "error",
            "exc_type": "BackendHttpError",
            "status_code": status,
            "body_snippet": body,
        },
    }


def get_fixtures() -> list[dict[str, Any]]:
    """Return shared contract fixtures for sync and async suites."""
    return [
        _success_fixture(
            name="valid_answer", prompt="seven", model="gemma-test", answer=7
        ),
        _success_fixture(name="valid_answer_alt", prompt="n?", model="fake", answer=42),
        _schema_error_fixture(),
        _http_500_fixture(),
    ]


def get_fixture_by_name(name: str) -> dict[str, Any]:
    """Return one fixture by ``name``."""
    for fixture in get_fixtures():
        if fixture["name"] == name:
            return fixture
    msg = f"fixture not found: {name}"
    raise ValueError(msg)


def generation_request(fixture: dict[str, Any]) -> GenerationRequest:
    """Build a ``GenerationRequest`` from a fixture."""
    req = fixture["request"]
    return GenerationRequest(
        prompt=req["prompt"],
        schema=req["schema"],
        model=req["model"],
    )


def _backend_http_from_fake(data: Mapping[str, Any]) -> BackendHttpError:
    status_code = int(data["status_code"])
    snippet = str(data["body_snippet"])
    return BackendHttpError(
        f"llama.cpp HTTP {status_code}: {snippet}",
        status_code=status_code,
        body_snippet=snippet,
    )


def sync_fake_adapter(fixture: dict[str, Any]) -> FakeGenerationAdapter:
    """Build the offline fake for a fixture."""
    fake = fixture["fake"]
    if "value" in fake:
        return FakeGenerationAdapter(value=fake["value"])
    if "fail_backend_http" in fake:
        return FakeGenerationAdapter(
            fail=_backend_http_from_fake(fake["fail_backend_http"])
        )
    msg = "fixture fake config unsupported"
    raise ValueError(msg)


def async_fake_adapter(fixture: dict[str, Any]) -> AsyncFakeGenerationAdapter:
    """Build the async offline fake for a fixture."""
    fake = fixture["fake"]
    if "value" in fake:
        return AsyncFakeGenerationAdapter(value=fake["value"])
    if "fail_backend_http" in fake:
        return AsyncFakeGenerationAdapter(
            fail=_backend_http_from_fake(fake["fail_backend_http"])
        )
    msg = "fixture fake config unsupported"
    raise ValueError(msg)


def _replay(fixture: dict[str, Any]) -> httpx.Response:
    http = fixture["http"]
    if "content" in http:
        return httpx.Response(
            http.get("status", 200),
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": http["content"],
                        }
                    }
                ]
            },
        )
    return httpx.Response(http["status"], text=http.get("text", ""))


def mock_transport_for(fixture: dict[str, Any]) -> httpx.MockTransport:
    """Build ``httpx.MockTransport`` that replays the fixture HTTP stimulus."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/v1/chat/completions")
        body = json.loads(request.content.decode())
        assert body["response_format"]["type"] == "json_schema"
        assert body["chat_template_kwargs"] == {"enable_thinking": False}
        assert body["chat_template_kwargs"]["enable_thinking"] is False
        return _replay(fixture)

    return httpx.MockTransport(handler)


def vllm_mock_transport_for(fixture: dict[str, Any]) -> httpx.MockTransport:
    """Build a vLLM ``httpx.MockTransport`` that replays the fixture stimulus.

    The handler asserts the vLLM structured-output body shape before replay.

    Returns:
        Transport that checks the body and replays the fixture response.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/v1/chat/completions")
        body = json.loads(request.content.decode())
        assert body["structured_outputs"] == {"json": fixture["request"]["schema"]}
        assert "response_format" not in body
        return _replay(fixture)

    return httpx.MockTransport(handler)


def sync_llama_adapter(fixture: dict[str, Any]) -> LlamaCppGenerationAdapter:
    """Build ``LlamaCppGenerationAdapter`` with injected mock transport."""
    client = httpx.Client(
        transport=mock_transport_for(fixture),
        base_url="http://test",
    )
    return LlamaCppGenerationAdapter(base_url="http://test", client=client)


def sync_vllm_adapter(fixture: dict[str, Any]) -> VllmGenerationAdapter:
    """Build ``VllmGenerationAdapter`` with injected mock transport.

    Returns:
        Adapter whose client replays the fixture through the vLLM transport.
    """
    client = httpx.Client(
        transport=vllm_mock_transport_for(fixture),
        base_url="http://test",
    )
    return VllmGenerationAdapter(base_url="http://test", client=client)


def async_llama_adapter(fixture: dict[str, Any]) -> AsyncLlamaCppGenerationAdapter:
    """Build ``AsyncLlamaCppGenerationAdapter`` with injected mock transport."""
    client = httpx.AsyncClient(
        transport=mock_transport_for(fixture),
        base_url="http://test",
    )
    return AsyncLlamaCppGenerationAdapter(base_url="http://test", client=client)


def async_vllm_adapter(fixture: dict[str, Any]) -> AsyncVllmGenerationAdapter:
    """Build ``AsyncVllmGenerationAdapter`` with injected mock transport.

    Returns:
        Adapter whose async client replays the fixture through the vLLM transport.
    """
    client = httpx.AsyncClient(
        transport=vllm_mock_transport_for(fixture),
        base_url="http://test",
    )
    return AsyncVllmGenerationAdapter(base_url="http://test", client=client)


def exc_type_from_name(name: str) -> type[GenerationError]:
    """Resolve a fixture error name to a domain exception type."""
    mapping: dict[str, type[GenerationError]] = {
        "SchemaValidationError": SchemaValidationError,
        "BackendHttpError": BackendHttpError,
    }
    try:
        return mapping[name]
    except KeyError as exc:
        msg = f"unknown exc_type: {name}"
        raise ValueError(msg) from exc
