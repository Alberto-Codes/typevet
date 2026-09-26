"""Extra unit coverage for domain validation and fake edge paths."""

from __future__ import annotations

from typing import Any, cast

import httpx
import pytest

from typevet.adapters.outbound import FakeGenerationAdapter, LlamaCppGenerationAdapter
from typevet.domain.errors import GenerationError
from typevet.domain.models import GenerationRequest

SCHEMA = {
    "type": "object",
    "properties": {"n": {"type": "integer"}},
    "required": ["n"],
    "additionalProperties": False,
}


@pytest.mark.unit
def test_request_rejects_empty_model() -> None:
    with pytest.raises(ValueError, match="model"):
        GenerationRequest(prompt="hi", schema=SCHEMA, model=" ")


@pytest.mark.unit
def test_request_rejects_non_mapping_schema() -> None:
    with pytest.raises(TypeError, match="mapping"):
        GenerationRequest(
            prompt="hi",
            schema=cast(Any, ["not", "map"]),
            model="m",
        )


@pytest.mark.unit
def test_fake_requires_a_source() -> None:
    with pytest.raises(ValueError, match="provide"):
        FakeGenerationAdapter()


@pytest.mark.unit
def test_fake_responder_and_fail() -> None:
    with pytest.raises(GenerationError, match="boom"):
        FakeGenerationAdapter(fail=GenerationError("boom")).generate(
            GenerationRequest(prompt="x", schema=SCHEMA, model="m")
        )
    adapter = FakeGenerationAdapter(responder=lambda _r: {"n": 3})
    assert adapter.generate(
        GenerationRequest(prompt="x", schema=SCHEMA, model="m")
    ).value == {"n": 3}


@pytest.mark.unit
def test_llama_context_closes_owned_client() -> None:
    with LlamaCppGenerationAdapter(base_url="http://test") as adapter:
        assert adapter is not None
    # closing twice is safe
    adapter.close()


@pytest.mark.contract
def test_llama_non_json_body() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not-json")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    adapter = LlamaCppGenerationAdapter(base_url="http://test", client=client)
    with pytest.raises(GenerationError, match="non-JSON"):
        adapter.generate(GenerationRequest(prompt="x", schema=SCHEMA, model="m"))


@pytest.mark.contract
def test_llama_empty_content_and_non_object_json() -> None:
    def empty(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "  "}}]},
        )

    client = httpx.Client(transport=httpx.MockTransport(empty))
    adapter = LlamaCppGenerationAdapter(base_url="http://test", client=client)
    with pytest.raises(GenerationError, match="empty"):
        adapter.generate(GenerationRequest(prompt="x", schema=SCHEMA, model="m"))

    def array_root(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"choices": [{"message": {"role": "assistant", "content": "[1]"}}]},
        )

    client2 = httpx.Client(transport=httpx.MockTransport(array_root))
    adapter2 = LlamaCppGenerationAdapter(base_url="http://test", client=client2)
    with pytest.raises(Exception, match="object"):
        adapter2.generate(GenerationRequest(prompt="x", schema=SCHEMA, model="m"))


@pytest.mark.contract
def test_llama_transport_error() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    adapter = LlamaCppGenerationAdapter(base_url="http://test", client=client)
    with pytest.raises(GenerationError, match="request failed"):
        adapter.generate(GenerationRequest(prompt="x", schema=SCHEMA, model="m"))


@pytest.mark.contract
def test_llama_invalid_json_content() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": "{not-json"}}]
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    adapter = LlamaCppGenerationAdapter(base_url="http://test", client=client)
    with pytest.raises(GenerationError, match="not valid JSON"):
        adapter.generate(GenerationRequest(prompt="x", schema=SCHEMA, model="m"))


@pytest.mark.contract
def test_llama_missing_choices() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": []})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    adapter = LlamaCppGenerationAdapter(base_url="http://test", client=client)
    with pytest.raises(GenerationError, match="missing choices"):
        adapter.generate(GenerationRequest(prompt="x", schema=SCHEMA, model="m"))
