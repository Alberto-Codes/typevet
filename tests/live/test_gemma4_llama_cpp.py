"""Live path against local llama.cpp + Gemma 4 (opt-in)."""

from __future__ import annotations

import os

import httpx
import pytest

from typevet.adapters.outbound import LlamaCppGenerationAdapter
from typevet.domain.models import GenerationRequest

BASE = os.environ.get("TYPEVET_LLAMA_URL", "http://127.0.0.1:8090")
MODEL = os.environ.get("TYPEVET_GEMMA_MODEL", "gemma-4-31b-24gib-kv11-decoder")

SCHEMA = {
    "type": "object",
    "properties": {
        "sentiment": {"type": "string", "enum": ["pos", "neg", "neu"]},
        "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
    },
    "required": ["sentiment", "confidence"],
    "additionalProperties": False,
}


def _router_up() -> bool:
    try:
        response = httpx.get(f"{BASE.rstrip('/')}/v1/models", timeout=5.0)
    except httpx.HTTPError:
        return False
    return response.status_code == 200


def _model_listed() -> bool:
    try:
        response = httpx.get(f"{BASE.rstrip('/')}/v1/models", timeout=5.0)
        data = response.json()
    except (httpx.HTTPError, ValueError):
        return False
    ids = {item.get("id") for item in data.get("data", [])}
    return MODEL in ids


@pytest.mark.live
@pytest.mark.skipif(not _router_up(), reason="llama.cpp router not reachable")
@pytest.mark.skipif(not _model_listed(), reason=f"{MODEL} not in router catalog")
def test_gemma4_schema_in_valid_out() -> None:
    with LlamaCppGenerationAdapter(base_url=BASE, timeout=600.0) as adapter:
        result = adapter.generate(
            GenerationRequest(
                prompt=(
                    "Classify the sentiment of: 'I love this library.' "
                    "Return JSON only matching the schema."
                ),
                schema=SCHEMA,
                model=MODEL,
            )
        )
    assert result.value["sentiment"] in {"pos", "neg", "neu"}
    assert 0 <= int(result.value["confidence"]) <= 100
