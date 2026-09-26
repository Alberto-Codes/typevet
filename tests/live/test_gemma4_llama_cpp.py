"""Live path against local llama.cpp + Gemma 4 (opt-in)."""

from __future__ import annotations

from dataclasses import replace

import httpx
import pytest

from typevet.adapters.inbound.settings import llama_cpp_adapter, load_llama_settings
from typevet.domain.models import GenerationRequest

_LLAMA = load_llama_settings()
BASE = _LLAMA.base_url
MODEL = _LLAMA.default_model

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


@pytest.fixture
def gemma4_llama_router() -> str:
    """Probe local llama.cpp only when a live test is selected to run."""
    model = MODEL
    if not model:
        pytest.skip("TYPEVET_LLAMA__DEFAULT_MODEL (or TYPEVET_GEMMA_MODEL) not set")
    if not _router_up():
        pytest.skip("llama.cpp router not reachable")
    if not _model_listed():
        pytest.skip(f"{model} not in router catalog")
    return model


@pytest.mark.live
def test_gemma4_schema_in_valid_out(gemma4_llama_router: str) -> None:
    live_settings = replace(_LLAMA, timeout=600.0)
    with llama_cpp_adapter(live_settings) as adapter:
        result = adapter.generate(
            GenerationRequest(
                prompt=(
                    "Classify the sentiment of: 'I love this library.' "
                    "Return JSON only matching the schema."
                ),
                schema=SCHEMA,
                model=gemma4_llama_router,
            )
        )
    assert result.value["sentiment"] in {"pos", "neg", "neu"}
    assert 0 <= int(result.value["confidence"]) <= 100
