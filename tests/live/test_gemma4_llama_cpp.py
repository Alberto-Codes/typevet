"""Live path against local llama.cpp + Gemma 4 (opt-in)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from typevet.adapters.inbound.settings import llama_cpp_adapter, load_llama_settings
from typevet.domain.models import GenerationRequest
from typevet.eval_runner_live_gate import live_skip_reason

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


@pytest.fixture
def gemma4_llama_router() -> str:
    """Probe local llama.cpp only when a live test is selected to run."""
    reason = live_skip_reason(_LLAMA)
    if reason is not None:
        pytest.skip(reason)
    assert MODEL is not None
    return MODEL


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
