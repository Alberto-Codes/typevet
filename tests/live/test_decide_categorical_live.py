"""Live categorical decision via public entry (opt-in)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from typevet import decide_categorical
from typevet.adapters.inbound.settings import load_llama_settings
from typevet.adapters.outbound.llama_cpp_scoring import LlamaCppCandidateScoringAdapter
from typevet.domain.candidate_scoring_request import CandidateTokenSpec
from typevet.eval_runner_live_gate import live_skip_reason

_LLAMA = load_llama_settings()
_PINNED_MODEL = "gemma-4-31b-24gib-kv11-decoder"

_ENUM_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {
            "type": "string",
            "enum": ["a", "b"],
        }
    },
    "required": ["label"],
    "additionalProperties": False,
}


@pytest.fixture
def llama_decide_model() -> str:
    reason = live_skip_reason(_LLAMA)
    if reason is not None:
        pytest.skip(reason)
    assert _LLAMA.default_model is not None
    return _LLAMA.default_model


@pytest.mark.live
def test_decide_categorical_live_enum(llama_decide_model: str) -> None:
    """Smoke: public entry scores via POST /completion (pre-sampling)."""
    live_settings = replace(_LLAMA, timeout=600.0)
    candidates = (
        CandidateTokenSpec("a", (32,)),
        CandidateTokenSpec("b", (33,)),
    )
    with LlamaCppCandidateScoringAdapter(
        base_url=live_settings.base_url,
        timeout=live_settings.timeout,
        n_vocab=262144,
    ) as scoring_port:
        result = decide_categorical(
            field=_ENUM_SCHEMA,
            prompt="Pick a or b.",
            prefix="Answer:",
            model=llama_decide_model,
            scoring_port=scoring_port,
            candidates=candidates,
        )
    assert result.value in ("a", "b")
    assert len(result.probabilities) == 2
    assert result.model
    _ = _PINNED_MODEL  # documented pin for template/scoring family
