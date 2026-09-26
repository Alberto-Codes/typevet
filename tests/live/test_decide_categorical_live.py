"""Live categorical decision via public entry (opt-in)."""

from __future__ import annotations

import math
from dataclasses import replace

import httpx
import pytest

from typevet import decide_categorical
from typevet.adapters.inbound.settings import load_llama_settings
from typevet.adapters.outbound.llama_cpp_scoring import LlamaCppCandidateScoringAdapter
from typevet.domain.candidate_scoring_request import CandidateTokenSpec
from typevet.evaluation.runner.live_gate import live_skip_reason
from typevet.gemma_answer_binding import resolve_answer_anchor

_LLAMA = load_llama_settings()
_PINNED_MODEL = "gemma-4-31b-24gib-kv11-decoder"
_PINNED_N_VOCAB = 262144

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
def gemma4_decide_ready() -> str:
    """Require router + pinned Gemma id; override env model to the pin."""
    pinned = replace(_LLAMA, default_model=_PINNED_MODEL)
    reason = live_skip_reason(pinned)
    if reason is not None:
        pytest.skip(reason)
    return _PINNED_MODEL


@pytest.mark.live
def test_decide_categorical_live_enum(gemma4_decide_ready: str) -> None:
    """Smoke: public entry scores via POST /completion (pre-sampling)."""
    live_settings = replace(_LLAMA, timeout=600.0, default_model=gemma4_decide_ready)
    base = live_settings.base_url.rstrip("/")
    prompt = "Pick a or b for a binary classification smoke."
    with httpx.Client(base_url=base, timeout=60.0) as client:
        rendered = (
            client.post(
                "/apply-template",
                json={
                    "model": gemma4_decide_ready,
                    "messages": [{"role": "user", "content": prompt}],
                    "add_generation_prompt": True,
                },
            )
            .raise_for_status()
            .json()["prompt"]
        )

        def tokenize_with_special(text: str) -> list[int]:
            return (
                client.post(
                    "/tokenize",
                    json={
                        "model": gemma4_decide_ready,
                        "content": text,
                        "add_special": True,
                    },
                )
                .raise_for_status()
                .json()["tokens"]
            )

        def tokenize_content(text: str) -> list[int]:
            return (
                client.post(
                    "/tokenize",
                    json={
                        "model": gemma4_decide_ready,
                        "content": text,
                        "add_special": False,
                    },
                )
                .raise_for_status()
                .json()["tokens"]
            )

        anchor = resolve_answer_anchor(
            rendered,
            tokenize_with_special=tokenize_with_special,
        )
        a_ids = tuple(int(t) for t in tokenize_content("a"))
        b_ids = tuple(int(t) for t in tokenize_content("b"))

    assert len(a_ids) == 1
    assert len(b_ids) == 1
    candidates = (
        CandidateTokenSpec("a", a_ids),
        CandidateTokenSpec("b", b_ids),
    )
    with LlamaCppCandidateScoringAdapter(
        base_url=live_settings.base_url,
        timeout=live_settings.timeout,
        n_vocab=_PINNED_N_VOCAB,
    ) as scoring_port:
        result = decide_categorical(
            field=_ENUM_SCHEMA,
            context=prompt,
            prefix=anchor.prefix,
            inject_prefix=True,
            model=gemma4_decide_ready,
            scoring_port=scoring_port,
            candidates=candidates,
        )
    assert result.value in ("a", "b")
    assert len(result.probabilities) == 2
    assert result.model == gemma4_decide_ready
    probs = [p for _, p in result.probabilities]
    assert all(math.isfinite(p) for p in probs)
    assert abs(sum(probs) - 1.0) < 1e-6
