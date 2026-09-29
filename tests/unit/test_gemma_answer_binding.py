"""Unit tests for Gemma answer anchor and enum token binding (#118)."""

from __future__ import annotations

import pytest

from tests.fixtures.gemma_template_contract import (
    PINNED_DEGRADED_PROMPT,
    degraded_chatml_rendered,
    native_gemma4_rendered,
    pinned_tokenize_content,
    pinned_tokenize_with_special,
)
from typevet.adapters.outbound.gemma.answer_binding import (
    bind_enum_label,
    resolve_answer_anchor,
    termination_kind,
)
from typevet.adapters.outbound.gemma.scoring_prefix import compose_media_scoring_prefix
from typevet.adapters.outbound.gemma.served_template import (
    GEMMA4_MODEL_TURN_HEADER,
    GEMMA4_NO_THINKING_PREFILL,
    GEMMA4_TURN_CLOSE,
    GEMMA4_TURN_OPEN,
    ServedTemplateClass,
)
from typevet.domain.errors import GemmaTemplateError
from typevet.domain.media import MEDIA_MARKER


@pytest.mark.unit
def test_resolve_anchor_no_thinking_degraded_chatml() -> None:
    anchor = resolve_answer_anchor(
        PINNED_DEGRADED_PROMPT,
        tokenize_with_special=pinned_tokenize_with_special,
    )
    assert anchor.template_class is ServedTemplateClass.DEGRADED_CHATML
    assert anchor.prefix == PINNED_DEGRADED_PROMPT


@pytest.mark.unit
def test_bind_enum_label_rejects_leading_whitespace() -> None:
    with pytest.raises(GemmaTemplateError, match="whitespace"):
        bind_enum_label(" anger", tokenize_content=pinned_tokenize_content)


@pytest.mark.unit
def test_bind_enum_label_exact_string_pinned_ids() -> None:
    assert bind_enum_label("anger", tokenize_content=pinned_tokenize_content) == (4751,)


@pytest.mark.unit
def test_duplicated_bos_fails() -> None:
    with pytest.raises(GemmaTemplateError, match="duplicated BOS"):
        resolve_answer_anchor(
            PINNED_DEGRADED_PROMPT,
            tokenize_with_special=lambda _text: [2, 2, 1],
        )


@pytest.mark.unit
def test_missing_answer_boundary_fails() -> None:
    bad = degraded_chatml_rendered("Hi").replace("assistant", "assist")
    with pytest.raises(GemmaTemplateError, match="answer boundary"):
        resolve_answer_anchor(bad, tokenize_with_special=pinned_tokenize_with_special)


@pytest.mark.unit
def test_no_thinking_rejects_think_trigger_in_prefix() -> None:
    native = native_gemma4_rendered("Q?<|think|>")
    with pytest.raises(GemmaTemplateError, match="no_thinking"):
        resolve_answer_anchor(
            native, tokenize_with_special=pinned_tokenize_with_special
        )


def _native_no_thinking_final_turn(*, prior_model_body: str = "") -> str:
    """Native prompt with prior turns; final model turn ends at no-thinking prefill."""
    prior = ""
    if prior_model_body:
        prior = (
            f"{GEMMA4_TURN_OPEN}user\nFirst question.{GEMMA4_TURN_CLOSE}\n"
            f"{GEMMA4_MODEL_TURN_HEADER}{prior_model_body}{GEMMA4_TURN_CLOSE}\n"
            f"{GEMMA4_TURN_OPEN}user\nFollow-up.{GEMMA4_TURN_CLOSE}\n"
        )
    return f"{prior}{GEMMA4_MODEL_TURN_HEADER}{GEMMA4_NO_THINKING_PREFILL}"


@pytest.mark.unit
def test_resolve_anchor_native_header_only_still_accepted() -> None:
    rendered = native_gemma4_rendered("Pick one label.")
    anchor = resolve_answer_anchor(
        rendered,
        tokenize_with_special=pinned_tokenize_with_special,
    )
    assert anchor.template_class is ServedTemplateClass.NATIVE_GEMMA4_TURN
    assert anchor.prefix == rendered
    assert anchor.byte_length == len(rendered.encode("utf-8"))
    assert anchor.prefix_token_count == len(pinned_tokenize_with_special(rendered))


@pytest.mark.unit
def test_resolve_anchor_native_no_thinking_prefill_after_final_model_turn() -> None:
    rendered = _native_no_thinking_final_turn(prior_model_body="joy")
    anchor = resolve_answer_anchor(
        rendered,
        tokenize_with_special=pinned_tokenize_with_special,
    )
    assert anchor.template_class is ServedTemplateClass.NATIVE_GEMMA4_TURN
    assert anchor.prefix == rendered
    assert anchor.byte_length == len(rendered.encode("utf-8"))
    assert anchor.prefix_token_count == len(pinned_tokenize_with_special(rendered))


@pytest.mark.unit
def test_resolve_anchor_native_rejects_nonempty_thought_after_final_turn() -> None:
    rendered = _native_no_thinking_final_turn().replace(
        GEMMA4_NO_THINKING_PREFILL,
        "<|channel>thought\nhmm<channel|>",
    )
    with pytest.raises(GemmaTemplateError, match="unexpected content after model turn"):
        resolve_answer_anchor(
            rendered, tokenize_with_special=pinned_tokenize_with_special
        )


@pytest.mark.unit
def test_resolve_anchor_accepts_gemma4_media_scoring_prefix_with_prefill() -> None:
    prefix = compose_media_scoring_prefix(
        context=f"{MEDIA_MARKER}\nReceipt photo.",
        field_block="Control 0 → billing",
        template_class=ServedTemplateClass.NATIVE_GEMMA4_TURN,
    )
    anchor = resolve_answer_anchor(
        prefix,
        tokenize_with_special=pinned_tokenize_with_special,
    )
    assert anchor.template_class is ServedTemplateClass.NATIVE_GEMMA4_TURN
    assert anchor.prefix == prefix
    bad = compose_media_scoring_prefix(
        context=f"{MEDIA_MARKER}\nReceipt photo.<|think|>",
        field_block="Control 0 → billing",
        template_class=ServedTemplateClass.NATIVE_GEMMA4_TURN,
    )
    with pytest.raises(GemmaTemplateError, match="no_thinking"):
        resolve_answer_anchor(bad, tokenize_with_special=pinned_tokenize_with_special)


@pytest.mark.unit
def test_resolve_anchor_native_reproduces_missing_boundary_with_prefill_only() -> None:
    """Regression: header+prefill must not raise missing-boundary (#144)."""
    rendered = (
        f"{GEMMA4_TURN_OPEN}user\nQ.{GEMMA4_TURN_CLOSE}\n"
        f"{GEMMA4_MODEL_TURN_HEADER}{GEMMA4_NO_THINKING_PREFILL}"
    )
    anchor = resolve_answer_anchor(
        rendered,
        tokenize_with_special=pinned_tokenize_with_special,
    )
    assert anchor.template_class is ServedTemplateClass.NATIVE_GEMMA4_TURN


@pytest.mark.unit
def test_termination_premature_on_partial_label() -> None:
    kind = termination_kind(
        "admi",
        template_class=ServedTemplateClass.DEGRADED_CHATML,
        expected_label="admiration",
        tokenize_content=pinned_tokenize_content,
    )
    assert kind == "premature"
