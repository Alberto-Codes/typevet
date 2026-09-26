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
from typevet.domain.errors import GemmaTemplateError
from typevet.gemma_answer_binding import (
    bind_enum_label,
    resolve_answer_anchor,
    termination_kind,
)
from typevet.gemma_served_template import ServedTemplateClass


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


@pytest.mark.unit
def test_termination_premature_on_partial_label() -> None:
    kind = termination_kind(
        "admi",
        template_class=ServedTemplateClass.DEGRADED_CHATML,
        expected_label="admiration",
        tokenize_content=pinned_tokenize_content,
    )
    assert kind == "premature"
