"""Unit tests for pure field-prompt rendering (#103)."""

from __future__ import annotations

import pytest

from typevet.domain.decisions import Decision
from typevet.field_prompt import (
    compose_scoring_prefix,
    gold_reference_markers,
    render_field_instructions,
)
from typevet.gemma_served_template import CHATML_ASSISTANT_HEADER


@pytest.mark.unit
def test_render_field_instructions_lists_choice_labels() -> None:
    decision = Decision(
        "route",
        "Pick the department.",
        ("billing", "technical"),
        syntax="Choice",
    )
    block = render_field_instructions(decision)
    assert "route:" in block
    assert "Pick the department." in block
    assert "- billing" in block
    assert "- technical" in block


@pytest.mark.unit
def test_render_field_instructions_includes_criteria_descriptions() -> None:
    decision = Decision("m", "Choose.", ("a", "b"), syntax="Choice")
    block = render_field_instructions(
        decision,
        choice_criteria={"a": "Alpha option", "b": "Beta option"},
    )
    assert "- a: Alpha option" in block
    assert "- b: Beta option" in block


@pytest.mark.unit
def test_render_field_instructions_bool_and_unicode_field_name() -> None:
    decision = Decision(
        '"quoted"',
        "Yes or no?",
        (True, False),
        syntax="Bool",
    )
    block = render_field_instructions(decision)
    assert '"quoted":' in block
    assert "- true" in block
    assert "- false" in block


@pytest.mark.unit
def test_render_field_instructions_no_gold_reference_markers() -> None:
    decision = Decision("x", "Pick one.", ("a", "b"), syntax="Choice")
    block = render_field_instructions(decision)
    lowered = block.lower()
    for marker in gold_reference_markers():
        assert marker not in lowered


@pytest.mark.unit
def test_compose_scoring_prefix_ends_with_chatml_assistant_header() -> None:
    decision = Decision("c", "Question?", ("a", "b"), syntax="Choice")
    field_block = render_field_instructions(decision)
    prefix = compose_scoring_prefix(context="User task text.", field_block=field_block)
    assert "User task text." in prefix
    assert field_block in prefix
    assert prefix.endswith(CHATML_ASSISTANT_HEADER)
