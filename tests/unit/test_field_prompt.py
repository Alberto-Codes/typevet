"""Unit tests for pure field-prompt rendering (#103)."""

from __future__ import annotations

import pytest

from typevet.adapters.outbound.gemma.scoring_prefix import compose_scoring_prefix
from typevet.adapters.outbound.gemma.served_template import CHATML_ASSISTANT_HEADER
from typevet.domain.decisions import Decision
from typevet.domain.field_instructions import (
    gold_reference_markers,
    render_field_instructions,
)


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
def test_render_field_instructions_choice_control_mapping_with_descriptions() -> None:
    decision = Decision(
        "route",
        "Pick the department.",
        ("billing", "technical"),
        syntax="Choice",
    )
    block = render_field_instructions(
        decision,
        choice_criteria={"billing": "Money", "technical": "Bugs"},
        original_labels=("billing", "technical"),
    )
    assert "\n0 → billing: Money" in block
    assert "\n1 → technical: Bugs" in block
    assert "exactly one control string" in block.lower()


@pytest.mark.unit
def test_render_field_instructions_six_option_choice_omits_control_word() -> None:
    labels = ("alpha", "bravo", "charlie", "delta", "echo", "foxtrot")
    decision = Decision("kind", "Pick the kind.", labels, syntax="Choice")
    criteria = {label: f"About {label}" for label in labels}
    block = render_field_instructions(
        decision,
        choice_criteria=criteria,
        original_labels=labels,
    )
    assert "Control" not in block
    for index, label in enumerate(labels):
        assert f"{index} → {label}: About {label}" in block


@pytest.mark.unit
def test_render_field_instructions_noul_control_zero_is_false() -> None:
    decision = Decision("flag", "Is it urgent?", (False, True), syntax="Bool")
    block = render_field_instructions(
        decision,
        choice_criteria={"true": "Yes", "false": "No"},
        original_labels=("false", "true"),
    )
    false_pos = block.index("Control 0 → false: No")
    true_pos = block.index("Control 1 → true: Yes")
    assert false_pos < true_pos


@pytest.mark.unit
def test_render_field_instructions_score_maps_level_index_to_rubric() -> None:
    decision = Decision("quality", "Rate:", (0, 1, 2), syntax="Choice")
    block = render_field_instructions(
        decision,
        choice_criteria={"0": "Poor", "1": "Fair", "2": "Good"},
        original_labels=("0", "1", "2"),
    )
    assert "Control 0 → 0: Poor" in block
    assert "Control 2 → 2: Good" in block


@pytest.mark.unit
def test_render_field_instructions_choice_mapping_follows_criteria_order() -> None:
    decision = Decision("m", "Choose.", ("z_last", "a_first"), syntax="Choice")
    block = render_field_instructions(
        decision,
        choice_criteria={"z_last": "Z", "a_first": "A"},
        original_labels=("z_last", "a_first"),
    )
    z_pos = block.index("\n0 → z_last")
    a_pos = block.index("\n1 → a_first")
    assert z_pos < a_pos


@pytest.mark.unit
def test_compose_scoring_prefix_ends_with_chatml_assistant_header() -> None:
    decision = Decision("c", "Question?", ("a", "b"), syntax="Choice")
    field_block = render_field_instructions(decision)
    prefix = compose_scoring_prefix(context="User task text.", field_block=field_block)
    assert "User task text." in prefix
    assert field_block in prefix
    assert prefix.endswith(CHATML_ASSISTANT_HEADER)
