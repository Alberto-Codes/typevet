"""Contract tests for Gemma template gate and enum binding (#118)."""

from __future__ import annotations

import pytest

from tests.fixtures.gemma_template_contract import exc_type_from_name, get_fixtures
from typevet.gemma_answer_binding import (
    bind_enum_labels,
    resolve_answer_anchor,
    termination_kind,
)
from typevet.gemma_served_template import ServedTemplateClass


@pytest.mark.contract
@pytest.mark.parametrize("fixture", get_fixtures(), ids=lambda row: row["name"])
def test_gemma_template_contract_fixture(fixture: dict) -> None:
    expect = fixture["expect"]
    if expect["kind"] == "anchor":
        anchor = resolve_answer_anchor(
            fixture["rendered"],
            tokenize_with_special=fixture["tokenize_special"],
        )
        assert anchor.template_class.value == expect["template_class"]
        if "prefix_token_count" in expect:
            assert anchor.prefix_token_count == expect["prefix_token_count"]
        if "byte_length" in expect:
            assert anchor.byte_length == expect["byte_length"]
        return

    if expect["kind"] == "bind":
        specs = bind_enum_labels(
            fixture["labels"],
            tokenize_content=fixture["tokenize_content"],
        )
        by_label = {spec.label: spec.token_ids for spec in specs}
        for label, token_ids in expect["token_ids"].items():
            assert by_label[label] == token_ids
        return

    if expect["kind"] == "termination":
        result = termination_kind(
            fixture["completion"],
            template_class=ServedTemplateClass(fixture["template_class"]),
            expected_label=fixture["expected_label"],
            tokenize_content=fixture["tokenize_content"],
        )
        assert result == expect["result"]
        return

    exc_type = exc_type_from_name(expect["exc_type"])
    with pytest.raises(exc_type):
        if "rendered" in fixture:
            resolve_answer_anchor(
                fixture["rendered"],
                tokenize_with_special=fixture["tokenize_special"],
            )
        else:
            bind_enum_labels(
                fixture["labels"],
                tokenize_content=fixture["tokenize_content"],
            )
