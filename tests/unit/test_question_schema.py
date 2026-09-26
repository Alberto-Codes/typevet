"""Unit tests for question record → JSON Schema mapping (#102)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from typevet.domain import compile_json_schema
from typevet.domain.decisions import MAX_ENUM_CHOICES
from typevet.eval_boolq import (
    BOOLQ_ANSWER_NOUL_SCHEMA,
    PRIMARY_NOUL_NAME,
)
from typevet.eval_boolq import (
    questions_payload as boolq_questions,
)
from typevet.eval_clinc_shard import IN_SCOPE_NOUL_SCHEMA, choice_schema_for_domain
from typevet.eval_hyperpartisan import (
    HYPERPARTISAN_NOUL_SCHEMA,
)
from typevet.eval_hyperpartisan import (
    questions_payload as hyperpartisan_questions,
)
from typevet.eval_psai import questions_payload as psai_questions
from typevet.eval_psai_schema import METADATA_DECISIONS_SCHEMA
from typevet.question_schema import (
    compile_question_records,
    question_record_to_property,
    question_records_to_json_schema,
)

GO_EMOTION_SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "evals"
    / "fixtures"
    / "go_emotions_emotion_choice_schema_v1.json"
)


@pytest.mark.unit
def test_boolq_loader_questions_match_fixture_schema() -> None:
    schema = question_records_to_json_schema(boolq_questions())
    assert schema == BOOLQ_ANSWER_NOUL_SCHEMA
    decisions = compile_question_records(boolq_questions())
    assert decisions[0].name == PRIMARY_NOUL_NAME
    assert decisions[0].return_probabilities is True


@pytest.mark.unit
def test_hyperpartisan_loader_questions_match_fixture_schema() -> None:
    schema = question_records_to_json_schema(hyperpartisan_questions())
    assert schema == HYPERPARTISAN_NOUL_SCHEMA
    decisions = compile_question_records(hyperpartisan_questions())
    assert decisions[0].syntax == "Bool"


@pytest.mark.unit
def test_psai_loader_questions_match_metadata_schema() -> None:
    schema = question_records_to_json_schema(psai_questions())
    assert schema == METADATA_DECISIONS_SCHEMA
    assert len(compile_question_records(psai_questions())) == 7


@pytest.mark.unit
def test_choice_record_maps_to_string_enum() -> None:
    records = [
        {
            "name": "intent",
            "syntax": "Choice",
            "labels": ["a", "b"],
            "instructions": "Pick one.",
        }
    ]
    schema = question_records_to_json_schema(records)
    assert schema["properties"]["intent"]["enum"] == ["a", "b"]
    assert compile_json_schema(schema)[0].syntax == "Choice"


@pytest.mark.unit
def test_clinc_banking_choice_from_record_shape() -> None:
    labels = choice_schema_for_domain("banking")["properties"]["intent"]["enum"]
    records = [
        {
            "name": "intent",
            "syntax": "Choice",
            "labels": labels,
            "instructions": choice_schema_for_domain("banking")["properties"]["intent"][
                "instructions"
            ],
        }
    ]
    assert question_records_to_json_schema(records) == choice_schema_for_domain(
        "banking"
    )


@pytest.mark.unit
def test_noul_string_labels_match_clinc_in_scope_schema() -> None:
    records = [
        {
            "name": "in_scope",
            "syntax": "Noul",
            "labels": ["yes", "no"],
            "instructions": IN_SCOPE_NOUL_SCHEMA["properties"]["in_scope"][
                "instructions"
            ],
            "return_probabilities": True,
        }
    ]
    assert question_records_to_json_schema(records) == IN_SCOPE_NOUL_SCHEMA


@pytest.mark.unit
def test_score_digit_strings_become_integer_enum() -> None:
    _, field = question_record_to_property(
        {
            "name": "violations",
            "syntax": "Score",
            "labels": ["0", "1", "2", "3"],
            "instructions": "How many violations?",
        }
    )
    assert field == {
        "type": "integer",
        "enum": [0, 1, 2, 3],
        "instructions": "How many violations?",
    }


@pytest.mark.unit
def test_score_int_labels_and_go_emotions_fixture() -> None:
    on_disk = json.loads(GO_EMOTION_SCHEMA_PATH.read_text(encoding="utf-8"))
    labels = on_disk["properties"]["emotion"]["enum"]
    records = [
        {
            "name": "emotion",
            "syntax": "Choice",
            "labels": labels,
            "instructions": on_disk["properties"]["emotion"]["instructions"],
        }
    ]
    assert question_records_to_json_schema(records) == on_disk


@pytest.mark.unit
def test_rejects_empty_records() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        question_records_to_json_schema([])


@pytest.mark.unit
def test_rejects_duplicate_names() -> None:
    records = [
        {"name": "x", "syntax": "Noul", "instructions": "Q?"},
        {"name": "x", "syntax": "Noul", "instructions": "Q again?"},
    ]
    with pytest.raises(ValueError, match="duplicate"):
        question_records_to_json_schema(records)


@pytest.mark.unit
def test_rejects_choice_over_enum_cap() -> None:
    labels = [f"v{i}" for i in range(MAX_ENUM_CHOICES + 1)]
    with pytest.raises(ValueError, match=str(MAX_ENUM_CHOICES)):
        question_record_to_property(
            {
                "name": "big",
                "syntax": "Choice",
                "labels": labels,
                "instructions": "Too many.",
            }
        )
