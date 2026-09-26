"""Unit tests for question record → JSON Schema mapping (#102)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

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


@pytest.mark.unit
def test_rejects_non_mapping_record() -> None:
    with pytest.raises(TypeError, match="mapping"):
        question_record_to_property(cast(Any, ["not", "a", "mapping"]))


@pytest.mark.unit
def test_rejects_bad_name_syntax_and_instructions() -> None:
    with pytest.raises(ValueError, match="name"):
        question_record_to_property(
            {"name": "", "syntax": "Noul", "instructions": "Q?"}
        )
    with pytest.raises(ValueError, match="syntax"):
        question_record_to_property(
            {"name": "q", "syntax": "Invalid", "instructions": "Q?"}
        )
    with pytest.raises(ValueError, match="instructions"):
        question_record_to_property({"name": "q", "syntax": "Noul", "instructions": ""})


@pytest.mark.unit
def test_optional_metadata_validation_errors() -> None:
    base = {"name": "q", "syntax": "Noul", "instructions": "Q?"}
    with pytest.raises(ValueError, match="return_probabilities"):
        question_record_to_property({**base, "return_probabilities": "yes"})
    with pytest.raises(ValueError, match="depends_on"):
        question_record_to_property({**base, "depends_on": "parent"})
    with pytest.raises(ValueError, match="depends_on"):
        question_record_to_property({**base, "depends_on": [""]})
    with pytest.raises(ValueError, match="permutations"):
        question_record_to_property({**base, "permutations": 0})


@pytest.mark.unit
def test_optional_metadata_accepts_valid_flags() -> None:
    _, field = question_record_to_property(
        {
            "name": "q",
            "syntax": "Noul",
            "instructions": "Q?",
            "return_probabilities": False,
            "depends_on": ["parent"],
            "permutations": "all",
        }
    )
    assert field["return_probabilities"] is False
    assert field["depends_on"] == ["parent"]
    assert field["permutations"] == "all"


@pytest.mark.unit
def test_choice_requires_labels_and_non_empty_strings() -> None:
    with pytest.raises(ValueError, match="labels list"):
        question_record_to_property(
            {"name": "c", "syntax": "Choice", "instructions": "Pick."}
        )
    with pytest.raises(ValueError, match="non-empty strings"):
        question_record_to_property(
            {
                "name": "c",
                "syntax": "Choice",
                "labels": ["ok", ""],
                "instructions": "Pick.",
            }
        )


@pytest.mark.unit
def test_score_label_variants_and_errors() -> None:
    with pytest.raises(ValueError, match="at least two"):
        question_record_to_property(
            {
                "name": "s",
                "syntax": "Score",
                "labels": [1],
                "instructions": "Rate.",
            }
        )
    _, int_field = question_record_to_property(
        {
            "name": "s",
            "syntax": "Score",
            "labels": [1, 2],
            "instructions": "Rate.",
        }
    )
    assert int_field["type"] == "integer"
    assert int_field["enum"] == [1, 2]
    _, float_field = question_record_to_property(
        {
            "name": "s2",
            "syntax": "Score",
            "labels": [1.0, 2.0],
            "instructions": "Rate.",
        }
    )
    assert float_field["enum"] == [1, 2]
    _, str_field = question_record_to_property(
        {
            "name": "s3",
            "syntax": "Score",
            "labels": ["low", "high"],
            "instructions": "Rate.",
        }
    )
    assert str_field["type"] == "string"
    with pytest.raises(ValueError, match="must be ints"):
        question_record_to_property(
            {
                "name": "s4",
                "syntax": "Score",
                "labels": [1, "two"],
                "instructions": "Rate.",
            }
        )


@pytest.mark.unit
def test_noul_label_shapes() -> None:
    _, bare = question_record_to_property(
        {"name": "n", "syntax": "Noul", "instructions": "Yes or no?"}
    )
    assert bare["type"] == "boolean"
    _, null_labels = question_record_to_property(
        {
            "name": "n2",
            "syntax": "Noul",
            "instructions": "Yes or no?",
            "labels": None,
        }
    )
    assert null_labels["type"] == "boolean"
    _, empty = question_record_to_property(
        {
            "name": "n3",
            "syntax": "Noul",
            "instructions": "Yes or no?",
            "labels": [],
        }
    )
    assert empty["type"] == "boolean"
    with pytest.raises(TypeError, match="must be a list"):
        question_record_to_property(
            {
                "name": "n4",
                "syntax": "Noul",
                "instructions": "Yes or no?",
                "labels": "yes",
            }
        )
    with pytest.raises(ValueError, match="exactly two"):
        question_record_to_property(
            {
                "name": "n5",
                "syntax": "Noul",
                "instructions": "Yes or no?",
                "labels": ["only-one"],
            }
        )


@pytest.mark.unit
def test_compile_question_records_round_trip() -> None:
    records = [
        {
            "name": "answer",
            "syntax": "Noul",
            "instructions": "Answer yes or no.",
            "return_probabilities": True,
        }
    ]
    decisions = compile_question_records(records)
    assert len(decisions) == 1
    assert decisions[0].name == "answer"
