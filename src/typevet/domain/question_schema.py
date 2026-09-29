"""Map JevBench-shaped question records to JSON Schema for compilation (#102).

Loaders export task ``questions`` lists with ``name``, ``syntax`` (Noul,
Choice, Score), ``instructions``, and optional ``labels`` /
``return_probabilities``. These pure helpers build object schemas that
``compile_json_schema`` accepts.

Examples:
    ```python
    from typevet.evaluation.datasets.boolq import (
        BOOLQ_ANSWER_NOUL_SCHEMA,
        questions_payload,
    )
    from typevet.domain.question_schema import question_records_to_json_schema

    assert question_records_to_json_schema(questions_payload()) == (
        BOOLQ_ANSWER_NOUL_SCHEMA
    )
    ```

See Also:
    - [typevet.domain.decision_compile][]: ``compile_json_schema``
    - docs/reference/question-schema-map.md
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Final, Literal

from typevet.domain.decision_compile import compile_json_schema
from typevet.domain.decisions import MAX_ENUM_CHOICES, Decision

__all__ = [
    "SyntaxName",
    "compile_question_records",
    "question_record_to_property",
    "question_records_to_json_schema",
]

SyntaxName = Literal["Noul", "Choice", "Score"]
_SUPPORTED_SYNTAX: Final[frozenset[str]] = frozenset({"Noul", "Choice", "Score"})
_NOUL_LABELED_ARITY: Final[int] = 2
_SCORE_MIN_LABELS: Final[int] = 2


def _optional_field_metadata(
    name: str, record: Mapping[str, Any], field: dict[str, Any]
) -> None:
    if "return_probabilities" in record:
        flag = record["return_probabilities"]
        if type(flag) is not bool:
            msg = f"return_probabilities for {name!r} must be a boolean"
            raise ValueError(msg)
        field["return_probabilities"] = flag
    if "depends_on" in record:
        deps = record["depends_on"]
        if not isinstance(deps, list) or any(
            not isinstance(dep, str) or not dep for dep in deps
        ):
            msg = f"depends_on for {name!r} must be a list of non-empty strings"
            raise ValueError(msg)
        field["depends_on"] = deps
    if "permutations" in record:
        permutations = record["permutations"]
        if permutations != "all" and (
            type(permutations) is not int or permutations < 1
        ):
            msg = f"permutations for {name!r} must be a positive integer or 'all'"
            raise ValueError(msg)
        field["permutations"] = permutations


def question_record_to_property(
    record: Mapping[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Map one System One-style question record to a JSON Schema property.

    Args:
        record: Mapping with ``name``, ``syntax``, and ``instructions``. Choice
            and Score require ``labels``. Noul uses boolean when ``labels`` is
            omitted; two string labels yield a closed string enum (BoolQ-style).

    Returns:
        Field name and property schema object.

    Raises:
        ValueError: When the record shape or labels are invalid.
        TypeError: When ``record`` or ``labels`` have the wrong type.
    """
    if not isinstance(record, Mapping):
        msg = "each question record must be a mapping"
        raise TypeError(msg)
    name = record.get("name")
    if not isinstance(name, str) or not name:
        msg = "question record name must be a non-empty string"
        raise ValueError(msg)
    syntax = record.get("syntax")
    if syntax not in _SUPPORTED_SYNTAX:
        msg = f"question {name!r} syntax must be one of Noul, Choice, Score"
        raise ValueError(msg)
    instructions = record.get("instructions")
    if not isinstance(instructions, str) or not instructions:
        msg = f"question {name!r} instructions must be a non-empty string"
        raise ValueError(msg)

    field: dict[str, Any] = {"instructions": instructions}
    _optional_field_metadata(name, record, field)

    if syntax == "Choice":
        _apply_choice_field(name, record, field)
    elif syntax == "Score":
        _apply_score_field(name, record, field)
    else:
        _apply_noul_field(name, record, field)
    return name, field


def question_records_to_json_schema(
    records: Sequence[Mapping[str, Any]],
    *,
    additional_properties: bool = False,
) -> dict[str, Any]:
    """Build a root object schema from ordered question records.

    Args:
        records: Non-empty sequence of question mappings (loader export order).
        additional_properties: Value for ``additionalProperties`` on the root.

    Returns:
        Object schema with ``properties``, ``required``, and ``type: object``.

    Raises:
        ValueError: When ``records`` is empty or contains duplicate names.
    """
    if not records:
        msg = "question records must be a non-empty sequence"
        raise ValueError(msg)
    properties: dict[str, Any] = {}
    required: list[str] = []
    for record in records:
        field_name, field_schema = question_record_to_property(record)
        if field_name in properties:
            msg = f"duplicate question name {field_name!r}"
            raise ValueError(msg)
        properties[field_name] = field_schema
        required.append(field_name)
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": additional_properties,
    }


def compile_question_records(
    records: Sequence[Mapping[str, Any]],
    *,
    additional_properties: bool = False,
) -> list[Decision]:
    """Compile question records to TypeLLM ``Decision`` values.

    Args:
        records: Loader-shaped question list.
        additional_properties: Forwarded to ``question_records_to_json_schema``.

    Returns:
        Decisions from ``compile_json_schema`` in property order.
    """
    schema = question_records_to_json_schema(
        records,
        additional_properties=additional_properties,
    )
    return compile_json_schema(schema)


def _labels_list(name: str, record: Mapping[str, Any]) -> list[Any]:
    labels = record.get("labels")
    if not isinstance(labels, list) or not labels:
        msg = f"question {name!r} requires a non-empty labels list"
        raise ValueError(msg)
    if len(labels) > MAX_ENUM_CHOICES:
        msg = (
            f"question {name!r} has {len(labels)} labels; "
            f"the maximum is {MAX_ENUM_CHOICES}"
        )
        raise ValueError(msg)
    return list(labels)


def _apply_choice_field(
    name: str, record: Mapping[str, Any], field: dict[str, Any]
) -> None:
    labels = _labels_list(name, record)
    if any(not isinstance(label, str) or not label for label in labels):
        msg = f"Choice labels for {name!r} must be non-empty strings"
        raise ValueError(msg)
    field["type"] = "string"
    field["enum"] = labels


def _apply_score_field(
    name: str, record: Mapping[str, Any], field: dict[str, Any]
) -> None:
    labels = _labels_list(name, record)
    if len(labels) < _SCORE_MIN_LABELS:
        msg = f"Score question {name!r} needs at least two labels"
        raise ValueError(msg)
    if all(type(label) is int for label in labels):
        field["type"] = "integer"
        field["enum"] = labels
        return
    if all(type(label) is float and label.is_integer() for label in labels):
        field["type"] = "integer"
        field["enum"] = [int(label) for label in labels]
        return
    if all(isinstance(label, str) and label.isdigit() for label in labels):
        field["type"] = "integer"
        field["enum"] = [int(label) for label in labels]
        return
    if all(isinstance(label, str) and label for label in labels):
        field["type"] = "string"
        field["enum"] = labels
        return
    msg = f"Score labels for {name!r} must be ints, digit strings, or non-empty strings"
    raise ValueError(msg)


def _apply_noul_field(
    name: str, record: Mapping[str, Any], field: dict[str, Any]
) -> None:
    if "labels" not in record:
        field["type"] = "boolean"
        return
    labels = record["labels"]
    if labels is None:
        field["type"] = "boolean"
        return
    if not isinstance(labels, list):
        msg = f"Noul labels for {name!r} must be a list when present"
        raise TypeError(msg)
    if len(labels) == 0:
        field["type"] = "boolean"
        return
    if len(labels) != _NOUL_LABELED_ARITY or any(
        not isinstance(label, str) or not label for label in labels
    ):
        msg = (
            f"Noul question {name!r} with labels must have exactly two "
            "non-empty strings, or omit labels for boolean Noul"
        )
        raise ValueError(msg)
    field["type"] = "string"
    field["enum"] = list(labels)
