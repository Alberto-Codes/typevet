"""Unit tests for JSON Schema → Decision compilation."""

from __future__ import annotations

from typing import Any, cast

import pytest

from typevet.domain import Decision, SchemaError, compile_json_schema, dependency_layers
from typevet.domain.decisions import MAX_ENUM_CHOICES


def _object_schema(properties: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
        **extra,
    }


@pytest.mark.unit
def test_compile_happy_path_mixed_fields() -> None:
    schema = _object_schema(
        {
            "flag": {"type": "boolean", "description": "On or off?"},
            "color": {
                "type": "string",
                "enum": ["red", "green", "blue"],
                "instructions": "Pick a color.",
            },
            "count": {"type": "integer", "minimum": 0, "maximum": 10},
            "note": {"type": "string", "maxLength": 40},
        }
    )
    decisions = compile_json_schema(schema)
    assert [d.name for d in decisions] == ["flag", "color", "count", "note"]

    flag, color, count, note = decisions
    assert flag.syntax == "Bool"
    assert flag.choices == (True, False)
    assert flag.question == "On or off?"

    assert color.syntax == "Choice"
    assert color.choices == ("red", "green", "blue")
    assert color.question == "Pick a color."

    assert count.syntax == "Integer"
    assert count.choices == ()
    assert count.minimum == 0
    assert count.maximum == 10

    assert note.syntax == "Text"
    assert note.text_type is True
    assert note.max_length == 40
    assert note.choices == ()


@pytest.mark.unit
def test_compile_default_question_when_no_description() -> None:
    schema = _object_schema({"x": {"type": "boolean"}})
    decisions = compile_json_schema(schema)
    assert decisions[0].question == 'Choose the value for "x".'


@pytest.mark.unit
def test_compile_rejects_non_object_root() -> None:
    with pytest.raises(SchemaError, match="root JSON Schema type"):
        compile_json_schema({"type": "array", "properties": {"a": {"type": "string"}}})


@pytest.mark.unit
def test_compile_rejects_empty_properties() -> None:
    with pytest.raises(SchemaError, match="non-empty"):
        compile_json_schema(
            {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            }
        )


@pytest.mark.unit
def test_compile_rejects_unknown_required_field() -> None:
    schema = {
        "type": "object",
        "properties": {"a": {"type": "boolean"}},
        "required": ["a", "missing"],
        "additionalProperties": False,
    }
    with pytest.raises(SchemaError, match="missing from properties"):
        compile_json_schema(schema)


@pytest.mark.unit
def test_compile_rejects_enum_over_max_choices() -> None:
    values = list(range(MAX_ENUM_CHOICES + 1))
    schema = _object_schema({"n": {"type": "integer", "enum": values}})
    with pytest.raises(SchemaError, match=str(MAX_ENUM_CHOICES)):
        compile_json_schema(schema)


@pytest.mark.unit
def test_compile_rejects_legacy_question_keyword() -> None:
    schema = _object_schema({"x": {"type": "boolean", "question": "Old?"}})
    with pytest.raises(SchemaError, match="no longer supported"):
        compile_json_schema(schema)


@pytest.mark.unit
def test_dependency_layers_detects_cycle() -> None:
    schema = _object_schema(
        {
            "a": {"type": "boolean", "depends_on": ["b"]},
            "b": {"type": "boolean", "depends_on": ["a"]},
        }
    )
    with pytest.raises(SchemaError, match="cycle"):
        compile_json_schema(schema)


@pytest.mark.unit
def test_compile_rejects_non_mapping_schema() -> None:
    with pytest.raises(SchemaError, match="mapping"):
        compile_json_schema(cast(Any, ["not", "a", "mapping"]))


@pytest.mark.unit
def test_dependency_layers_standalone() -> None:
    d1 = Decision("a", "q", (), depends_on=None)
    d2 = Decision("b", "q", (), depends_on=("a",))
    layers = dependency_layers([d2, d1])
    assert [[x.name for x in layer] for layer in layers] == [["a"], ["b"]]
