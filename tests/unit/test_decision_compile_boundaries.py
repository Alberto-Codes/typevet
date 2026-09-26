"""Boundary tests for ``compile_json_schema`` (issue #83 matrix).

Mutation evidence (manual, pre-merge): each line names a one-line defect in
``decision_compile.py`` that makes the cited test fail.

- ``test_nullable_open_string_sets_nullable_flag``: force ``nullable=False`` in
  ``_decision_for_open_text`` → ``decisions[0].nullable is False``.
- ``test_nullable_boolean_without_enum_includes_none_choice``: drop ``+ [None] * nullable``
  in ``_enum_values_and_syntax`` for booleans → ``None not in choices``.
- ``test_compile_rejects_non_object_root`` (in ``test_decisions.py``): accept
  ``type`` ``array`` at root → no ``SchemaError``.
- ``test_permutations_all_at_max_factorial_budget``: skip ``budget > MAX_PERMUTATIONS``
  check → compiles 7-item enum with ``permutations`` ``all``.
- ``test_return_probabilities_rejected_on_open_text``: drop the enum/boolean guard in
  ``_return_probabilities`` → compiles text field with ``return_probabilities``.
"""

from __future__ import annotations

import math
from typing import Any, cast

import pytest

from typevet.domain import SchemaError, compile_json_schema
from typevet.domain.decisions import MAX_PERMUTATIONS

# ---------------------------------------------------------------------------
# 1. Nullable type forms and enum values
# ---------------------------------------------------------------------------


def _object_schema(properties: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
        **extra,
    }


@pytest.mark.unit
def test_nullable_open_string_sets_nullable_flag() -> None:
    schema = _object_schema({"note": {"type": ["string", "null"]}})
    decisions = compile_json_schema(schema)
    assert decisions[0].syntax == "Text"
    assert decisions[0].nullable is True
    assert decisions[0].choices == ()


@pytest.mark.unit
def test_nullable_boolean_without_enum_includes_none_choice() -> None:
    schema = _object_schema({"flag": {"type": ["boolean", "null"]}})
    decisions = compile_json_schema(schema)
    assert decisions[0].syntax == "Bool"
    assert decisions[0].nullable is True
    assert decisions[0].choices == (True, False, None)


@pytest.mark.unit
def test_nullable_string_enum_may_list_null() -> None:
    schema = _object_schema(
        {
            "color": {
                "type": ["string", "null"],
                "enum": ["red", None],
            }
        }
    )
    decisions = compile_json_schema(schema)
    assert decisions[0].syntax == "Choice"
    assert decisions[0].nullable is True
    assert decisions[0].choices == ("red", None)


@pytest.mark.unit
def test_non_nullable_string_enum_rejects_null_member() -> None:
    schema = _object_schema({"color": {"type": "string", "enum": ["red", None]}})
    with pytest.raises(SchemaError, match="do not match type"):
        compile_json_schema(schema)


@pytest.mark.unit
def test_nullable_type_rejects_multi_non_null_union() -> None:
    schema = _object_schema({"x": {"type": ["string", "integer", "null"]}})
    with pytest.raises(SchemaError, match='one type or \\[type, "null"\\]'):
        compile_json_schema(schema)


@pytest.mark.unit
def test_nullable_type_rejects_wrong_null_arity() -> None:
    schema = _object_schema({"x": {"type": ["string"]}})
    with pytest.raises(SchemaError, match='one type or \\[type, "null"\\]'):
        compile_json_schema(schema)


@pytest.mark.unit
def test_omitted_property_type_is_not_supported() -> None:
    schema = _object_schema({"x": {"description": "no type keyword"}})
    with pytest.raises(NotImplementedError, match="unsupported JSON Schema type"):
        compile_json_schema(schema)


# ---------------------------------------------------------------------------
# 2. Permutation budget boundaries including ``all``
# ---------------------------------------------------------------------------


def _string_enum_field(count: int, **extra: Any) -> dict[str, Any]:
    return {
        "type": "string",
        "enum": [f"v{i}" for i in range(count)],
        **extra,
    }


@pytest.mark.unit
def test_permutations_key_absent_defaults_without_budget_check() -> None:
    schema = _object_schema({"tags": {"type": "string", "enum": ["a", "b", "c"]}})
    decisions = compile_json_schema(schema)
    assert decisions[0].permutations == 1


@pytest.mark.unit
def test_permutations_all_at_max_factorial_budget() -> None:
    size = next(n for n in range(1, 20) if math.factorial(n) == MAX_PERMUTATIONS)
    schema = _object_schema({"tags": _string_enum_field(size, permutations="all")})
    decisions = compile_json_schema(schema)
    assert decisions[0].permutations == "all"


@pytest.mark.unit
def test_permutations_all_exceeds_max_factorial_budget() -> None:
    size = next(n for n in range(1, 20) if math.factorial(n) > MAX_PERMUTATIONS)
    schema = _object_schema({"tags": _string_enum_field(size, permutations="all")})
    with pytest.raises(SchemaError, match=str(MAX_PERMUTATIONS)):
        compile_json_schema(schema)


@pytest.mark.unit
def test_permutations_integer_budget_at_cap() -> None:
    size = next(n for n in range(1, 20) if math.factorial(n) == MAX_PERMUTATIONS)
    schema = _object_schema(
        {"tags": _string_enum_field(size, permutations=MAX_PERMUTATIONS)}
    )
    decisions = compile_json_schema(schema)
    assert decisions[0].permutations == MAX_PERMUTATIONS


@pytest.mark.unit
def test_permutations_integer_budget_one_over_cap() -> None:
    size = next(n for n in range(1, 20) if math.factorial(n) > MAX_PERMUTATIONS)
    schema = _object_schema(
        {"tags": _string_enum_field(size, permutations=MAX_PERMUTATIONS + 1)}
    )
    with pytest.raises(SchemaError, match=str(MAX_PERMUTATIONS)):
        compile_json_schema(schema)


@pytest.mark.unit
def test_permutations_rejects_zero_and_negative() -> None:
    for bad in (0, -1):
        schema = _object_schema(
            {"tags": {"type": "string", "enum": ["a", "b"], "permutations": bad}}
        )
        with pytest.raises(SchemaError, match="positive integer or 'all'"):
            compile_json_schema(schema)


@pytest.mark.unit
def test_permutations_rejects_non_all_string() -> None:
    schema = _object_schema(
        {"tags": {"type": "string", "enum": ["a"], "permutations": "two"}}
    )
    with pytest.raises(SchemaError, match="positive integer or 'all'"):
        compile_json_schema(schema)


@pytest.mark.unit
def test_permutations_requires_explicit_enum() -> None:
    schema = _object_schema({"flag": {"type": "boolean", "permutations": 2}})
    with pytest.raises(SchemaError, match="requires an explicit enum"):
        compile_json_schema(schema)


# ---------------------------------------------------------------------------
# 3. ``return_probabilities`` legality
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_return_probabilities_allowed_on_boolean() -> None:
    schema = _object_schema({"flag": {"type": "boolean", "return_probabilities": True}})
    decisions = compile_json_schema(schema)
    assert decisions[0].return_probabilities is True


@pytest.mark.unit
def test_return_probabilities_allowed_on_enum() -> None:
    schema = _object_schema(
        {
            "color": {
                "type": "string",
                "enum": ["red", "blue"],
                "return_probabilities": True,
            }
        }
    )
    decisions = compile_json_schema(schema)
    assert decisions[0].return_probabilities is True


@pytest.mark.unit
def test_return_probabilities_rejected_on_open_text() -> None:
    schema = _object_schema({"note": {"type": "string", "return_probabilities": True}})
    with pytest.raises(SchemaError, match="only supported for enum or boolean"):
        compile_json_schema(schema)


@pytest.mark.unit
def test_return_probabilities_rejected_on_open_numeric() -> None:
    schema = _object_schema(
        {"n": {"type": "integer", "minimum": 0, "return_probabilities": True}}
    )
    with pytest.raises(SchemaError, match="only supported for enum or boolean"):
        compile_json_schema(schema)


@pytest.mark.unit
def test_return_probabilities_must_be_boolean() -> None:
    schema = _object_schema(
        {"flag": {"type": "boolean", "return_probabilities": "yes"}}
    )
    with pytest.raises(SchemaError, match="must be a boolean"):
        compile_json_schema(schema)


@pytest.mark.unit
def test_return_probabilities_false_on_open_text_still_rejected_when_present() -> None:
    schema = _object_schema({"note": {"type": "string", "return_probabilities": False}})
    with pytest.raises(SchemaError, match="only supported for enum or boolean"):
        compile_json_schema(schema)


# ---------------------------------------------------------------------------
# 4. Malformed schema inputs (existing compile law)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_compile_rejects_duplicate_required_names() -> None:
    schema = {
        "type": "object",
        "properties": {"a": {"type": "boolean"}},
        "required": ["a", "a"],
        "additionalProperties": False,
    }
    with pytest.raises(SchemaError, match="duplicate field names"):
        compile_json_schema(schema)


@pytest.mark.unit
def test_compile_rejects_non_list_required() -> None:
    schema = {
        "type": "object",
        "properties": {"a": {"type": "boolean"}},
        "required": "a",
        "additionalProperties": False,
    }
    with pytest.raises(SchemaError, match="list of field names"):
        compile_json_schema(schema)


@pytest.mark.unit
def test_compile_rejects_non_object_property() -> None:
    schema = _object_schema({"a": cast(Any, "not-a-mapping")})
    with pytest.raises(SchemaError, match="must be a schema object"):
        compile_json_schema(schema)


@pytest.mark.unit
def test_compile_rejects_empty_property_name() -> None:
    schema = _object_schema({"": {"type": "boolean"}})
    with pytest.raises(SchemaError, match="non-empty strings"):
        compile_json_schema(schema)


@pytest.mark.unit
def test_compile_rejects_duplicate_enum_values() -> None:
    schema = _object_schema({"x": {"type": "string", "enum": ["a", "a"]}})
    with pytest.raises(SchemaError, match="duplicate values"):
        compile_json_schema(schema)


@pytest.mark.unit
def test_compile_rejects_empty_enum_list() -> None:
    schema = _object_schema({"x": {"type": "string", "enum": []}})
    with pytest.raises(SchemaError, match="non-empty list"):
        compile_json_schema(schema)


@pytest.mark.unit
def test_compile_rejects_negative_max_length() -> None:
    schema = _object_schema({"note": {"type": "string", "maxLength": -1}})
    with pytest.raises(SchemaError, match="non-negative integer"):
        compile_json_schema(schema)


@pytest.mark.unit
def test_compile_rejects_unsupported_text_keyword() -> None:
    schema = _object_schema({"note": {"type": "string", "pattern": "^x$"}})
    with pytest.raises(SchemaError, match="not supported for text fields"):
        compile_json_schema(schema)


@pytest.mark.unit
def test_compile_rejects_x_score_legacy_keyword() -> None:
    schema = _object_schema({"x": {"type": "integer", "enum": [1, 2], "x-score": True}})
    with pytest.raises(SchemaError, match="x-score"):
        compile_json_schema(schema)
