"""Compile JSON Schema mappings into [Decision][] values.

Examples:
    ```python
    from typevet.domain.decision_compile import compile_json_schema

    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
    }
    assert compile_json_schema(schema)[0].syntax == "Bool"
    ```

See Also:
    - [typevet.domain.decisions][]: ``Decision``, ``SchemaError``, and layers
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from typevet.domain.decisions import (
    MAX_ENUM_CHOICES,
    MAX_PERMUTATIONS,
    Decision,
    SchemaError,
    dependency_layers,
)

_NULLABLE_TYPE_ARITY = 2


def _has_duplicates(values: Sequence[Any]) -> bool:
    for index, value in enumerate(values):
        for previous in values[:index]:
            both_numbers = type(value) in {int, float} and type(previous) in {
                int,
                float,
            }
            if (both_numbers and value == previous) or (
                type(value) is type(previous) and value == previous
            ):
                return True
    return False


def _is_finite_number(value: Any) -> bool:
    return type(value) is int or (type(value) is float and math.isfinite(value))


def _validated_properties(schema: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(schema, Mapping):
        raise SchemaError("schema must be a mapping")
    if schema.get("type") != "object":
        raise SchemaError("root JSON Schema type must be 'object'")
    properties = schema.get("properties")
    if not isinstance(properties, Mapping) or not properties:
        raise SchemaError("JSON Schema properties must be a non-empty object")

    required = schema.get("required", [])
    if not isinstance(required, list) or any(not isinstance(x, str) for x in required):
        raise SchemaError("JSON Schema required must be a list of field names")
    if len(set(required)) != len(required):
        raise SchemaError("JSON Schema required contains duplicate field names")
    unknown_required = [name for name in required if name not in properties]
    if unknown_required:
        raise SchemaError(
            f"required fields are missing from properties: {unknown_required!r}"
        )
    return properties


def _field_question(name: str, field: Mapping[str, Any]) -> str:
    instructions = field.get("instructions")
    if "instructions" in field and not isinstance(instructions, str):
        raise SchemaError(f"instructions for {name!r} must be a string")
    for old_key in ("question", "x-question"):
        if old_key in field:
            raise SchemaError(
                f"{old_key} for {name!r} is no longer supported; use instructions"
            )
    description = field.get("description")
    if "description" in field and not isinstance(description, str):
        raise SchemaError(f"description for {name!r} must be a string")
    if instructions is not None:
        return instructions
    if description is not None:
        return description
    return f'Choose the value for "{name}".'


def _resolve_field_type(name: str, raw_type: Any) -> tuple[Any, bool]:
    if not isinstance(raw_type, list):
        return raw_type, False
    kinds = [kind for kind in raw_type if kind != "null"]
    if (
        len(raw_type) != _NULLABLE_TYPE_ARITY
        or len(kinds) != 1
        or not isinstance(kinds[0], str)
    ):
        raise SchemaError(f'type for {name!r} must be one type or [type, "null"]')
    return kinds[0], True


def _validate_permutations(
    name: str, enum: Any, permutations: int | str, field: Mapping[str, Any]
) -> None:
    if "permutations" not in field:
        return
    if enum is None:
        raise SchemaError(f"permutations for {name!r} requires an explicit enum")
    valid = permutations == "all" or (type(permutations) is int and permutations > 0)
    if not valid:
        raise SchemaError(
            f"permutations for {name!r} must be a positive integer or 'all'"
        )
    if isinstance(enum, list):
        count = math.factorial(len(enum))
        budget = count if permutations == "all" else min(permutations, count)
        if budget > MAX_PERMUTATIONS:
            raise SchemaError(
                f"permutations for {name!r} exceeds {MAX_PERMUTATIONS}; "
                "use a smaller integer budget"
            )


def _validate_max_length(name: str, field: Mapping[str, Any]) -> int | None:
    max_length = field.get("maxLength")
    if "maxLength" in field and (type(max_length) is not int or max_length < 0):
        raise SchemaError(f"maxLength for {name!r} must be a non-negative integer")
    return max_length


def _reject_legacy_keywords(name: str, field: Mapping[str, Any]) -> None:
    if "x-score" in field:
        raise SchemaError(f"x-score for {name!r} is not supported; use a number enum")
    if "x-other" in field:
        raise SchemaError(f"x-other for {name!r} is not supported; use a closed enum")


def _return_probabilities(
    name: str, field: Mapping[str, Any], field_type: Any, enum: Any
) -> bool:
    return_probabilities = field.get("return_probabilities", False)
    if type(return_probabilities) is not bool:
        raise SchemaError(f"return_probabilities for {name!r} must be a boolean")
    if "return_probabilities" in field and field_type != "boolean" and enum is None:
        raise SchemaError(
            f"return_probabilities for {name!r} is only supported for "
            "enum or boolean fields"
        )
    return return_probabilities


def _decision_for_open_text(
    name: str,
    question: str,
    field: Mapping[str, Any],
    field_type: Any,
    enum: Any,
    max_length: int | None,
    nullable: bool,
) -> Decision | None:
    if field_type != "string" or enum is not None:
        return None
    for keyword in ("minLength", "pattern", "format"):
        if keyword in field:
            raise SchemaError(f"{keyword} is not supported for text fields")
    return Decision(
        name,
        question,
        (),
        "Text",
        text_type=True,
        max_length=max_length,
        nullable=nullable,
    )


def _decision_for_open_numeric(
    name: str,
    question: str,
    field_type: Any,
    field: Mapping[str, Any],
    enum: Any,
    nullable: bool,
) -> Decision | None:
    if field_type not in {"integer", "number"} or enum is not None:
        return None
    minimum = field.get("minimum")
    maximum = field.get("maximum")
    for keyword, bound in (("minimum", minimum), ("maximum", maximum)):
        if bound is not None and not _is_finite_number(bound):
            raise SchemaError(f"{keyword} for {name!r} must be a finite number")
    if minimum is not None and maximum is not None and minimum > maximum:
        raise SchemaError(f"minimum for {name!r} must not exceed maximum")
    return Decision(
        name=name,
        question=question,
        choices=(),
        syntax="Integer" if field_type == "integer" else "Number",
        numeric_type=field_type,
        minimum=minimum,
        maximum=maximum,
        nullable=nullable,
    )


def _enum_values_and_syntax(
    name: str,
    field_type: Any,
    enum: Any,
    nullable: bool,
    max_length: int | None,
) -> tuple[list[Any], str]:
    if field_type == "boolean":
        values = ([True, False] + [None] * nullable) if enum is None else enum
        if not isinstance(values, list) or not values:
            raise SchemaError(f"enum for {name!r} must be a non-empty list")
        if any(
            type(value) is not bool and not (nullable and value is None)
            for value in values
        ):
            raise SchemaError(f"boolean enum for {name!r} may contain only booleans")
        return values, "Bool"

    if field_type not in {"string", "integer", "number"}:
        raise NotImplementedError(
            f"property {name!r} has unsupported JSON Schema type {field_type!r}"
        )
    if enum is None:
        raise NotImplementedError(
            f"property {name!r} has type {field_type!r} without a finite enum"
        )
    if not isinstance(enum, list) or not enum:
        raise SchemaError(f"enum for {name!r} must be a non-empty list")
    if len(enum) > MAX_ENUM_CHOICES:
        raise SchemaError(
            f"enum for {name!r} has {len(enum)} values; "
            f"the maximum is {MAX_ENUM_CHOICES}"
        )
    typed = [value for value in enum if not (nullable and value is None)]
    if field_type == "string":
        valid = all(isinstance(value, str) for value in typed)
    elif field_type == "integer":
        valid = all(type(value) is int for value in typed)
    else:
        valid = all(_is_finite_number(value) for value in typed)
    if not valid:
        raise SchemaError(f"enum values for {name!r} do not match type {field_type!r}")
    if (
        field_type == "string"
        and max_length is not None
        and any(len(v) > max_length for v in typed)
    ):
        raise SchemaError(f"enum values for {name!r} exceed maxLength")
    return enum, "Choice"


def _compile_property(name: str, field: Mapping[str, Any]) -> Decision:
    if not isinstance(name, str) or not name:
        raise SchemaError("property names must be non-empty strings")
    if not isinstance(field, Mapping):
        raise SchemaError(f"property {name!r} must be a schema object")

    question = _field_question(name, field)
    field_type, nullable = _resolve_field_type(name, field.get("type"))
    enum = field.get("enum")
    permutations = field.get("permutations", 1)
    _validate_permutations(name, enum, permutations, field)
    return_probabilities = _return_probabilities(name, field, field_type, enum)
    _reject_legacy_keywords(name, field)
    max_length = _validate_max_length(name, field)

    text = _decision_for_open_text(
        name, question, field, field_type, enum, max_length, nullable
    )
    if text is not None:
        return text

    numeric = _decision_for_open_numeric(
        name, question, field_type, field, enum, nullable
    )
    if numeric is not None:
        return numeric

    values, syntax = _enum_values_and_syntax(
        name, field_type, enum, nullable, max_length
    )
    if len(values) > MAX_ENUM_CHOICES:
        raise SchemaError(
            f"enum for {name!r} has {len(values)} values; "
            f"the maximum is {MAX_ENUM_CHOICES}"
        )
    if _has_duplicates(values):
        raise SchemaError(f"enum for {name!r} contains duplicate values")
    return Decision(
        name,
        question,
        tuple(values),
        syntax,
        return_probabilities=return_probabilities,
        permutations=permutations,
        nullable=nullable,
    )


def _attach_depends_on(
    decisions: Sequence[Decision], properties: Mapping[str, Any]
) -> list[Decision]:
    compiled: list[Decision] = []
    for decision in decisions:
        field = properties[decision.name]
        dependencies = field.get("depends_on")
        if "depends_on" in field:
            if not isinstance(dependencies, list) or any(
                not isinstance(dep_name, str) or not dep_name
                for dep_name in dependencies
            ):
                raise SchemaError(
                    f"depends_on for {decision.name!r} must be a list of field names"
                )
            if len(set(dependencies)) != len(dependencies):
                raise SchemaError(
                    f"depends_on for {decision.name!r} contains duplicates"
                )
            dependencies = tuple(dependencies)
        compiled.append(replace(decision, depends_on=dependencies))
    return compiled


def compile_json_schema(schema: Mapping[str, Any]) -> list[Decision]:
    """Compile an ordered JSON Schema object into TypeLLM decisions.

    Args:
        schema: Root object schema with ``type``, ``properties``, and optional
            ``required``.

    Returns:
        Decisions in property declaration order, with ``depends_on`` resolved
        and dependency layers validated.

    Raises:
        SchemaError: When the schema is invalid or unsupported.
        NotImplementedError: For unsupported types without a finite enum.

    Examples:
        ```python
        from typevet.domain.decision_compile import compile_json_schema

        compile_json_schema(
            {
                "type": "object",
                "properties": {"n": {"type": "integer", "enum": [1, 2]}},
                "required": ["n"],
            }
        )
        ```
    """
    properties = _validated_properties(schema)
    decisions = [_compile_property(name, field) for name, field in properties.items()]
    compiled = _attach_depends_on(decisions, properties)
    dependency_layers(compiled)
    return compiled
