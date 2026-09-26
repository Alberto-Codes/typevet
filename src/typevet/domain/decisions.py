"""TypeLLM-shaped decisions and dependency ordering for compiled schemas.

Examples:
    ```python
    from typevet.domain import Decision, compile_json_schema

    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
    }
    assert compile_json_schema(schema)[0].syntax == "Bool"
    ```

See Also:
    - [typevet.domain.decision_compile][]: JSON Schema compilation implementation
    - [typevet.domain.models][]: Generation request carrying a schema mapping
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

MAX_ENUM_CHOICES = 24
MAX_PERMUTATIONS = 720


class SchemaError(ValueError):
    """The schema mapping is outside the supported compile subset.

    Examples:
        ```python
        from typevet.domain import SchemaError, compile_json_schema

        try:
            compile_json_schema({"type": "array"})
        except SchemaError:
            pass
        ```
    """


@dataclass(frozen=True)
class Decision:
    """One tokenizer-independent field compiled from JSON Schema.

    Attributes:
        name (str): Property name in the output object.
        question (str): Model-facing prompt for this field.
        choices (tuple[Any, ...]): Closed choices; empty for open numeric or text.
        syntax (str): TypeLLM syntax label (Choice, Bool, Integer, Number, Text).
        numeric_type (str | None): ``integer`` or ``number`` when syntax is numeric.
        minimum (int | float | None): Lower bound for open numeric fields.
        maximum (int | float | None): Upper bound for open numeric fields.
        text_type (bool): True when the field is open-ended text.
        max_length (int | None): Optional max string length for text fields.
        permutations (int | str): Enum permutation budget or ``all``.
        return_probabilities (bool): Whether to request choice probabilities.
        depends_on (tuple[str, ...] | None): Prior fields that must be set first.
        nullable (bool): Whether JSON null is allowed for this field.

    Examples:
        ```python
        from typevet.domain.decisions import Decision

        Decision("x", "Pick one.", ("a", "b"), syntax="Choice")
        ```
    """

    name: str
    question: str
    choices: tuple[Any, ...]
    syntax: str = "Choice"
    numeric_type: str | None = None
    minimum: int | float | None = None
    maximum: int | float | None = None
    text_type: bool = False
    max_length: int | None = None
    permutations: int | str = 1
    return_probabilities: bool = False
    depends_on: tuple[str, ...] | None = None
    nullable: bool = False


def dependency_layers(decisions: Sequence[Decision]) -> list[list[Decision]]:
    """Return stable topological layers and validate dependencies.

    Args:
        decisions: Compiled decisions, typically from ``compile_json_schema``.

    Returns:
        Lists of decisions that may run in parallel within each layer.

    Raises:
        SchemaError: For unknown dependencies, self-deps, or cycles.

    Examples:
        ```python
        from typevet.domain.decisions import Decision, dependency_layers

        a = Decision("a", "q", ())
        b = Decision("b", "q", (), depends_on=("a",))
        dependency_layers([b, a])
        ```
    """
    names = {decision.name for decision in decisions}
    for decision in decisions:
        for dependency in decision.depends_on or ():
            if dependency not in names:
                raise SchemaError(
                    f"unknown dependency {dependency!r} for {decision.name!r}"
                )
            if dependency == decision.name:
                raise SchemaError(f"field {decision.name!r} cannot depend on itself")
    remaining = list(decisions)
    completed: set[str] = set()
    layers: list[list[Decision]] = []
    while remaining:
        layer = [d for d in remaining if set(d.depends_on or ()) <= completed]
        if not layer:
            raise SchemaError(
                f"dependency cycle among fields: {[d.name for d in remaining]!r}"
            )
        layers.append(layer)
        completed.update(d.name for d in layer)
        remaining = [d for d in remaining if d.name not in completed]
    return layers
