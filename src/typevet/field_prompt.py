"""Pure field-instruction rendering and ChatML scoring-prefix composition (#103).

Examples:
    ```python
    from typevet.domain.decisions import Decision
    from typevet.field_prompt import compose_scoring_prefix, render_field_instructions

    decision = Decision("label", "Pick one.", ("a", "b"), syntax="Choice")
    block = render_field_instructions(decision)
    prefix = compose_scoring_prefix(context="Task text.", field_block=block)
    ```

See Also:
    - [typevet.decide_categorical][]: Native vs injected scoring prefix
    - [typevet.gemma_served_template][]: ChatML assistant header constant
"""

from __future__ import annotations

from collections.abc import Mapping

from typevet.domain.decisions import Decision
from typevet.gemma_served_template import (
    CHATML_ASSISTANT_HEADER,
    CHATML_IM_END,
    CHATML_IM_START,
)

_GOLD_REFERENCE_MARKERS: frozenset[str] = frozenset(
    {
        "gold answer",
        "reference answer",
        "correct answer",
        "expected answer",
    }
)


def gold_reference_markers() -> frozenset[str]:
    """Return lowercase substrings that must not appear in rendered field blocks.

    Returns:
        Forbidden reference-answer marker phrases for regression tests.
    """
    return _GOLD_REFERENCE_MARKERS


def _choice_label(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def render_field_instructions(
    decision: Decision,
    *,
    choice_criteria: Mapping[str, str] | None = None,
) -> str:
    """Render model-facing instructions for one categorical ``Decision``.

    Args:
        decision: Compiled Choice or Bool field.
        choice_criteria: Optional label-to-description lines for choices.

    Returns:
        Plain-text field block without user context or template wrappers.
    """
    lines: list[str] = [f"{decision.name}: {decision.question}", "", "Options:"]
    criteria = choice_criteria or {}
    for choice in decision.choices:
        if choice is None:
            continue
        label = _choice_label(choice)
        description = criteria.get(label) or criteria.get(str(choice))
        if description:
            lines.append(f"- {label}: {description}")
        else:
            lines.append(f"- {label}")
    return "\n".join(lines)


def compose_scoring_prefix(*, context: str, field_block: str) -> str:
    """Compose degraded ChatML text ending at the assistant answer boundary.

    Args:
        context: Caller-owned user or task text for the judgment.
        field_block: Rendered field instructions from ``render_field_instructions``.

    Returns:
        Prefix string ending with ``CHATML_ASSISTANT_HEADER``.
    """
    return (
        f"{CHATML_IM_START}user\n{context}\n\n{field_block}{CHATML_IM_END}\n"
        f"{CHATML_ASSISTANT_HEADER}"
    )


def choice_criteria_from_schema(
    schema: Mapping[str, object],
    field_name: str,
) -> Mapping[str, str] | None:
    """Extract string choice descriptions from a compiled property schema.

    Args:
        schema: Root object schema passed to ``decide_categorical``.
        field_name: Compiled property name.

    Returns:
        Label-to-description map when ``criteria`` is present on the property,
        otherwise ``None``.
    """
    properties = schema.get("properties")
    if not isinstance(properties, Mapping):
        return None
    field = properties.get(field_name)
    if not isinstance(field, Mapping):
        return None
    raw = field.get("criteria")
    if not isinstance(raw, Mapping):
        return None
    out = {
        key: value
        for key, value in raw.items()
        if isinstance(key, str) and isinstance(value, str) and value.strip()
    }
    return out or None
