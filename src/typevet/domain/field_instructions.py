"""Model-agnostic categorical field instruction rendering (#148).

A native Choice lists controls as ``<i> → <label>`` (#207). Noul and Score
keep the ``Control <i> → <label>`` form.

Examples:
    ```python
    from typevet.domain.decisions import Decision
    from typevet.domain.field_instructions import render_field_instructions

    decision = Decision("label", "Pick one.", ("a", "b"), syntax="Choice")
    block = render_field_instructions(decision)
    ```

See Also:
    - [typevet.runtime.scoring_prefix][]: ChatML prefix composition
    - [typevet.field_prompt][]: Root compatibility shim
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from typevet.domain.decisions import Decision
from typevet.domain.judgment_normalize import control_binding_pairs

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


def _is_native_choice(decision: Decision) -> bool:
    # Score decisions also use the Choice syntax but carry int levels.
    return decision.syntax == "Choice" and not any(
        isinstance(choice, int) for choice in decision.choices
    )


_ANSWER_WITH_CONTROL_INSTRUCTION = (
    "Answer with exactly one control string (the digit shown), "
    "not the original label text."
)


def render_field_instructions(
    decision: Decision,
    *,
    choice_criteria: Mapping[str, str] | None = None,
    original_labels: Sequence[str] | None = None,
) -> str:
    """Render model-facing instructions for one categorical ``Decision``.

    Args:
        decision: Compiled Choice or Bool field.
        choice_criteria: Optional label-to-description lines for choices.
        original_labels: When set, list ordinal control mappings aligned with
            ``bind_control_candidates``. A native Choice writes
            ``<i> → <label>``. Noul and Score write ``Control <i> → <label>``.

    Returns:
        Plain-text field block without user context or template wrappers.
    """
    lines: list[str] = [f"{decision.name}: {decision.question}", "", "Options:"]
    criteria = choice_criteria or {}
    if original_labels is not None:
        prefix = "" if _is_native_choice(decision) else "Control "
        for control, label in control_binding_pairs(original_labels):
            description = criteria.get(label) or criteria.get(label.lower())
            if description:
                lines.append(f"{prefix}{control} → {label}: {description}")
            else:
                lines.append(f"{prefix}{control} → {label}")
        lines.extend(["", _ANSWER_WITH_CONTROL_INSTRUCTION])
        return "\n".join(lines)
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
