"""Model-agnostic categorical field instruction rendering (#148).

A native Choice lists controls as ``<i> → <label>`` (#207). Noul and Score
keep the ``Control <i> → <label>`` form. A native Choice also keeps the
``Control`` word when a label collides with the active controls (#237, #287).
A caller ``option_block`` template sets the words around the control lines
(#364); ``typevet.domain.text_parts`` holds its rules.

Examples:
    ```python
    from typevet.domain.decisions import Decision
    from typevet.domain.field_instructions import render_field_instructions

    decision = Decision("label", "Pick one.", ("a", "b"), syntax="Choice")
    block = render_field_instructions(decision)
    ```

See Also:
    - [typevet.runtime.scoring_prefix][]: ChatML prefix composition
    - [typevet.domain.text_parts][]: Option block template rules
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from typevet.domain.decisions import Decision
from typevet.domain.judgment_normalize import control_binding_pairs
from typevet.domain.text_parts import (
    DEFAULT_OPTION_BLOCK,
    gold_reference_markers,
    split_option_block,
    validate_option_block,
)

__all__ = [
    "choice_criteria_from_schema",
    "gold_reference_markers",
    "render_field_instructions",
]


def _choice_label(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _is_native_choice(decision: Decision) -> bool:
    # Score decisions also use the Choice syntax but carry int levels.
    return decision.syntax == "Choice" and not any(
        isinstance(choice, int) for choice in decision.choices
    )


def _label_collides(label: str, controls: frozenset[str]) -> bool:
    # A digit-string label ("10") or a label equal to an active control
    # ("A", or "a" beside letter controls) reads like a control.
    text = label.strip()
    return text.isdigit() or text.upper() in controls


def _control_prefix(decision: Decision, pairs: Sequence[tuple[str, str]]) -> str:
    if not _is_native_choice(decision):
        return "Control "
    controls = frozenset(control for control, _ in pairs)
    if any(_label_collides(label, controls) for _, label in pairs):
        return "Control "
    return ""


def _answer_instruction(pairs: Sequence[tuple[str, str]]) -> str:
    # Ten or fewer controls are digits; keep that wording byte-identical.
    shown = "digit" if all(c.isdigit() for c, _ in pairs) else "digit or letter"
    return (
        f"Answer with exactly one control string (the {shown} shown), "
        "not the original label text."
    )


def _render_option_block(
    template: str,
    decision: Decision,
    pairs: Sequence[tuple[str, str]],
    criteria: Mapping[str, str],
) -> str:
    head, line, tail = split_option_block(template)
    prefix = _control_prefix(decision, pairs)
    values = {
        "name": decision.name,
        "question": decision.question,
        "answer_rule": _answer_instruction(pairs),
    }
    lines = [] if head is None else [head.format(**values)]
    for control, label in pairs:
        description = criteria.get(label) or criteria.get(label.lower())
        lines.append(
            line.format(
                control=f"{prefix}{control}",
                label=label,
                description=f": {description}" if description else "",
            )
        )
    if tail is not None:
        lines.append(tail.format(**values))
    return "\n".join(lines)


def render_field_instructions(
    decision: Decision,
    *,
    choice_criteria: Mapping[str, str] | None = None,
    original_labels: Sequence[str] | None = None,
    option_block: str | None = None,
) -> str:
    """Render model-facing instructions for one categorical ``Decision``.

    Args:
        decision: Compiled Choice or Bool field.
        choice_criteria: Optional label-to-description lines for choices.
        original_labels: When set, list ordinal control mappings aligned with
            ``bind_control_candidates``. A native Choice writes
            ``<i> → <label>`` unless a label collides with the active
            controls. Noul and Score write ``Control <i> → <label>``.
        option_block: Option block template used with ``original_labels``,
            or ``None`` for ``DEFAULT_OPTION_BLOCK`` (#364).

    Returns:
        Plain-text field block without user context or template wrappers.

    Raises:
        JudgmentValidationError: ``option_block`` breaks an option block
            rule; the message never holds the template text.
    """
    criteria = choice_criteria or {}
    if original_labels is not None:
        if option_block is None:
            option_block = DEFAULT_OPTION_BLOCK
        else:
            validate_option_block(option_block)
        pairs = control_binding_pairs(original_labels)
        return _render_option_block(option_block, decision, pairs, criteria)
    lines: list[str] = [f"{decision.name}: {decision.question}", "", "Options:"]
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
