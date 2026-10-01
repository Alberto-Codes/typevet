"""Caller-substitutable text parts of a judgment prompt (#364).

Two parts of the scoring prefix are templates with named placeholders. The
``option_block`` template renders the field block: the question, one line
per option and the answer rule. The ``context_template`` template places
the state context and the field block in the user text. Each default equals
the rendering before #364, so an unset part changes no byte.

The control-token rule and the no-thinking prefill are not text parts. A
template cannot change which control maps to which label: every
``<control> → <label>`` pair must survive rendering, in order.

A template is refused when it lacks a required placeholder, repeats a
placeholder, names an unknown placeholder, uses a conversion or a format
spec, drops a control line, holds a gold-reference marker, holds a media
marker or holds a chat turn marker. An ``option_block`` also needs
``{answer_rule}`` (#373). The error names the part and the rule, never the
template text. The rule guards the templates only: caller values (state,
criteria, labels) are not checked (#373).

Attributes:
    OPTION_BLOCK (str): Component name of the option block part.
    CONTEXT_TEMPLATE (str): Component name of the context template part.
    DEFAULT_OPTION_BLOCK (str): Option block template typevet uses when the
        part is unset.
    DEFAULT_CONTEXT_TEMPLATE (str): Context template typevet uses when the
        part is unset.
    DEFAULT_PART (str): Receipt value of a part that is unset.
    TURN_MARKERS (tuple[str, ...]): ChatML, Gemma 3 and Gemma 4 turn markers,
        and the Gemma 4 channel, think and tool-response tokens, that no
        template may hold (#373).

Examples:
    ```python
    from typevet.domain.text_parts import TextParts

    parts = TextParts(context_template="Text: {context} | {field_block}")
    parts.receipt()["option_block"]  # "default"
    ```

See Also:
    - [typevet.domain.field_instructions][]: Renders the option block
    - [typevet.domain.judgment_normalize][]: Control-token rule
"""

from __future__ import annotations

import hashlib
import string
from dataclasses import dataclass
from typing import Final, NoReturn

from typevet.domain.errors import JudgmentValidationError
from typevet.domain.media import MEDIA_MARKER

__all__ = [
    "CONTEXT_TEMPLATE",
    "DEFAULT_CONTEXT_TEMPLATE",
    "DEFAULT_OPTION_BLOCK",
    "DEFAULT_PART",
    "OPTION_BLOCK",
    "TURN_MARKERS",
    "TextParts",
    "gold_reference_markers",
    "render_context",
    "split_option_block",
    "validate_context_template",
    "validate_option_block",
]

OPTION_BLOCK: Final[str] = "option_block"
CONTEXT_TEMPLATE: Final[str] = "context_template"
DEFAULT_PART: Final[str] = "default"
DEFAULT_OPTION_BLOCK: Final[str] = (
    "{name}: {question}\n\nOptions:\n{control} → {label}{description}\n\n{answer_rule}"
)
DEFAULT_CONTEXT_TEMPLATE: Final[str] = "{context}\n\n{field_block}"
# Copies of the served-template turn markers; the domain does not import
# adapters. A unit test keeps them equal to the adapter constants.
TURN_MARKERS: Final[tuple[str, ...]] = (
    "<|im_start|>",
    "<|" + "im_end|>",
    "<start_of_turn>",
    "<end_of_turn>",
    "<|turn>",
    "<turn|>",
    "<|channel>",
    "<channel|>",
    "<|think|>",
    "<|tool_response>",
)

_LINE_FIELDS: Final[frozenset[str]] = frozenset({"control", "label", "description"})
_BLOCK_FIELDS: Final[frozenset[str]] = frozenset({"name", "question", "answer_rule"})
_OPTION_REQUIRED: Final[tuple[str, ...]] = (
    "question",
    "control",
    "label",
    "description",
    "answer_rule",
)
_CONTEXT_FIELDS: Final[tuple[str, ...]] = ("context", "field_block")
_PROBE_CONTROL: Final[str] = "\x00control"
_PROBE_LABEL: Final[str] = "\x00label"
_PROBE_DESCRIPTION: Final[str] = "\x00description"
_FORMATTER: Final[string.Formatter] = string.Formatter()
_GOLD_REFERENCE_MARKERS: Final[frozenset[str]] = frozenset(
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


def _refuse(part: str, rule: str, placeholder: str | None = None) -> NoReturn:
    # The message holds the part, the rule and a vocabulary name only.
    named = f" ({{{placeholder}}})" if placeholder else ""
    msg = f"{part} template refused: {rule}{named}"
    raise JudgmentValidationError(msg)


def _fields(part: str, text: str) -> list[str]:
    try:
        parsed = list(_FORMATTER.parse(text))
    except ValueError:
        _refuse(part, "malformed template")
    names: list[str] = []
    for _, name, spec, conversion in parsed:
        if name is None:
            continue
        if spec or conversion:
            _refuse(part, "malformed placeholder")
        names.append(name)
    return names


def _check_common(
    part: str, template: object, allowed: frozenset[str], required: tuple[str, ...]
) -> str:
    if not isinstance(template, str):
        _refuse(part, "template is not a string")
    names = _fields(part, template)
    if any(name not in allowed for name in names):
        _refuse(part, "unknown placeholder")
    for name in sorted(allowed):
        if names.count(name) > 1:
            _refuse(part, "repeated placeholder", name)
    for name in required:
        if name not in names:
            _refuse(part, "missing placeholder", name)
    lowered = template.lower()
    if any(marker in lowered for marker in _GOLD_REFERENCE_MARKERS):
        _refuse(part, "holds a gold-reference marker")
    if MEDIA_MARKER in template:
        _refuse(part, "holds a media marker")
    if any(marker in template for marker in TURN_MARKERS):
        _refuse(part, "holds a turn marker")
    return template


def split_option_block(template: str) -> tuple[str | None, str, str | None]:
    """Split an option block template at its option line.

    Args:
        template: Option block template that holds one ``{control}``.

    Returns:
        The lines before the option line (``None`` when there are none), the
        option line, and the lines after it (``None`` when there are none).

    Raises:
        JudgmentValidationError: The template holds no ``{control}`` line.
    """
    lines = template.split("\n")
    for index, line in enumerate(lines):
        if "control" in _fields(OPTION_BLOCK, line):
            head = "\n".join(lines[:index]) if index else None
            tail = "\n".join(lines[index + 1 :]) if index + 1 < len(lines) else None
            return head, line, tail
    _refuse(OPTION_BLOCK, "missing placeholder", "control")


def validate_option_block(template: str) -> None:
    """Refuse an ``option_block`` template that breaks a rule.

    Placeholders: ``{question}``, ``{control}``, ``{label}``,
    ``{description}`` and ``{answer_rule}`` are required; ``{name}`` is
    optional. Each appears at most once. ``{control}``, ``{label}`` and
    ``{description}`` sit on one option line, which repeats once per
    option; the other placeholders sit off it.

    Args:
        template: Option block template text.

    Raises:
        JudgmentValidationError: The template breaks a rule. The message
            names the part and the rule, never the template text.
    """
    allowed = _LINE_FIELDS | _BLOCK_FIELDS
    text = _check_common(OPTION_BLOCK, template, allowed, _OPTION_REQUIRED)
    _, line, _ = split_option_block(text)
    line_fields = set(_fields(OPTION_BLOCK, line))
    for name in sorted(_fields(OPTION_BLOCK, text)):
        if (name in _LINE_FIELDS) != (name in line_fields):
            _refuse(OPTION_BLOCK, "misplaced placeholder", name)
    # Probe with and without a description: neither may split the pair.
    pair = f"{_PROBE_CONTROL} → {_PROBE_LABEL}"
    probes = (
        line.format(control=_PROBE_CONTROL, label=_PROBE_LABEL, description=text)
        for text in ("", _PROBE_DESCRIPTION)
    )
    if any(pair not in probe for probe in probes):
        _refuse(OPTION_BLOCK, "drops a control line")


def validate_context_template(template: str) -> None:
    """Refuse a ``context_template`` template that breaks a rule.

    Placeholders: ``{context}`` and ``{field_block}`` are both required, and
    each appears once.

    Args:
        template: Context template text.

    Raises:
        JudgmentValidationError: The template breaks a rule. The message
            names the part and the rule, never the template text.
    """
    _check_common(
        CONTEXT_TEMPLATE, template, frozenset(_CONTEXT_FIELDS), _CONTEXT_FIELDS
    )


def render_context(template: str | None, *, context: str, field_block: str) -> str:
    """Place the state context and the field block in the user text.

    Args:
        template: Context template, or ``None`` for ``DEFAULT_CONTEXT_TEMPLATE``.
        context: Rendered state context, media markers included.
        field_block: Rendered field instructions.

    Returns:
        User text of the scoring prefix.

    Raises:
        JudgmentValidationError: ``template`` breaks a context template rule.
    """
    if template is None:
        template = DEFAULT_CONTEXT_TEMPLATE
    else:
        validate_context_template(template)
    return template.format(context=context, field_block=field_block)


def _part_digest(template: str | None) -> str:
    if template is None:
        return DEFAULT_PART
    return hashlib.sha256(template.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class TextParts:
    """Caller templates for the option block and the context, validated.

    ``None`` keeps the default template for that part.

    Attributes:
        option_block (str | None): Option block template, or ``None``.
        context_template (str | None): Context template, or ``None``.

    Examples:
        ```python
        TextParts().receipt()  # both parts are "default"
        ```
    """

    option_block: str | None = None
    context_template: str | None = None

    def __post_init__(self) -> None:
        """Refuse a template that breaks a rule of its part.

        Raises:
            JudgmentValidationError: A set template breaks a rule. The message
                names the part and the rule, never the template text.
        """
        if self.option_block is not None:
            validate_option_block(self.option_block)
        if self.context_template is not None:
            validate_context_template(self.context_template)

    def receipt(self) -> dict[str, str]:
        """Return the ``text_parts`` receipt block.

        Returns:
            One entry per part: the SHA-256 hex digest of the UTF-8 template
            text, or ``"default"`` when the part is unset.
        """
        return {
            OPTION_BLOCK: _part_digest(self.option_block),
            CONTEXT_TEMPLATE: _part_digest(self.context_template),
        }
