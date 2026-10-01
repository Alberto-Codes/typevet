"""Unit tests: the option block and the context template are text parts (#364).

Examples:
    ```bash
    uv run pytest -q tests/unit/test_text_parts.py
    ```

See Also:
    - [typevet.domain.text_parts][]: Templates, validator and receipt
    - [typevet.domain.field_instructions][]: Field block renderer
"""

from __future__ import annotations

import hashlib
from typing import cast

import pytest

from typevet.adapters.outbound.gemma.served_template import (
    CHATML_IM_END,
    CHATML_IM_START,
    GEMMA3_END_OF_TURN,
    GEMMA3_START_OF_TURN,
    GEMMA4_CHANNEL_CLOSE,
    GEMMA4_NO_THINKING_PREFILL,
    GEMMA4_THINK_TRIGGER,
    GEMMA4_TOOL_RESPONSE,
    GEMMA4_TURN_CLOSE,
    GEMMA4_TURN_OPEN,
)
from typevet.domain.decisions import Decision
from typevet.domain.errors import JudgmentValidationError
from typevet.domain.field_instructions import render_field_instructions
from typevet.domain.text_parts import (
    CONTEXT_TEMPLATE,
    DEFAULT_CONTEXT_TEMPLATE,
    DEFAULT_OPTION_BLOCK,
    DEFAULT_PART,
    OPTION_BLOCK,
    TURN_MARKERS,
    TextParts,
    render_context,
    validate_context_template,
    validate_option_block,
)

pytestmark = pytest.mark.unit

_SENTINEL = "ZEBRA-SENTINEL"
_LINE = "{control} → {label}{description}"

_RULE = "\n{answer_rule}"
# served_template.py has no constant for the channel open marker; the
# no-thinking prefill starts with it (#373).
_CHANNEL_OPEN = GEMMA4_NO_THINKING_PREFILL.removesuffix(
    f"thought\n{GEMMA4_CHANNEL_CLOSE}"
)
_NEW_MARKERS = (
    _CHANNEL_OPEN,
    GEMMA4_CHANNEL_CLOSE,
    GEMMA4_THINK_TRIGGER,
    GEMMA4_TOOL_RESPONSE,
)

_BAD_OPTION_BLOCKS: list[tuple[str, str, str]] = [
    ("missing", f"{_SENTINEL}\n{_LINE}{_RULE}", "missing placeholder"),
    ("missing_control", f"{_SENTINEL} {{question}}{_RULE}", "missing placeholder"),
    (
        "missing_answer_rule",
        f"{_SENTINEL} {{question}}\n{_LINE}",
        "missing placeholder ({answer_rule})",
    ),
    (
        "repeated",
        f"{_SENTINEL} {{question}} {{question}}\n{_LINE}{_RULE}",
        "repeated placeholder",
    ),
    (
        "drops_control_line",
        f"{_SENTINEL} {{question}}\n{{label}} is {{control}}{{description}}{_RULE}",
        "drops a control line",
    ),
    (
        "gold_marker",
        f"{_SENTINEL} {{question}} Match the Gold Answer.\n{_LINE}{_RULE}",
        "gold-reference marker",
    ),
    (
        "description_before_arrow",
        f"{_SENTINEL} {{question}}\n{{control}}{{description}} → {{label}}{_RULE}",
        "drops a control line",
    ),
    (
        "description_after_arrow",
        f"{_SENTINEL} {{question}}\n{{control}} →{{description}} {{label}}{_RULE}",
        "drops a control line",
    ),
    (
        "unknown",
        f"{{question}} {{{_SENTINEL}}}\n{_LINE}{_RULE}",
        "unknown placeholder",
    ),
    ("conversion", f"{_SENTINEL} {{question!r}}\n{_LINE}{_RULE}", "malformed"),
    ("format_spec", f"{_SENTINEL} {{question:>9}}\n{_LINE}{_RULE}", "malformed"),
    ("unbalanced", _SENTINEL + " {question} }\n" + _LINE + _RULE, "malformed"),
    (
        "misplaced",
        f"{_SENTINEL} {{question}}\n{_LINE} {{answer_rule}}",
        "misplaced placeholder",
    ),
    (
        "media_marker",
        f"{_SENTINEL} <__media__> {{question}}\n{_LINE}{_RULE}",
        "media marker",
    ),
    *(
        (
            f"turn_marker_{index}",
            f"{_SENTINEL} {marker} {{question}}\n{_LINE}{_RULE}",
            "turn marker",
        )
        for index, marker in enumerate(TURN_MARKERS)
    ),
    *(
        (
            f"control_marker_{index}",
            f"{_SENTINEL} {{question}} {marker}\n{_LINE}{_RULE}",
            "turn marker",
        )
        for index, marker in enumerate(_NEW_MARKERS)
    ),
]

_BAD_CONTEXT_TEMPLATES: list[tuple[str, str, str]] = [
    ("missing", f"{_SENTINEL} {{context}}", "missing placeholder"),
    (
        "repeated",
        f"{_SENTINEL} {{context}} {{context}}\n{{field_block}}",
        "repeated placeholder",
    ),
    (
        "gold_marker",
        f"{_SENTINEL} the reference answer\n{{context}}\n{{field_block}}",
        "gold-reference marker",
    ),
    (
        "unknown",
        f"{{context}} {{{_SENTINEL}}} {{field_block}}",
        "unknown placeholder",
    ),
    (
        "media_marker",
        f"<__media__> {_SENTINEL} {{context}}{{field_block}}",
        "media marker",
    ),
    *(
        (
            f"turn_marker_{index}",
            f"{_SENTINEL} {{context}}\n{marker}{{field_block}}",
            "turn marker",
        )
        for index, marker in enumerate(TURN_MARKERS)
    ),
    *(
        (
            f"control_marker_{index}",
            f"{_SENTINEL} {marker}{{context}}\n{{field_block}}",
            "turn marker",
        )
        for index, marker in enumerate(_NEW_MARKERS)
    ),
]


def _assert_value_free(message: str, template: str, part: str, rule: str) -> None:
    assert _SENTINEL not in message
    assert template not in message
    assert "gold answer" not in message.lower()
    assert "reference answer" not in message.lower()
    assert not any(marker in message for marker in TURN_MARKERS)
    assert part in message
    assert rule in message


@pytest.mark.parametrize(
    ("template", "rule"),
    [(t, r) for _, t, r in _BAD_OPTION_BLOCKS],
    ids=[i for i, _, _ in _BAD_OPTION_BLOCKS],
)
def test_bad_option_block_is_refused_without_template_text(
    template: str, rule: str
) -> None:
    with pytest.raises(JudgmentValidationError) as caught:
        validate_option_block(template)
    _assert_value_free(str(caught.value), template, OPTION_BLOCK, rule)
    with pytest.raises(JudgmentValidationError):
        TextParts(option_block=template)


@pytest.mark.parametrize(
    ("template", "rule"),
    [(t, r) for _, t, r in _BAD_CONTEXT_TEMPLATES],
    ids=[i for i, _, _ in _BAD_CONTEXT_TEMPLATES],
)
def test_bad_context_template_is_refused_without_template_text(
    template: str, rule: str
) -> None:
    with pytest.raises(JudgmentValidationError) as caught:
        validate_context_template(template)
    _assert_value_free(str(caught.value), template, CONTEXT_TEMPLATE, rule)
    with pytest.raises(JudgmentValidationError):
        TextParts(context_template=template)


def test_domain_turn_markers_equal_the_served_template_constants() -> None:
    assert TURN_MARKERS == (
        CHATML_IM_START,
        CHATML_IM_END,
        GEMMA3_START_OF_TURN,
        GEMMA3_END_OF_TURN,
        GEMMA4_TURN_OPEN,
        GEMMA4_TURN_CLOSE,
        _CHANNEL_OPEN,
        GEMMA4_CHANNEL_CLOSE,
        GEMMA4_THINK_TRIGGER,
        GEMMA4_TOOL_RESPONSE,
    )
    assert _CHANNEL_OPEN == "<|channel>"


def test_a_non_string_template_is_refused_by_part_name() -> None:
    with pytest.raises(JudgmentValidationError, match=OPTION_BLOCK):
        TextParts(option_block=cast("str", 42))


def test_default_templates_are_valid_and_equal_today_rendering() -> None:
    validate_option_block(DEFAULT_OPTION_BLOCK)
    validate_context_template(DEFAULT_CONTEXT_TEMPLATE)
    assert render_context(DEFAULT_CONTEXT_TEMPLATE, context="c", field_block="f") == (
        "c\n\nf"
    )
    decision = Decision("noul", "Billing issue?", (False, True), syntax="Bool")
    block = render_field_instructions(
        decision,
        choice_criteria={"true": "Yes"},
        original_labels=("false", "true"),
        option_block=DEFAULT_OPTION_BLOCK,
    )
    assert block == (
        "noul: Billing issue?\n\nOptions:\nControl 0 → false\n"
        "Control 1 → true: Yes\n\nAnswer with exactly one control string "
        "(the digit shown), not the original label text."
    )


def test_substituted_option_block_keeps_every_control_line_in_order() -> None:
    labels = tuple(f"label{i}" for i in range(12))
    decision = Decision("pick", "Which one?", labels, syntax="Choice")
    template = (
        "Q: {question}\nControls:\n* {control} → {label}{description} *\n{answer_rule}"
    )
    block = render_field_instructions(
        decision, original_labels=labels, option_block=template
    )
    controls = [*"0123456789", "A", "B"]
    rule = (
        "Answer with exactly one control string (the digit or letter shown), "
        "not the original label text."
    )
    lines = [f"* {c} → {label} *" for c, label in zip(controls, labels, strict=True)]
    assert block.split("\n") == [
        "Q: Which one?",
        "Controls:",
        *lines,
        rule,
    ]


def test_receipt_records_default_for_unset_and_digest_for_set_parts() -> None:
    assert TextParts().receipt() == {
        OPTION_BLOCK: DEFAULT_PART,
        CONTEXT_TEMPLATE: DEFAULT_PART,
    }
    template = "Text:\n{context}\n---\n{field_block}"
    digest = hashlib.sha256(template.encode("utf-8")).hexdigest()
    assert TextParts(context_template=template).receipt() == {
        OPTION_BLOCK: DEFAULT_PART,
        CONTEXT_TEMPLATE: digest,
    }
    assert DEFAULT_PART == "default"
