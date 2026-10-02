"""Contract tests: substitutable text parts on the scripted scoring fake (#364).

A caller substitutes the ``option_block`` or the ``context_template``. The
control lines, the candidate binding and the no-thinking prefill stay the
same. The default parts reproduce today's prefix byte for byte.

Examples:
    ```bash
    uv run pytest -q tests/contract/test_text_parts_scoring.py
    ```

See Also:
    - [typevet.domain.text_parts][]: Templates, validator and receipt
    - [typevet.adapters.outbound.judgment_scoring][]: Adapter under test
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

import pytest

from tests.fixtures.judgment_scoring_contract import SequentialScoringFake
from typevet.adapters.outbound.gemma import ServedTemplateClass
from typevet.adapters.outbound.gemma.served_template import (
    GEMMA4_MODEL_TURN_HEADER,
    GEMMA4_NO_THINKING_PREFILL,
)
from typevet.adapters.outbound.judgment_scoring import ScoringJudgmentAdapter
from typevet.adapters.outbound.vllm.scoring import ChatContentFraming
from typevet.domain.errors import GemmaTemplateError, JudgmentValidationError
from typevet.domain.judgment_questions import Choice, Noul, Score
from typevet.domain.judgment_response import JudgmentResponse
from typevet.domain.media import MEDIA_MARKER, ImageInput
from typevet.domain.text_parts import (
    CONTEXT_TEMPLATE,
    DEFAULT_CONTEXT_TEMPLATE,
    DEFAULT_OPTION_BLOCK,
    DEFAULT_PART,
    OPTION_BLOCK,
    TURN_MARKERS,
    TextParts,
)
from typevet.ports.framing import ModelFramingPort

pytestmark = pytest.mark.contract

_STATE = "Charged twice."
_QUESTIONS = {
    "noul": Noul(
        instructions="Billing issue?", criteria={"true": "Yes", "false": "No"}
    ),
    "route": Choice(
        criteria={"billing": "Money", "technical": "Bugs"}, instructions="Pick:"
    ),
    "quality": Score(criteria=["Poor", "Fair", "Good"], instructions="Rate:"),
}
_RULE = (
    "Answer with exactly one control string (the digit shown), "
    "not the original label text."
)
_TODAY_CHATML = [
    (
        "<|im_start|>user\nCharged twice.\n\nnoul: Billing issue?\n\nOptions:\n"
        f"Control 0 → false: No\nControl 1 → true: Yes\n\n{_RULE}<|im_end|>\n"
        "<|im_start|>assistant\n"
    ),
    (
        "<|im_start|>user\nCharged twice.\n\nroute: Pick:\n\nOptions:\n"
        f"0 → billing: Money\n1 → technical: Bugs\n\n{_RULE}<|im_end|>\n"
        "<|im_start|>assistant\n"
    ),
    (
        "<|im_start|>user\nCharged twice.\n\nquality: Rate:\n\nOptions:\n"
        "Control 0 → 0: Poor\nControl 1 → 1: Fair\nControl 2 → 2: Good\n\n"
        f"{_RULE}<|im_end|>\n<|im_start|>assistant\n"
    ),
]
_CONTROL_LINES = [
    ["Control 0 → false: No", "Control 1 → true: Yes"],
    ["0 → billing: Money", "1 → technical: Bugs"],
    ["Control 0 → 0: Poor", "Control 1 → 1: Fair", "Control 2 → 2: Good"],
]
_OPTION_BLOCK = (
    "Field {name}. {question}\nReply with one control.\n"
    "- {control} → {label}{description}\nEnd of options.\n{answer_rule}"
)
_CONTEXT_TEMPLATE = "Text under review:\n{context}\n---\n{field_block}\nDecide."


def _tokenize(text: str) -> tuple[int, ...]:
    return (ord(text[0]),) if text else ()


def _logprobs() -> list[dict[str, float]]:
    return [
        {"True": math.log(0.6), "False": math.log(0.4)},
        {"billing": math.log(0.75), "technical": math.log(0.25)},
        {"0": math.log(0.5), "1": math.log(0.3), "2": math.log(0.2)},
    ]


def _run(
    media: tuple[ImageInput, ...] | None = None, **kwargs: Any
) -> tuple[ScoringJudgmentAdapter, SequentialScoringFake, Any]:
    fake = SequentialScoringFake(_logprobs())
    adapter = ScoringJudgmentAdapter(fake, tokenize_content=_tokenize, **kwargs)
    response = adapter.judge(_STATE, _QUESTIONS, "fake-judgment", media=media)
    return adapter, fake, response


class _Gemma4FramingWithoutPrefill:
    """Framing that ends in a Gemma 4 model turn without the prefill.

    Examples:
        ```python
        _Gemma4FramingWithoutPrefill().compose_prefix(user_text="u", media=())
        ```
    """

    def compose_prefix(self, *, user_text: str, media: tuple[ImageInput, ...]) -> str:
        """Return the turn without the prefill.

        Returns:
            A Gemma 4 user turn and a bare model header.
        """
        del media
        return f"<|turn>user\n{user_text}<turn|>\n{GEMMA4_MODEL_TURN_HEADER}"


class _Gemma4FramingWithPrefill:
    """Framing that wraps the user text in a Gemma 4 turn with the prefill.

    Examples:
        ```python
        _Gemma4FramingWithPrefill().compose_prefix(user_text="u", media=())
        ```
    """

    def compose_prefix(self, *, user_text: str, media: tuple[ImageInput, ...]) -> str:
        """Return the user turn, the model header and the prefill.

        Returns:
            A Gemma 4 prefix that ends with the no-thinking prefill.
        """
        del media
        return (
            f"<|turn>user\n{user_text}<turn|>\n{GEMMA4_MODEL_TURN_HEADER}"
            f"{GEMMA4_NO_THINKING_PREFILL}"
        )


@pytest.mark.parametrize(
    "served", [None, ServedTemplateClass.NATIVE_GEMMA4_TURN], ids=["chatml", "gemma4"]
)
def test_default_parts_reproduce_today_prefix_byte_for_byte(
    served: ServedTemplateClass | None,
) -> None:
    _, implicit, _ = _run(served_template=served)
    _, explicit, _ = _run(
        served_template=served,
        text_parts=TextParts(
            option_block=DEFAULT_OPTION_BLOCK,
            context_template=DEFAULT_CONTEXT_TEMPLATE,
        ),
    )
    implicit_prefixes = [call.prefix for call in implicit.calls]
    assert [call.prefix for call in explicit.calls] == implicit_prefixes
    if served is None:
        assert implicit_prefixes == _TODAY_CHATML


@pytest.mark.parametrize(
    "served", [None, ServedTemplateClass.NATIVE_GEMMA4_TURN], ids=["chatml", "gemma4"]
)
def test_substituted_option_block_keeps_controls_and_binding(
    served: ServedTemplateClass | None,
) -> None:
    _, default_fake, default_response = _run(served_template=served)
    _, fake, response = _run(
        served_template=served, text_parts=TextParts(option_block=_OPTION_BLOCK)
    )
    for call, default_call, lines in zip(
        fake.calls, default_fake.calls, _CONTROL_LINES, strict=True
    ):
        assert call.prefix != default_call.prefix
        assert "End of options." in call.prefix
        positions = [call.prefix.index(f"- {line}\n") for line in lines]
        assert positions == sorted(positions)
        assert call.candidates == default_call.candidates
        if served is not None:
            assert call.prefix.endswith(GEMMA4_NO_THINKING_PREFILL)
    assert response.answers == default_response.answers


def test_substituted_context_template_keeps_prefill_and_media_markers() -> None:
    image = ImageInput(data=b"\x89PNG", mime_type="image/png")
    _, fake, _ = _run(
        media=(image,),
        served_template=ServedTemplateClass.NATIVE_GEMMA4_TURN,
        text_parts=TextParts(context_template=_CONTEXT_TEMPLATE),
    )
    for call in fake.calls:
        assert f"Text under review:\n{MEDIA_MARKER}\n{_STATE}\n---\n" in call.prefix
        assert call.prefix.count(MEDIA_MARKER) == 1
        assert call.prefix.endswith(GEMMA4_NO_THINKING_PREFILL)


def test_option_block_with_framing_still_requires_the_prefill() -> None:
    with pytest.raises(GemmaTemplateError):
        _run(
            framing=_Gemma4FramingWithoutPrefill(),
            text_parts=TextParts(option_block=_OPTION_BLOCK),
        )


def test_context_template_with_framing_still_requires_the_prefill() -> None:
    fake = SequentialScoringFake(_logprobs())
    with pytest.raises(GemmaTemplateError):
        ScoringJudgmentAdapter(
            fake,
            tokenize_content=_tokenize,
            framing=_Gemma4FramingWithoutPrefill(),
            text_parts=TextParts(context_template=_CONTEXT_TEMPLATE),
        ).judge(_STATE, _QUESTIONS, "fake-judgment")
    assert fake.calls == []


@pytest.mark.parametrize(
    ("framing", "start", "end"),
    [
        (ChatContentFraming(), "", ""),
        (
            _Gemma4FramingWithPrefill(),
            "<|turn>user\n",
            f"<turn|>\n{GEMMA4_MODEL_TURN_HEADER}{GEMMA4_NO_THINKING_PREFILL}",
        ),
    ],
    ids=["chat_content", "gemma4"],
)
def test_framing_renders_a_substituted_context_template(
    framing: ModelFramingPort, start: str, end: str
) -> None:
    _, default_fake, default_response = _run(framing=framing)
    _, fake, response = _run(
        framing=framing, text_parts=TextParts(context_template=_CONTEXT_TEMPLATE)
    )
    for call, default_call in zip(fake.calls, default_fake.calls, strict=True):
        field_block = default_call.prefix.removeprefix(
            f"{start}{_STATE}\n\n"
        ).removesuffix(end)
        user_text = f"Text under review:\n{_STATE}\n---\n{field_block}\nDecide."
        assert call.prefix == f"{start}{user_text}{end}"
        assert call.candidates == default_call.candidates
    assert response.answers == default_response.answers


def test_framing_and_served_template_are_still_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="served_template"):
        ScoringJudgmentAdapter(
            SequentialScoringFake([]),
            tokenize_content=_tokenize,
            framing=_Gemma4FramingWithPrefill(),
            served_template=ServedTemplateClass.NATIVE_GEMMA4_TURN,
            text_parts=TextParts(context_template=_CONTEXT_TEMPLATE),
        )


_LINE = "{control} → {label}{description}"
_RULE = "\n{answer_rule}"
_BAD_TEMPLATES: list[tuple[str, str]] = [
    ("option_block", "{question}\n{control}: {label}{description}" + _RULE),
    ("option_block", "{question}\n{control} → {label}" + _RULE),
    ("option_block", "{question}\n{control}{description} → {label}" + _RULE),
    ("option_block", "{question}\n{control} →{description} {label}" + _RULE),
    ("option_block", "{question} {question}\n" + _LINE + _RULE),
    ("option_block", "{question} gold answer\n" + _LINE + _RULE),
    ("option_block", "{question}\n" + _LINE),
    ("context_template", "{context} only"),
    ("context_template", "{context}{field_block}{field_block}"),
    ("context_template", "{context} expected answer {field_block}"),
    *(
        ("option_block", f"{{question}} {marker}\n{_LINE}{_RULE}")
        for marker in TURN_MARKERS
    ),
    *(
        ("context_template", f"{{context}}\n{marker}\n{{field_block}}")
        for marker in TURN_MARKERS
    ),
]


@pytest.mark.parametrize(("keyword", "template"), _BAD_TEMPLATES)
def test_bad_template_is_refused_before_any_scoring(
    keyword: str, template: str
) -> None:
    fake = SequentialScoringFake(_logprobs())
    with pytest.raises(JudgmentValidationError) as caught:
        ScoringJudgmentAdapter(
            fake,
            tokenize_content=_tokenize,
            text_parts=TextParts(**{keyword: template}),
        )
    message = str(caught.value)
    assert template not in message
    assert keyword in message
    assert not any(marker in message for marker in TURN_MARKERS)
    assert fake.calls == []


def test_receipt_records_digests_and_default() -> None:
    adapter, _, response = _run()
    default_block = {OPTION_BLOCK: DEFAULT_PART, CONTEXT_TEMPLATE: DEFAULT_PART}
    assert adapter.text_parts.receipt() == default_block
    assert response.text_parts == default_block
    substituted, _, substituted_response = _run(
        text_parts=TextParts(option_block=_OPTION_BLOCK)
    )
    digest = hashlib.sha256(_OPTION_BLOCK.encode("utf-8")).hexdigest()
    expected = {OPTION_BLOCK: digest, CONTEXT_TEMPLATE: DEFAULT_PART}
    assert substituted.text_parts.receipt() == expected
    assert substituted_response.text_parts == expected


def test_response_without_a_scoring_adapter_has_an_empty_text_parts_block() -> None:
    assert JudgmentResponse(model="m").text_parts == {}
