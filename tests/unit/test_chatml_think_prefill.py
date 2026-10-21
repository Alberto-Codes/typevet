"""Unit tests: a ChatML thinking prefix must end with an empty think block (#443).

A ChatML assistant tail on the framing path must end with one known empty
think block, so the first scored token is a menu option and not ``<think>``.
The Gemma 4 refusal message stays the same.

Examples:
    ```bash
    uv run pytest -q tests/unit/test_chatml_think_prefill.py
    ```

See Also:
    - [typevet.adapters.outbound.gemma.scoring_prefix][]: Check under test
"""

from __future__ import annotations

import math

import pytest

from tests.fixtures.judgment_scoring_contract import SequentialScoringFake
from typevet.adapters.outbound.gemma import (
    CHATML_ASSISTANT_HEADER,
    CHATML_IM_END,
    CHATML_IM_START,
    GEMMA4_MODEL_TURN_HEADER,
    GEMMA4_TURN_CLOSE,
    GEMMA4_TURN_OPEN,
    require_no_thinking_prefill,
)
from typevet.adapters.outbound.judgment_scoring import ScoringJudgmentAdapter
from typevet.domain.errors import GemmaTemplateError
from typevet.domain.judgment_questions import Noul
from typevet.domain.media import ImageInput

pytestmark = pytest.mark.unit

_FIELD_BLOCK = "field-block"
_STATE_TEXT = "state-text-that-must-not-leak"
_QWEN_FORM = "<think>\n\n</think>\n\n"
_COMPACT_FORM = "<think></think>"


def _chatml(tail: str) -> str:
    return (
        f"{CHATML_IM_START}user\n{_STATE_TEXT}{_FIELD_BLOCK}{CHATML_IM_END}\n"
        f"{CHATML_ASSISTANT_HEADER}{tail}"
    )


class _BareChatMLFraming:
    """Caller framing that stops at the ChatML assistant header."""

    def compose_prefix(self, *, user_text: str, media: tuple[ImageInput, ...]) -> str:
        """Return a ChatML prefix without an empty think block.

        Returns:
            Prefix that ends with the ChatML assistant header.
        """
        del media
        return f"{CHATML_IM_START}user\n{user_text}{CHATML_IM_END}\n{CHATML_ASSISTANT_HEADER}"


@pytest.mark.parametrize(
    "tail",
    ["", "<think>\n", "<think>\n\n</think>", "answer"],
    ids=["bare", "open_think", "no_trailing_newlines", "other_text"],
)
def test_chatml_tail_without_empty_think_block_is_refused(tail: str) -> None:
    """A ChatML assistant tail that lacks a known empty think block is refused."""
    with pytest.raises(GemmaTemplateError) as excinfo:
        require_no_thinking_prefill(
            _chatml(tail), field_block=_FIELD_BLOCK, framing_class="Probe"
        )
    message = str(excinfo.value)
    assert "Probe" in message
    assert _STATE_TEXT not in message
    assert CHATML_IM_START not in message


@pytest.mark.parametrize("form", [_QWEN_FORM, _COMPACT_FORM], ids=["qwen", "compact"])
def test_chatml_tail_with_empty_think_block_is_allowed(form: str) -> None:
    """Each known empty think block passes the check."""
    require_no_thinking_prefill(
        _chatml(form), field_block=_FIELD_BLOCK, framing_class="Probe"
    )


def test_gemma4_refusal_message_is_unchanged() -> None:
    """The Gemma 4 refusal keeps its message."""
    prefix = (
        f"{GEMMA4_TURN_OPEN}user\n{_FIELD_BLOCK}{GEMMA4_TURN_CLOSE}\n"
        f"{GEMMA4_MODEL_TURN_HEADER}"
    )
    with pytest.raises(GemmaTemplateError) as excinfo:
        require_no_thinking_prefill(
            prefix, field_block=_FIELD_BLOCK, framing_class="Probe"
        )
    assert str(excinfo.value) == (
        "framing Probe renders a Gemma 4 model turn without the no-thinking prefill"
    )


def test_bare_chatml_framing_is_refused_before_any_call() -> None:
    """The adapter refuses a bare ChatML framing before it calls the scorer."""
    fake = SequentialScoringFake([{"True": math.log(0.7), "False": math.log(0.3)}])
    adapter = ScoringJudgmentAdapter(
        fake,
        tokenize_content=lambda text: (ord(text[0]),) if text else (),
        framing=_BareChatMLFraming(),
    )
    with pytest.raises(GemmaTemplateError) as excinfo:
        adapter.judge(_STATE_TEXT, {"flagged": Noul(instructions="Flag it?")}, "m")
    assert "_BareChatMLFraming" in str(excinfo.value)
    assert _STATE_TEXT not in str(excinfo.value)
    assert fake.calls == []


def test_prefix_without_field_block_has_no_framing_tail() -> None:
    """A bare ChatML prefix that does not hold the field block is not checked."""
    prefix = f"{CHATML_IM_START}user\nother{CHATML_IM_END}\n{CHATML_ASSISTANT_HEADER}"
    require_no_thinking_prefill(prefix, field_block=_FIELD_BLOCK, framing_class="P")
