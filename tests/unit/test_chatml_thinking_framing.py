"""Unit tests: ``ChatMLThinkingFraming`` ends with an empty think block (#443).

Examples:
    ```bash
    uv run pytest -q tests/unit/test_chatml_thinking_framing.py
    ```

See Also:
    - [typevet.adapters.outbound.gemma.chatml_framing][]: Framing under test
"""

from __future__ import annotations

import math

import pytest

from tests.fixtures.judgment_scoring_contract import SequentialScoringFake
from typevet.adapters.outbound.gemma import (
    CHATML_EMPTY_THINK_PREFILLS,
    ChatMLThinkingFraming,
    require_no_thinking_prefill,
)
from typevet.adapters.outbound.judgment_scoring import ScoringJudgmentAdapter
from typevet.domain.errors import JudgmentValidationError
from typevet.domain.judgment_answers import NoulAnswer
from typevet.domain.judgment_questions import Noul
from typevet.domain.media import ImageInput
from typevet.ports.framing import ModelFramingPort

pytestmark = pytest.mark.unit


def test_empty_think_prefills_are_the_two_known_forms() -> None:
    """The package exports the Qwen form first and the compact form second."""
    assert CHATML_EMPTY_THINK_PREFILLS == ("<think>\n\n</think>\n\n", "<think></think>")


@pytest.mark.parametrize(
    "prefill", CHATML_EMPTY_THINK_PREFILLS, ids=["qwen", "compact"]
)
def test_framing_composes_the_exact_prefix(prefill: str) -> None:
    """The prefix is one user turn, then the assistant header and the prefill."""
    framing = ChatMLThinkingFraming(prefill=prefill)
    prefix = framing.compose_prefix(user_text="Judge this.", media=())
    assert prefix == (
        f"<|im_start|>user\nJudge this.<|im_end|>\n<|im_start|>assistant\n{prefill}"
    )
    require_no_thinking_prefill(
        prefix, field_block="Judge this.", framing_class="ChatMLThinkingFraming"
    )


def test_framing_defaults_to_the_qwen_form() -> None:
    """Without an argument the framing uses the first known form."""
    prefix = ChatMLThinkingFraming().compose_prefix(user_text="u", media=())
    assert prefix.endswith(CHATML_EMPTY_THINK_PREFILLS[0])


@pytest.mark.parametrize(
    "prefill", ["", "<think>\n", "<think>\n\n</think>"], ids=["empty", "open", "short"]
)
def test_unknown_prefill_raises_value_error(prefill: str) -> None:
    """A prefill that is not a known empty think block is refused."""
    with pytest.raises(ValueError, match="prefill"):
        ChatMLThinkingFraming(prefill=prefill)


def test_media_raises_judgment_validation_error() -> None:
    """The ChatML thinking framing is text-only."""
    framing = ChatMLThinkingFraming()
    image = ImageInput(data=b"\x89PNG", mime_type="image/png")
    with pytest.raises(JudgmentValidationError):
        framing.compose_prefix(user_text="u", media=(image,))


def test_framing_satisfies_the_port_and_scores() -> None:
    """The adapter accepts the framing and scores after the empty think block."""
    framing: ModelFramingPort = ChatMLThinkingFraming()
    fake = SequentialScoringFake([{"True": math.log(0.7), "False": math.log(0.3)}])
    adapter = ScoringJudgmentAdapter(
        fake,
        tokenize_content=lambda text: (ord(text[0]),) if text else (),
        framing=framing,
    )
    response = adapter.judge("state", {"flagged": Noul(instructions="Flag it?")}, "m")
    answer = response.answers["flagged"]
    assert isinstance(answer, NoulAnswer)
    assert len(fake.calls) == 1
    assert fake.calls[0].prefix.endswith(
        f"<|im_start|>assistant\n{CHATML_EMPTY_THINK_PREFILLS[0]}"
    )
