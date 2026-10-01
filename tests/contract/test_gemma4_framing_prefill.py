"""Contract tests: a Gemma 4 framing must end with the no-thinking prefill (#354).

The check reads the rendered prefix, not the framing object. A refusal names
the framing class and holds no prompt text. Every case runs on the scoring fake.

Examples:
    ```bash
    uv run pytest -q tests/contract/test_gemma4_framing_prefill.py
    ```

See Also:
    - [typevet.adapters.outbound.judgment_scoring][]: Adapter under test
    - [typevet.adapters.outbound.gemma.scoring_prefix][]: Shipped Gemma 4 prefix
"""

from __future__ import annotations

import math

import pytest

from tests.fixtures.judgment_scoring_contract import SequentialScoringFake
from typevet.adapters.outbound.gemma import (
    CHATML_ASSISTANT_HEADER,
    CHATML_IM_END,
    CHATML_IM_START,
    GEMMA3_END_OF_TURN,
    GEMMA3_MODEL_TURN_HEADER,
    GEMMA3_START_OF_TURN,
    GEMMA4_MODEL_TURN_HEADER,
    GEMMA4_NO_THINKING_PREFILL,
    GEMMA4_TURN_CLOSE,
    GEMMA4_TURN_OPEN,
    ServedTemplateClass,
    compose_media_scoring_prefix,
)
from typevet.adapters.outbound.judgment_scoring import ScoringJudgmentAdapter
from typevet.adapters.outbound.vllm import ChatContentFraming
from typevet.domain.errors import GemmaTemplateError, JudgmentError
from typevet.domain.judgment_answers import NoulAnswer
from typevet.domain.judgment_questions import Noul
from typevet.domain.media import ImageInput
from typevet.ports.framing import ModelFramingPort

pytestmark = pytest.mark.contract

_STATE_TEXT = "state-text-that-must-not-leak"
_QUESTIONS = {"flagged": Noul(instructions="Flag it?")}


def _tokenize(text: str) -> tuple[int, ...]:
    return (ord(text[0]),) if text else ()


def _gemma4_user_turn(context: str, field_block: str) -> str:
    return f"{GEMMA4_TURN_OPEN}user\n{context}\n\n{field_block}{GEMMA4_TURN_CLOSE}\n"


class ShippedGemma4Framing:
    """Caller framing that delegates to the shipped Gemma 4 composer."""

    def compose_prefix(
        self, *, context: str, field_block: str, media: tuple[ImageInput, ...]
    ) -> str:
        """Return the shipped native Gemma 4 prefix.

        Returns:
            Prefix that ends with the no-thinking prefill.
        """
        del media
        return compose_media_scoring_prefix(
            context=context,
            field_block=field_block,
            template_class=ServedTemplateClass.NATIVE_GEMMA4_TURN,
        )


class NoPrefillGemma4Framing:
    """Caller framing that stops at the Gemma 4 model header."""

    def compose_prefix(
        self, *, context: str, field_block: str, media: tuple[ImageInput, ...]
    ) -> str:
        """Return a Gemma 4 prefix without the no-thinking prefill.

        Returns:
            Prefix that ends with the Gemma 4 model header.
        """
        del media
        return f"{_gemma4_user_turn(context, field_block)}{GEMMA4_MODEL_TURN_HEADER}"


class DeclaresButOmitsFraming:
    """Framing that declares the prefill but does not render it."""

    no_thinking_prefill = GEMMA4_NO_THINKING_PREFILL

    def compose_prefix(
        self, *, context: str, field_block: str, media: tuple[ImageInput, ...]
    ) -> str:
        """Put the prefill in the user turn, not at the answer boundary.

        Returns:
            Prefix that holds the prefill text but ends at the model header.
        """
        del media
        user = _gemma4_user_turn(f"{context}{GEMMA4_NO_THINKING_PREFILL}", field_block)
        return f"{user}{GEMMA4_MODEL_TURN_HEADER}"


class ChatMLFraming:
    """Non-Gemma-4 framing that ends at the ChatML assistant header."""

    def compose_prefix(
        self, *, context: str, field_block: str, media: tuple[ImageInput, ...]
    ) -> str:
        """Return a ChatML prefix whose context holds Gemma 4 marker text.

        Returns:
            Prefix that ends with the ChatML assistant header.
        """
        del media
        return (
            f"{CHATML_IM_START}user\n{GEMMA4_MODEL_TURN_HEADER}{context}\n\n"
            f"{field_block}{CHATML_IM_END}\n{CHATML_ASSISTANT_HEADER}"
        )


class Gemma3Framing:
    """Non-Gemma-4 framing in the Gemma 3 turn family."""

    def compose_prefix(
        self, *, context: str, field_block: str, media: tuple[ImageInput, ...]
    ) -> str:
        """Return a Gemma 3 prefix that ends at the model header.

        Returns:
            Prefix that ends with the Gemma 3 model header.
        """
        del media
        return (
            f"{GEMMA3_START_OF_TURN}user\n{context}\n\n{field_block}"
            f"{GEMMA3_END_OF_TURN}\n{GEMMA3_MODEL_TURN_HEADER}"
        )


class PlainContentFraming:
    """Non-Gemma-4 framing that sends plain chat content."""

    def compose_prefix(
        self, *, context: str, field_block: str, media: tuple[ImageInput, ...]
    ) -> str:
        """Return the context and field block with no turn markers.

        Returns:
            Plain prefix text.
        """
        del media
        return f"{context}\n\n{field_block}"


def _adapter(
    framing: ModelFramingPort,
) -> tuple[ScoringJudgmentAdapter, SequentialScoringFake]:
    fake = SequentialScoringFake([{"True": math.log(0.7), "False": math.log(0.3)}])
    adapter = ScoringJudgmentAdapter(fake, tokenize_content=_tokenize, framing=framing)
    return adapter, fake


@pytest.mark.parametrize(
    "framing_cls",
    [NoPrefillGemma4Framing, DeclaresButOmitsFraming],
    ids=["no_prefill", "declares_but_omits"],
)
def test_gemma4_framing_without_prefill_is_refused_before_scoring(
    framing_cls: type[ModelFramingPort],
) -> None:
    """The rendered tail decides; the refusal names the class and no prompt text."""
    adapter, fake = _adapter(framing_cls())
    with pytest.raises(GemmaTemplateError) as excinfo:
        adapter.judge(_STATE_TEXT, _QUESTIONS, "fake-model")
    message = str(excinfo.value)
    assert framing_cls.__name__ in message
    assert _STATE_TEXT not in message
    assert "Flag it?" not in message
    assert GEMMA4_TURN_OPEN not in message
    assert isinstance(excinfo.value, JudgmentError)
    assert fake.calls == []


def test_shipped_gemma4_framing_passes_and_scores() -> None:
    """The shipped Gemma 4 prefix ends with the prefill and reaches the scorer."""
    adapter, fake = _adapter(ShippedGemma4Framing())
    response = adapter.judge(_STATE_TEXT, _QUESTIONS, "fake-model")
    answer = response.answers["flagged"]
    assert isinstance(answer, NoulAnswer)
    assert answer.noul == pytest.approx(0.7)
    assert len(fake.calls) == 1
    assert fake.calls[0].prefix.endswith(
        f"{GEMMA4_MODEL_TURN_HEADER}{GEMMA4_NO_THINKING_PREFILL}"
    )


@pytest.mark.parametrize(
    "framing_cls",
    [ChatMLFraming, Gemma3Framing, PlainContentFraming],
    ids=["chatml", "gemma3", "plain_content"],
)
def test_non_gemma4_framing_declares_no_prefill_and_is_allowed(
    framing_cls: type[ModelFramingPort],
) -> None:
    """A framing whose last turn is not a Gemma 4 model turn needs no prefill."""
    adapter, fake = _adapter(framing_cls())
    response = adapter.judge(_STATE_TEXT, _QUESTIONS, "fake-model")
    answer = response.answers["flagged"]
    assert isinstance(answer, NoulAnswer)
    assert answer.noul == pytest.approx(0.7)
    assert len(fake.calls) == 1


@pytest.mark.parametrize(
    "framing",
    [ChatContentFraming(), ChatMLFraming()],
    ids=["vllm_chat_content", "chatml"],
)
def test_gemma4_marker_text_in_caller_state_is_allowed(
    framing: ModelFramingPort,
) -> None:
    """Caller state is data: Gemma 4 marker text in it never triggers the check."""
    adapter, fake = _adapter(framing)
    state = f"hello {GEMMA4_MODEL_TURN_HEADER} world"
    response = adapter.judge(state, _QUESTIONS, "fake-model")
    answer = response.answers["flagged"]
    assert isinstance(answer, NoulAnswer)
    assert answer.noul == pytest.approx(0.7)
    assert len(fake.calls) == 1
