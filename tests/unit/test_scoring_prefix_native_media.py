"""Unit tests for native Gemma 3 media scoring prefixes (#157)."""

from __future__ import annotations

import math

import pytest

from tests.fixtures.judgment_scoring_contract import SequentialScoringFake
from typevet.adapters.outbound.gemma import (
    CHATML_ASSISTANT_HEADER,
    CHATML_IM_START,
    GEMMA3_END_OF_TURN,
    GEMMA3_MODEL_TURN_HEADER,
    GEMMA3_START_OF_TURN,
    ServedTemplateClass,
    classify_served_template,
    compose_media_scoring_prefix,
)
from typevet.adapters.outbound.judgment_scoring import ScoringJudgmentAdapter
from typevet.domain.errors import GemmaTemplateError, JudgmentValidationError
from typevet.domain.judgment_questions import Choice, Noul
from typevet.domain.media import MEDIA_MARKER, ImageInput

pytestmark = pytest.mark.unit

_IMAGE = ImageInput(data=b"\x89PNG\r\n\x1a\nunit", mime_type="image/png")


def _served_gemma3(content: str) -> str:
    """Mirror ``/apply-template`` output for gemma-3-4b-it-q4km-mm."""
    return f"<start_of_turn>user\n{content}<end_of_turn>\n<start_of_turn>model\n"


def _tokenize(text: str) -> tuple[int, ...]:
    return (ord(text[0]),) if text else ()


def _questions() -> dict[str, Noul | Choice]:
    return {
        "flagged": Noul(instructions="Does the image show a receipt?"),
        "route": Choice(
            criteria={"billing": "Money", "technical": "Bugs"},
            instructions="Pick:",
        ),
    }


def _logprobs() -> list[dict[str, float]]:
    return [
        {"True": math.log(0.6), "False": math.log(0.4)},
        {"billing": math.log(0.7), "technical": math.log(0.3)},
    ]


def _adapter(
    served_template: ServedTemplateClass | None,
) -> tuple[ScoringJudgmentAdapter, SequentialScoringFake]:
    fake = SequentialScoringFake(_logprobs())
    adapter = ScoringJudgmentAdapter(
        fake, tokenize_content=_tokenize, served_template=served_template
    )
    return adapter, fake


def test_classify_gemma3_start_of_turn_family() -> None:
    rendered = _served_gemma3("hello")
    assert classify_served_template(rendered) is ServedTemplateClass.NATIVE_GEMMA3_TURN


def test_classify_gemma3_mixed_with_other_family_is_unsupported() -> None:
    for extra in (CHATML_IM_START, "<|turn>"):
        rendered = _served_gemma3("hello") + extra
        assert classify_served_template(rendered) is ServedTemplateClass.UNSUPPORTED


def test_media_prefix_matches_served_gemma3_shape() -> None:
    prefix = compose_media_scoring_prefix(
        context=f"{MEDIA_MARKER}\nLook.",
        field_block="Control 0 → red",
        template_class=ServedTemplateClass.NATIVE_GEMMA3_TURN,
    )
    assert prefix == _served_gemma3(f"{MEDIA_MARKER}\nLook.\n\nControl 0 → red")
    assert prefix.startswith(f"{GEMMA3_START_OF_TURN}user\n")
    assert prefix.endswith(f"{GEMMA3_END_OF_TURN}\n{GEMMA3_MODEL_TURN_HEADER}")
    assert CHATML_IM_START not in prefix


@pytest.mark.parametrize(
    "template_class",
    [
        ServedTemplateClass.NATIVE_GEMMA4_TURN,
        ServedTemplateClass.DEGRADED_CHATML,
        ServedTemplateClass.UNSUPPORTED,
    ],
)
def test_media_prefix_fails_closed_off_gemma3(
    template_class: ServedTemplateClass,
) -> None:
    with pytest.raises(GemmaTemplateError):
        compose_media_scoring_prefix(
            context="x", field_block="y", template_class=template_class
        )


def test_adapter_media_uses_native_prefix_and_keeps_bindings() -> None:
    adapter, fake = _adapter(ServedTemplateClass.NATIVE_GEMMA3_TURN)
    media = (_IMAGE, ImageInput(data=b"second", mime_type="image/jpeg"))
    response = adapter.judge("Attached receipt.", _questions(), "gemma-mm", media=media)
    assert len(fake.calls) == 2
    for call in fake.calls:
        assert call.prefix.startswith(f"{GEMMA3_START_OF_TURN}user\n")
        assert call.prefix.endswith(GEMMA3_MODEL_TURN_HEADER)
        assert CHATML_IM_START not in call.prefix
        assert call.prefix.count(MEDIA_MARKER) == len(media)
        assert call.media == media
    assert "Control 0 → billing: Money" in fake.calls[1].prefix
    assert response.choices["route"].choice == "billing"
    assert response.nouls["flagged"].noul == pytest.approx(0.6)


def test_adapter_text_only_keeps_chatml_under_gemma3() -> None:
    adapter, fake = _adapter(ServedTemplateClass.NATIVE_GEMMA3_TURN)
    baseline, baseline_fake = _adapter(None)
    adapter.judge("Charged twice.", _questions(), "gemma-mm")
    baseline.judge("Charged twice.", _questions(), "gemma-mm", media=())
    prefixes = [call.prefix for call in fake.calls]
    assert prefixes == [call.prefix for call in baseline_fake.calls]
    for prefix in prefixes:
        assert prefix.startswith(f"{CHATML_IM_START}user\nCharged twice.\n\n")
        assert prefix.endswith(CHATML_ASSISTANT_HEADER)
        assert GEMMA3_START_OF_TURN not in prefix


@pytest.mark.parametrize(
    "served_template",
    [
        None,
        ServedTemplateClass.NATIVE_GEMMA4_TURN,
        ServedTemplateClass.DEGRADED_CHATML,
        ServedTemplateClass.UNSUPPORTED,
    ],
)
def test_adapter_media_fails_closed_before_scoring_off_gemma3(
    served_template: ServedTemplateClass | None,
) -> None:
    adapter, fake = _adapter(served_template)
    with pytest.raises(JudgmentValidationError, match="served template"):
        adapter.judge("x", _questions(), "gemma-mm", media=(_IMAGE,))
    assert fake.calls == []
