"""Unit tests for native media scoring prefixes (#157, #171, #179, #187)."""

from __future__ import annotations

import base64
import json
import math
from pathlib import Path

import pytest

from tests.fixtures.judgment_scoring_contract import SequentialScoringFake
from typevet.adapters.outbound.gemma import (
    CHATML_ASSISTANT_HEADER,
    CHATML_IM_START,
    GEMMA3_END_OF_TURN,
    GEMMA3_MODEL_TURN_HEADER,
    GEMMA3_START_OF_TURN,
    GEMMA4_MODEL_TURN_HEADER,
    GEMMA4_NO_THINKING_PREFILL,
    GEMMA4_TURN_CLOSE,
    GEMMA4_TURN_OPEN,
    ServedTemplateClass,
    classify_served_template,
    compose_media_scoring_prefix,
)
from typevet.adapters.outbound.judgment_scoring import ScoringJudgmentAdapter
from typevet.domain.errors import GemmaTemplateError, JudgmentValidationError
from typevet.domain.judgment_questions import Choice, Noul
from typevet.domain.media import MEDIA_MARKER, ImageInput

pytestmark = pytest.mark.unit

_GEMMA4_31B_SUFFIX_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "gemma4"
    / "gemma4_31b_no_thinking_generation_suffix.json"
)

_IMAGE = ImageInput(data=b"\x89PNG\r\n\x1a\nunit", mime_type="image/png")


def _pinned_gemma4_31b_no_thinking_suffix() -> str:
    payload = json.loads(_GEMMA4_31B_SUFFIX_FIXTURE.read_text(encoding="utf-8"))
    raw = base64.b64decode(payload["generation_prompt_suffix_b64"])
    return raw.decode("utf-8")


def _served_gemma3(content: str) -> str:
    """Mirror ``/apply-template`` output for gemma-3-4b-it-q4km-mm."""
    return f"<start_of_turn>user\n{content}<end_of_turn>\n<start_of_turn>model\n"


def _served_gemma4(content: str) -> str:
    """Mirror the native Gemma 4 turn shape for a single user turn.

    Returns:
        User turn wrapped in ``<|turn>`` markers, ending at no-thinking prefill.
    """
    return (
        f"{GEMMA4_TURN_OPEN}user\n{content}{GEMMA4_TURN_CLOSE}\n"
        f"{GEMMA4_MODEL_TURN_HEADER}{GEMMA4_NO_THINKING_PREFILL}"
    )


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


def test_gemma4_media_prefix_ends_with_pinned_31b_no_thinking_suffix() -> None:
    pinned_suffix = _pinned_gemma4_31b_no_thinking_suffix()
    field_block = "Control 0 → red"
    with_media = compose_media_scoring_prefix(
        context=f"{MEDIA_MARKER}\nLook.",
        field_block=field_block,
        template_class=ServedTemplateClass.NATIVE_GEMMA4_TURN,
    )
    without_media = compose_media_scoring_prefix(
        context="Look.",
        field_block=field_block,
        template_class=ServedTemplateClass.NATIVE_GEMMA4_TURN,
    )
    assert with_media.endswith(pinned_suffix)
    assert without_media.endswith(pinned_suffix)
    assert with_media.replace(f"{MEDIA_MARKER}\n", "", 1) == without_media


def test_media_prefix_matches_served_gemma4_shape() -> None:
    prefix = compose_media_scoring_prefix(
        context=f"{MEDIA_MARKER}\nLook.",
        field_block="Control 0 → red",
        template_class=ServedTemplateClass.NATIVE_GEMMA4_TURN,
    )
    assert prefix == _served_gemma4(f"{MEDIA_MARKER}\nLook.\n\nControl 0 → red")
    assert prefix.startswith(f"{GEMMA4_TURN_OPEN}user\n")
    assert prefix.endswith(
        f"{GEMMA4_TURN_CLOSE}\n{GEMMA4_MODEL_TURN_HEADER}{GEMMA4_NO_THINKING_PREFILL}"
    )
    assert MEDIA_MARKER in prefix
    assert CHATML_IM_START not in prefix
    assert GEMMA3_START_OF_TURN not in prefix
    assert GEMMA3_END_OF_TURN not in prefix
    assert classify_served_template(prefix) is ServedTemplateClass.NATIVE_GEMMA4_TURN


@pytest.mark.parametrize(
    "template_class",
    [
        ServedTemplateClass.DEGRADED_CHATML,
        ServedTemplateClass.UNSUPPORTED,
    ],
)
def test_media_prefix_fails_closed_off_native_families(
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
    assert "\n0 → billing: Money" in fake.calls[1].prefix
    assert response.choices["route"].choice == "billing"
    assert response.nouls["flagged"].noul == pytest.approx(0.6)


@pytest.mark.parametrize("omitted", [None, ()])
def test_adapter_omitted_media_keeps_native_gemma3_prefix(
    omitted: tuple[ImageInput, ...] | None,
) -> None:
    present, present_fake = _adapter(ServedTemplateClass.NATIVE_GEMMA3_TURN)
    absent, absent_fake = _adapter(ServedTemplateClass.NATIVE_GEMMA3_TURN)
    present.judge("Charged twice.", _questions(), "gemma-mm", media=(_IMAGE,))
    absent.judge("Charged twice.", _questions(), "gemma-mm", media=omitted)
    pairs = list(zip(present_fake.calls, absent_fake.calls, strict=True))
    assert len(pairs) == 2
    for with_image, without_image in pairs:
        assert without_image.prefix.startswith(f"{GEMMA3_START_OF_TURN}user\n")
        assert without_image.prefix.endswith(
            f"{GEMMA3_END_OF_TURN}\n{GEMMA3_MODEL_TURN_HEADER}"
        )
        assert CHATML_IM_START not in without_image.prefix
        assert MEDIA_MARKER not in without_image.prefix
        assert with_image.prefix.replace(f"{MEDIA_MARKER}\n", "", 1) == (
            without_image.prefix
        )
        assert with_image.candidates == without_image.candidates
        assert with_image.media == (_IMAGE,)
        assert without_image.media == ()


def test_adapter_media_uses_native_gemma4_prefix_and_keeps_bindings() -> None:
    adapter, fake = _adapter(ServedTemplateClass.NATIVE_GEMMA4_TURN)
    media = (_IMAGE, ImageInput(data=b"second", mime_type="image/jpeg"))
    response = adapter.judge(
        "Attached receipt.", _questions(), "gemma4-mm", media=media
    )
    assert len(fake.calls) == 2
    for call in fake.calls:
        assert call.prefix.startswith(f"{GEMMA4_TURN_OPEN}user\n")
        assert call.prefix.endswith(
            f"{GEMMA4_TURN_CLOSE}\n{GEMMA4_MODEL_TURN_HEADER}{GEMMA4_NO_THINKING_PREFILL}"
        )
        assert CHATML_IM_START not in call.prefix
        assert GEMMA3_START_OF_TURN not in call.prefix
        assert call.prefix.count(MEDIA_MARKER) == len(media)
        assert call.media == media
        assert (
            classify_served_template(call.prefix)
            is ServedTemplateClass.NATIVE_GEMMA4_TURN
        )
    assert "\n0 → billing: Money" in fake.calls[1].prefix
    assert response.choices["route"].choice == "billing"
    assert response.nouls["flagged"].noul == pytest.approx(0.6)


@pytest.mark.parametrize("omitted", [None, ()])
def test_adapter_omitted_media_keeps_native_gemma4_prefix(
    omitted: tuple[ImageInput, ...] | None,
) -> None:
    present, present_fake = _adapter(ServedTemplateClass.NATIVE_GEMMA4_TURN)
    absent, absent_fake = _adapter(ServedTemplateClass.NATIVE_GEMMA4_TURN)
    present.judge("Charged twice.", _questions(), "gemma4-mm", media=(_IMAGE,))
    absent.judge("Charged twice.", _questions(), "gemma4-mm", media=omitted)
    pairs = list(zip(present_fake.calls, absent_fake.calls, strict=True))
    assert len(pairs) == 2
    for with_image, without_image in pairs:
        assert without_image.prefix.startswith(f"{GEMMA4_TURN_OPEN}user\n")
        assert without_image.prefix.endswith(
            f"{GEMMA4_TURN_CLOSE}\n{GEMMA4_MODEL_TURN_HEADER}{GEMMA4_NO_THINKING_PREFILL}"
        )
        assert CHATML_IM_START not in without_image.prefix
        assert GEMMA3_START_OF_TURN not in without_image.prefix
        assert MEDIA_MARKER not in without_image.prefix
        assert with_image.prefix.replace(f"{MEDIA_MARKER}\n", "", 1) == (
            without_image.prefix
        )
        assert with_image.candidates == without_image.candidates
        assert with_image.media == (_IMAGE,)
        assert without_image.media == ()


def test_adapter_text_only_uses_native_gemma4_wrappers() -> None:
    adapter, fake = _adapter(ServedTemplateClass.NATIVE_GEMMA4_TURN)
    adapter.judge("Charged twice.", _questions(), "gemma4-mm")
    assert len(fake.calls) == 2
    for call in fake.calls:
        assert call.prefix.startswith(f"{GEMMA4_TURN_OPEN}user\nCharged twice.\n\n")
        assert call.prefix.endswith(
            f"{GEMMA4_TURN_CLOSE}\n{GEMMA4_MODEL_TURN_HEADER}{GEMMA4_NO_THINKING_PREFILL}"
        )
        assert CHATML_IM_START not in call.prefix
        assert GEMMA3_START_OF_TURN not in call.prefix
        assert MEDIA_MARKER not in call.prefix


@pytest.mark.parametrize("served_template", [None, ServedTemplateClass.DEGRADED_CHATML])
def test_adapter_text_only_keeps_chatml_without_native_family(
    served_template: ServedTemplateClass | None,
) -> None:
    adapter, fake = _adapter(served_template)
    adapter.judge("Charged twice.", _questions(), "gemma-mm")
    assert len(fake.calls) == 2
    for call in fake.calls:
        assert call.prefix.startswith(f"{CHATML_IM_START}user\nCharged twice.\n\n")
        assert call.prefix.endswith(CHATML_ASSISTANT_HEADER)
        assert GEMMA3_START_OF_TURN not in call.prefix


def test_adapter_text_only_rejects_unsupported_family_before_scoring() -> None:
    adapter, fake = _adapter(ServedTemplateClass.UNSUPPORTED)
    with pytest.raises(JudgmentValidationError, match="served template"):
        adapter.judge("x", _questions(), "gemma-mm")
    assert fake.calls == []


@pytest.mark.parametrize(
    "served_template",
    [
        None,
        ServedTemplateClass.DEGRADED_CHATML,
        ServedTemplateClass.UNSUPPORTED,
    ],
)
def test_adapter_media_fails_closed_before_scoring_off_native_families(
    served_template: ServedTemplateClass | None,
) -> None:
    adapter, fake = _adapter(served_template)
    with pytest.raises(JudgmentValidationError, match="served template"):
        adapter.judge("x", _questions(), "gemma-mm", media=(_IMAGE,))
    assert fake.calls == []
