"""Contract tests for media threading through ScoringJudgmentAdapter (#142)."""

from __future__ import annotations

import math
from collections.abc import Callable
from inspect import Parameter, signature

import pytest

from tests.fixtures.judgment_scoring_contract import adapter_for
from typevet.adapters.outbound.gemma import (
    CHATML_IM_START,
    GEMMA3_MODEL_TURN_HEADER,
    GEMMA3_START_OF_TURN,
    ServedTemplateClass,
)
from typevet.adapters.outbound.judgment_scoring import ScoringJudgmentAdapter
from typevet.domain.judgment_questions import Choice, Noul
from typevet.domain.media import MEDIA_MARKER, ImageInput
from typevet.ports.judgment import JudgmentPort

_PNG = b"\x89PNG\r\n\x1a\nsynthetic-bytes"
_GEMMA3 = ServedTemplateClass.NATIVE_GEMMA3_TURN


def _image() -> ImageInput:
    return ImageInput(data=_PNG, mime_type="image/png")


def _two_field_logprobs() -> list[dict[str, float]]:
    return [
        {"True": math.log(0.6), "False": math.log(0.4)},
        {"billing": math.log(0.7), "technical": math.log(0.3)},
    ]


def _questions() -> dict[str, Noul | Choice]:
    return {
        "flagged": Noul(instructions="Does the image show a receipt?"),
        "route": Choice(
            criteria={"billing": "Money", "technical": "Bugs"},
            instructions="Pick:",
        ),
    }


@pytest.mark.contract
def test_media_rides_on_every_scored_field_request() -> None:
    adapter, fake = adapter_for(
        logprobs_by_call=_two_field_logprobs(), served_template=_GEMMA3
    )
    media = (_image(),)
    adapter.judge("Attached receipt.", _questions(), "gemma-mm", media=media)
    assert len(fake.calls) == 2
    for call in fake.calls:
        assert call.media == media
        assert call.prefix.count(MEDIA_MARKER) == 1
        assert call.prefix.startswith(f"{GEMMA3_START_OF_TURN}user\n")
        assert call.prefix.endswith(GEMMA3_MODEL_TURN_HEADER)
        assert CHATML_IM_START not in call.prefix


@pytest.mark.contract
def test_media_markers_match_image_count() -> None:
    adapter, fake = adapter_for(
        logprobs_by_call=_two_field_logprobs()[:1], served_template=_GEMMA3
    )
    media = (_image(), ImageInput(data=b"second", mime_type="image/jpeg"))
    adapter.judge("Two attachments.", {"flagged": Noul()}, "gemma-mm", media=media)
    assert fake.calls[0].prefix.count(MEDIA_MARKER) == 2
    assert fake.calls[0].media == media


@pytest.mark.contract
def test_media_prefix_keeps_control_to_label_binding() -> None:
    adapter, fake = adapter_for(
        logprobs_by_call=_two_field_logprobs(), served_template=_GEMMA3
    )
    response = adapter.judge(
        "Attached receipt.",
        _questions(),
        "gemma-mm",
        media=(_image(),),
    )
    route_prefix = fake.calls[1].prefix
    assert "Control 0 → billing: Money" in route_prefix
    assert "Control 1 → technical: Bugs" in route_prefix
    assert response.choices["route"].choice == "billing"
    assert response.nouls["flagged"].noul == pytest.approx(0.6)


@pytest.mark.contract
@pytest.mark.parametrize("media", [None, ()], ids=["none", "empty"])
def test_text_path_is_identical_when_media_is_absent(
    media: tuple[ImageInput, ...] | None,
) -> None:
    baseline_adapter, baseline_fake = adapter_for(
        logprobs_by_call=_two_field_logprobs()
    )
    baseline_adapter.judge("Charged twice.", _questions(), "gemma-mm")

    adapter, fake = adapter_for(logprobs_by_call=_two_field_logprobs())
    adapter.judge("Charged twice.", _questions(), "gemma-mm", media=media)

    assert [call.prefix for call in fake.calls] == [
        call.prefix for call in baseline_fake.calls
    ]
    for call in fake.calls:
        assert call.media == ()
        assert MEDIA_MARKER not in call.prefix


@pytest.mark.contract
@pytest.mark.parametrize(
    "judge",
    [ScoringJudgmentAdapter.judge, JudgmentPort.judge],
    ids=["adapter", "port"],
)
def test_media_is_keyword_only(judge: Callable[..., object]) -> None:
    parameter = signature(judge).parameters["media"]
    assert parameter.kind is Parameter.KEYWORD_ONLY
    assert parameter.default is None


@pytest.mark.contract
def test_media_aware_adapter_still_satisfies_judgment_port() -> None:
    adapter, fake = adapter_for(
        logprobs_by_call=_two_field_logprobs()[:1], served_template=_GEMMA3
    )

    def _accept(port: JudgmentPort) -> None:
        port.judge("state", {"flagged": Noul()}, "gemma-mm", media=(_image(),))

    _accept(adapter)
    assert fake.calls[0].media == (_image(),)
