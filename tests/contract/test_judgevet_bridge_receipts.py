"""Contract: the judgevet bridge returns the off-option receipt (#370).

The bridge reads ``off_option_threshold`` from its keyword or from the
judgevet ``provider_options`` mapping. It puts each typevet
``OffOptionReceipt`` into ``SystemOneResponse.receipts`` under the answer
name. It refuses any other option key. The async port takes the same
``provider_options`` (#380). An invalid keyword or option threshold is
refused before any backend call, and the message does not show the value.

Examples:
    ```bash
    uv run pytest -q tests/contract/test_judgevet_bridge_receipts.py
    ```

See Also:
    - [typevet.adapters.inbound.judgevet][]: The bridge under test
"""

from __future__ import annotations

import asyncio
import dataclasses
import math
from collections.abc import Mapping
from typing import Any

import pytest
from judgevet.domain.media import ImageAttachment, ImageEvidence, MediaCapabilities
from judgevet.domain.questions import Noul as JevNoul
from judgevet.media import judge_with_images
from judgevet.ports.options import (
    AsyncProviderOptionsSystemOnePort,
    ProviderOptionsMediaSystemOnePort,
    ProviderOptionsSystemOnePort,
)
from judgevet.providers import ProviderCapabilityError, ProviderRequestError

from tests.fixtures.judgevet_bridge import FAKE_MODEL, STATE, judgment_port
from typevet.adapters.inbound.judgevet import (
    AsyncTypevetSystemOnePort,
    TypevetMediaSystemOnePort,
    TypevetSystemOnePort,
)
from typevet.domain.judgment_questions import Question
from typevet.domain.judgment_response import JudgmentResponse
from typevet.domain.media import ImageInput
from typevet.testing import ScriptedScoringFake

pytestmark = pytest.mark.contract

_MASS = 0.3
_LOGPROBS = {"True": math.log(0.5), "False": math.log(0.2)}
"""Noul candidate scores; with the 0.3 off-option mass they sum to 1."""
_PNG = b"\x89PNG\r\n\x1a\nreceipts"
_PNG_CAPS = MediaCapabilities({"image/png"})


def _flagged(threshold: float) -> dict[str, float | bool | None]:
    """Return the receipt a 0.3 off-option mass gives at ``threshold``.

    Args:
        threshold: The threshold the bridge sent.

    Returns:
        The expected judgevet receipt mapping.
    """
    return {
        "off_option_mass": _MASS,
        "off_option_threshold": threshold,
        "off_option_flag": threshold < _MASS,
    }


def _bridge() -> TypevetSystemOnePort:
    """Build the text bridge over a scorer with a 0.3 off-option mass.

    Returns:
        The bridge port; it makes no network call.
    """
    scorer = ScriptedScoringFake(logprobs=_LOGPROBS, off_option_mass=_MASS)
    return TypevetSystemOnePort(judgment_port(scorer=scorer))


def _media_bridge() -> TypevetMediaSystemOnePort:
    """Build the media bridge over a scorer with a 0.3 off-option mass.

    Returns:
        The media bridge port; it makes no network call.
    """
    scorer = ScriptedScoringFake(logprobs=_LOGPROBS, off_option_mass=_MASS)
    return TypevetMediaSystemOnePort(
        judgment_port(media=True, scorer=scorer), media=_PNG_CAPS
    )


class _CountingPort:
    """``JudgmentPort`` that counts calls and can drop every receipt.

    Attributes:
        calls (int): Number of ``judge`` calls.
    """

    def __init__(self, *, drop_receipts: bool = False) -> None:
        self.calls = 0
        self._drop = drop_receipts
        scorer = ScriptedScoringFake(logprobs=_LOGPROBS, off_option_mass=_MASS)
        self._inner = judgment_port(scorer=scorer)

    def judge(
        self,
        state: str | dict[str, Any] | list[Any],
        questions: Mapping[str, Question | Mapping[str, Any]],
        model: str,
        *,
        media: tuple[ImageInput, ...] | None = None,
        off_option_threshold: float | None = None,
    ) -> JudgmentResponse:
        """Count the call, judge offline and drop the receipts when asked.

        Args:
            state: Content under evaluation.
            questions: Named questions.
            model: Model id.
            media: Images for every question.
            off_option_threshold: Threshold to forward.

        Returns:
            The offline response, with no receipts when ``drop_receipts``.
        """
        self.calls += 1
        response = self._inner.judge(
            state,
            questions,
            model,
            media=media,
            off_option_threshold=off_option_threshold,
        )
        if not self._drop:
            return response
        return dataclasses.replace(response, off_option={})


def test_keyword_threshold_gives_flagged_receipt() -> None:
    response = _bridge().system_one(
        STATE, {"q": JevNoul()}, FAKE_MODEL, off_option_threshold=0.25
    )
    assert response.receipts == {"q": _flagged(0.25)}


def test_provider_options_threshold_gives_flagged_receipt() -> None:
    response = _bridge().system_one(
        STATE,
        {"q": JevNoul()},
        FAKE_MODEL,
        provider_options={"off_option_threshold": 0.25},
    )
    assert response.receipts == {"q": _flagged(0.25)}


def test_keyword_wins_over_provider_options() -> None:
    response = _bridge().system_one(
        STATE,
        {"q": JevNoul()},
        FAKE_MODEL,
        off_option_threshold=0.35,
        provider_options={"off_option_threshold": 0.25},
    )
    assert response.receipts == {"q": _flagged(0.35)}
    assert response.receipts["q"]["off_option_flag"] is False


def test_no_threshold_keeps_mass_without_flag() -> None:
    response = _bridge().system_one(
        STATE, {"q": JevNoul()}, FAKE_MODEL, provider_options={}
    )
    assert response.receipts == {
        "q": {
            "off_option_mass": _MASS,
            "off_option_threshold": None,
            "off_option_flag": False,
        }
    }


def test_absent_receipt_leaves_the_name_absent() -> None:
    port = TypevetSystemOnePort(_CountingPort(drop_receipts=True))
    response = port.system_one(
        STATE, {"q": JevNoul()}, FAKE_MODEL, off_option_threshold=0.25
    )
    assert response.receipts == {}
    assert set(response.answers) == {"q"}


def test_unknown_option_key_is_refused_before_any_call() -> None:
    inner = _CountingPort()
    port = TypevetSystemOnePort(inner)
    with pytest.raises(ProviderCapabilityError):
        port.system_one(
            STATE, {"q": JevNoul()}, FAKE_MODEL, provider_options={"temperature": 0.1}
        )
    assert inner.calls == 0


@pytest.mark.parametrize("value", [1.5, -0.1, math.nan, True, "0.2", "secret-0.77"])
def test_invalid_option_threshold_is_refused_value_free(value: object) -> None:
    inner = _CountingPort()
    port = TypevetSystemOnePort(inner)
    with pytest.raises(ProviderRequestError) as caught:
        port.system_one(
            STATE,
            {"q": JevNoul()},
            FAKE_MODEL,
            provider_options={"off_option_threshold": value},
        )
    assert inner.calls == 0
    assert repr(value) not in str(caught.value)
    assert caught.value.__cause__ is None


def test_media_bridge_merges_receipts_from_every_group() -> None:
    evidence = ImageEvidence([ImageAttachment("a", _PNG, "image/png")], {"q": ["a"]})
    response = _media_bridge().system_one_media(
        STATE,
        {"q": JevNoul(), "r": JevNoul()},
        FAKE_MODEL,
        evidence=evidence,
        provider_options={"off_option_threshold": 0.25},
    )
    assert response.receipts == {"q": _flagged(0.25), "r": _flagged(0.25)}


def test_media_bridge_refuses_unknown_option_key() -> None:
    evidence = ImageEvidence([ImageAttachment("a", _PNG, "image/png")], {"q": ["a"]})
    with pytest.raises(ProviderCapabilityError):
        _media_bridge().system_one_media(
            STATE,
            {"q": JevNoul()},
            FAKE_MODEL,
            evidence=evidence,
            provider_options={"seed": 1},
        )


def test_judge_with_images_forwards_provider_options() -> None:
    evidence = ImageEvidence([ImageAttachment("a", _PNG, "image/png")], {"q": ["a"]})
    response = judge_with_images(
        _media_bridge(),
        STATE,
        {"q": JevNoul()},
        FAKE_MODEL,
        evidence=evidence,
        provider_options={"off_option_threshold": 0.25},
    )
    assert response.receipts == {"q": _flagged(0.25)}


def test_bridge_satisfies_the_provider_options_ports() -> None:
    port: ProviderOptionsSystemOnePort = _bridge()
    media_port: ProviderOptionsMediaSystemOnePort = _media_bridge()
    options = {"off_option_threshold": 0.25}
    evidence = ImageEvidence([ImageAttachment("a", _PNG, "image/png")], {"q": ["a"]})
    text = port.system_one(
        STATE, {"q": JevNoul()}, FAKE_MODEL, provider_options=options
    )
    media = media_port.system_one_media(
        STATE, {"q": JevNoul()}, FAKE_MODEL, evidence=evidence, provider_options=options
    )
    assert text.receipts == media.receipts == {"q": _flagged(0.25)}


def test_async_bridge_takes_provider_options() -> None:
    port = AsyncTypevetSystemOnePort(_bridge())
    response = asyncio.run(
        port.system_one(
            STATE,
            {"q": JevNoul()},
            FAKE_MODEL,
            provider_options={"off_option_threshold": 0.25},
        )
    )
    assert response.receipts == {"q": _flagged(0.25)}


def test_async_bridge_satisfies_the_async_options_port() -> None:
    port: AsyncProviderOptionsSystemOnePort = AsyncTypevetSystemOnePort(_bridge())
    response = asyncio.run(
        port.system_one(
            STATE,
            {"q": JevNoul()},
            FAKE_MODEL,
            provider_options={"off_option_threshold": 0.35},
        )
    )
    assert response.receipts == {"q": _flagged(0.35)}


@pytest.mark.parametrize("value", [1.5, -0.1, math.nan, 7.77])
def test_invalid_keyword_threshold_is_refused_value_free(value: float) -> None:
    inner = _CountingPort()
    port = TypevetSystemOnePort(inner)
    with pytest.raises(ProviderRequestError) as caught:
        port.system_one(STATE, {"q": JevNoul()}, FAKE_MODEL, off_option_threshold=value)
    assert inner.calls == 0
    assert repr(value) not in str(caught.value)
    assert caught.value.__cause__ is None


def test_invalid_option_refused_when_keyword_wins() -> None:
    inner = _CountingPort()
    port = TypevetSystemOnePort(inner)
    with pytest.raises(ProviderRequestError) as caught:
        port.system_one(
            STATE,
            {"q": JevNoul()},
            FAKE_MODEL,
            off_option_threshold=0.35,
            provider_options={"off_option_threshold": "secret-0.77"},
        )
    assert inner.calls == 0
    assert "secret-0.77" not in str(caught.value)
