"""Offline CORD expense live harness gate and attachment controls ([#185][i185]).

Examples:
    ```bash
    uv run pytest -q tests/unit/test_cord_expense_smoke_harness_gate.py
    ```

See Also:
    - [typevet.evaluation.cord_expense_smoke][]: capability gate and attachment
    - [tests.live.test_cord_expense_smoke_live][]: opt-in live wiring

[i185]: https://github.com/Alberto-Codes/typevet/issues/185
"""

from __future__ import annotations

from typing import Any

import pytest

from typevet.evaluation.cord_expense_smoke import (
    GEMMA3_DIRECT_RECEIPT_MODEL,
    GEMMA3_NATIVE_TURN,
    GEMMA4_DIRECT_RECEIPT_MODEL,
    GEMMA4_NATIVE_TURN,
    CordExpenseLiveSmokeGateError,
    assert_cord_expense_attachment,
    assert_cord_expense_live_smoke_gate,
    cord_image_only_attachment_gap_floor,
    resolve_cord_expense_attachment_profile,
)

pytestmark = pytest.mark.unit

_SCORING_PATH: list[str] = []


def _record_scoring_after_gate(model_id: str) -> None:
    """Simulate the live harness entering scoring only after the gate passes."""
    assert_cord_expense_live_smoke_gate(
        vision=True,
        served_template=GEMMA4_NATIVE_TURN,
        model_id=model_id,
    )
    _SCORING_PATH.append(model_id)


def test_live_harness_gate_native_gemma4_reaches_scoring() -> None:
    """Gemma 4 on the native turn must pass the capability gate before scoring."""
    _SCORING_PATH.clear()
    _record_scoring_after_gate(GEMMA4_DIRECT_RECEIPT_MODEL)
    assert _SCORING_PATH == [GEMMA4_DIRECT_RECEIPT_MODEL]


def test_live_harness_gate_native_gemma3_reaches_scoring() -> None:
    """Gemma 3 on the native turn remains compatible with the live gate."""
    profile = assert_cord_expense_live_smoke_gate(
        vision=True,
        served_template=GEMMA3_NATIVE_TURN,
        model_id=GEMMA3_DIRECT_RECEIPT_MODEL,
    )
    assert profile.served_template == GEMMA3_NATIVE_TURN


def test_live_harness_gate_rejects_non_vision_before_scoring() -> None:
    """Text-only capability must fail before any scoring call."""
    _SCORING_PATH.clear()
    with pytest.raises(CordExpenseLiveSmokeGateError, match="text-only"):
        assert_cord_expense_live_smoke_gate(
            vision=False,
            served_template=GEMMA4_NATIVE_TURN,
            model_id=GEMMA4_DIRECT_RECEIPT_MODEL,
        )
    assert _SCORING_PATH == []


def test_live_harness_gate_rejects_degraded_chatml_before_scoring() -> None:
    """Unsupported served families must fail before scoring."""
    with pytest.raises(CordExpenseLiveSmokeGateError, match="native Gemma"):
        assert_cord_expense_live_smoke_gate(
            vision=True,
            served_template="degraded_chatml",
            model_id=GEMMA4_DIRECT_RECEIPT_MODEL,
        )


def test_live_harness_gate_must_accept_gemma4_not_reject() -> None:
    """Regression: the gate must not wrongly reject native Gemma 4 (#185 red)."""
    profile = assert_cord_expense_live_smoke_gate(
        vision=True,
        served_template=GEMMA4_NATIVE_TURN,
        model_id=GEMMA4_DIRECT_RECEIPT_MODEL,
    )
    assert profile.model_id == GEMMA4_DIRECT_RECEIPT_MODEL


def test_unknown_attachment_profile_fails_closed() -> None:
    """Attachment evidence rejects unverified model and template pairs."""
    with pytest.raises(ValueError, match="verified"):
        resolve_cord_expense_attachment_profile(
            "llama-vision-mm",
            GEMMA4_NATIVE_TURN,
        )


def _minimal_rows(tokens: int) -> dict[str, Any]:
    return {"tokens_evaluated": tokens}


def test_image_only_attachment_rejects_long_text_without_media_gap() -> None:
    """Long text without an image must not pass the image_only gap check."""
    profile = resolve_cord_expense_attachment_profile(
        GEMMA4_DIRECT_RECEIPT_MODEL,
        "native_gemma4_turn",
    )
    omission = 520
    dropped_media_tokens = 520
    assert dropped_media_tokens >= cord_image_only_attachment_gap_floor(
        GEMMA4_DIRECT_RECEIPT_MODEL
    )
    with pytest.raises(ValueError, match="image was not attached"):
        assert_cord_expense_attachment(
            profile=profile,
            text_only={"C1": _minimal_rows(200)},
            image_only={"R01": _minimal_rows(dropped_media_tokens)},
            combined={"C1": _minimal_rows(900)},
            claim_ids=("C1",),
            receipt_ids=("R01",),
            image_only_omission_tokens=omission,
        )
