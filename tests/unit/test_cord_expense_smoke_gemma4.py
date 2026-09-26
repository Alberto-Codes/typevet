"""Offline Gemma 4 CORD expense smoke regression ([#185][i185]).

Examples:
    ```bash
    uv run pytest -q tests/unit/test_cord_expense_smoke_gemma4.py
    ```

See Also:
    - [typevet.evaluation.cord_expense_smoke][]: attachment floors and capability
    - [typevet.evaluation.cord_semantic_acceptance][]: #161 revision 1 floors

[i185]: https://github.com/Alberto-Codes/typevet/issues/185
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from typevet.evaluation.cord_expense_smoke import (
    GEMMA4_DIRECT_RECEIPT_MODEL,
    GEMMA4_NATIVE_TURN,
    assert_cord_expense_attachment,
    cord_combined_attachment_floor,
    cord_image_only_attachment_gap_floor,
    measured_image_prompt_tokens,
    resolve_cord_expense_attachment_profile,
    validate_gemma4_smoke_capability,
)
from typevet.evaluation.cord_semantic_acceptance import (
    ACCURACY_CHECK,
    CONTRADICTED_RECALL_CHECK,
    CheckStatus,
    accept_combined_receipt,
)
from typevet.evaluation.datasets.cord_expense import load_expense_cases

pytestmark = pytest.mark.unit

_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "cord"
    / "expense_smoke"
    / "gemma4_kv9_direct_receipt.json"
)
_MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "cord"
    / "expense_smoke"
    / "manifest.json"
)


def _load_receipt() -> dict[str, Any]:
    return json.loads(_FIXTURE.read_text(encoding="utf-8"))


def _capability_receipt(**overrides: object) -> dict[str, Any]:
    """Minimal receipt mapping for ``validate_gemma4_smoke_capability`` tests.

    Other Parameters:
        **overrides: Receipt fields to replace on the Gemma 4 capability baseline.

    Returns:
        Receipt mapping with ``model``, ``vision`` and ``served_template`` set.
    """
    base: dict[str, Any] = {
        "model": GEMMA4_DIRECT_RECEIPT_MODEL,
        "vision": True,
        "served_template": GEMMA4_NATIVE_TURN,
    }
    base.update(overrides)
    return base


def _claim_and_receipt_ids() -> tuple[tuple[str, ...], tuple[str, ...]]:
    cases = load_expense_cases(_MANIFEST.read_text(encoding="utf-8"))
    claim_ids = tuple(case.claim_id for case in cases)
    receipt_ids = tuple(dict.fromkeys(case.receipt_id for case in cases))
    return claim_ids, receipt_ids


def test_gemma4_attachment_floors_use_measured_image_costs() -> None:
    """Gemma 3 and Gemma 4 ids carry different measured image prompt costs."""
    assert measured_image_prompt_tokens("gemma-3-4b-it-q4km-mm") == 256
    assert measured_image_prompt_tokens(GEMMA4_DIRECT_RECEIPT_MODEL) == 245
    assert cord_combined_attachment_floor(GEMMA4_DIRECT_RECEIPT_MODEL) == 214
    assert cord_image_only_attachment_gap_floor("gemma-3-4b-it-q4km-mm") == 225


def test_unsupported_model_id_has_no_attachment_floor() -> None:
    """Unknown multimodal ids fail closed before a live smoke run."""
    with pytest.raises(ValueError, match="gemma-3 and gemma-4"):
        measured_image_prompt_tokens("llama-vision-mm")


def test_vendored_gemma4_receipt_records_native_turn_capability() -> None:
    """The pinned receipt declares vision on the native Gemma 4 turn path."""
    receipt = _load_receipt()
    validate_gemma4_smoke_capability(receipt)
    assert receipt["model"] == GEMMA4_DIRECT_RECEIPT_MODEL


def test_gemma4_capability_rejects_vision_false() -> None:
    """Unsupported capability must fail closed before a live smoke run."""
    with pytest.raises(ValueError, match="vision=true"):
        validate_gemma4_smoke_capability(_capability_receipt(vision=False))


def test_gemma4_capability_rejects_degraded_chatml_served_template() -> None:
    """ChatML served family is not the Gemma 4 native turn contract."""
    with pytest.raises(ValueError, match=GEMMA4_NATIVE_TURN):
        validate_gemma4_smoke_capability(
            _capability_receipt(served_template="degraded_chatml"),
        )


def test_gemma4_capability_rejects_native_gemma3_turn() -> None:
    """Wrong native turn family must not pass as Gemma 4 capability."""
    with pytest.raises(ValueError, match=GEMMA4_NATIVE_TURN):
        validate_gemma4_smoke_capability(
            _capability_receipt(served_template="native_gemma3_turn"),
        )


def _image_only_omission_tokens(receipt: dict[str, Any]) -> int:
    image_only = receipt["image_only"]
    if isinstance(image_only, dict) and "omission_tokens_evaluated" in image_only:
        value = image_only["omission_tokens_evaluated"]
        assert isinstance(value, int)
        return value
    profile = resolve_cord_expense_attachment_profile(
        str(receipt["model"]),
        str(receipt["served_template"]),
    )
    return profile.image_only_state_omission_tokens


def test_vendored_gemma4_receipt_passes_offline_attachment_gate() -> None:
    """Saved token counts meet the Gemma 4 attachment floors."""
    receipt = _load_receipt()
    claim_ids, receipt_ids = _claim_and_receipt_ids()
    image_rows = receipt["image_only"]["rows"]
    profile = resolve_cord_expense_attachment_profile(
        str(receipt["model"]),
        str(receipt["served_template"]),
    )
    assert_cord_expense_attachment(
        profile=profile,
        text_only=receipt["text_only"],
        image_only=image_rows,
        combined=receipt["combined"],
        claim_ids=claim_ids,
        receipt_ids=receipt_ids,
        image_only_omission_tokens=_image_only_omission_tokens(receipt),
    )


def test_vendored_gemma4_combined_arm_fails_issue_161_answerable_and_contradicted() -> (
    None
):
    """The direct combined arm misses two #161 revision 1 floors on Gemma 4."""
    outcome = accept_combined_receipt(_load_receipt())
    assert not outcome.accepted
    accuracy = outcome.check(ACCURACY_CHECK)
    contradicted = outcome.check(CONTRADICTED_RECALL_CHECK)
    assert accuracy.status is CheckStatus.FAIL
    assert contradicted.status is CheckStatus.FAIL
    assert accuracy.measured == pytest.approx(0.25)
    assert contradicted.measured == pytest.approx(0.3333333333333333)
