"""Unit tests: consumer receipt acceptance checks ([#177][i177]).

Examples:
    ```bash
    uv run pytest -q tests/unit/test_psai_vision_consumer_receipt.py
    ```

See Also:
    - [typevet.evaluation.psai_vision_consumer_receipt][]: acceptance helper
"""

from __future__ import annotations

import pytest

from typevet.evaluation.psai_vision_consumer_receipt import (
    evaluate_consumer_receipt_acceptance,
)


def _minimal_pass_receipt() -> dict[str, object]:
    return {
        "judgment_call_count": 1,
        "capability": {"vision": True},
        "unsupported_capability_negative": {"ok": True},
        "paired_ordering": [{"unique_data_id": "u1", "ordered": True}],
        "matrix_rows": [{"answers": {"q": {"kind": "Noul", "noul": 0.5}}}],
    }


@pytest.mark.unit
def test_evaluate_receipt_fails_when_capability_vision_false() -> None:
    """``capability.vision`` false adds a failure message."""
    receipt = _minimal_pass_receipt()
    receipt["capability"] = {"vision": False}
    accepted, failures = evaluate_consumer_receipt_acceptance(receipt)
    assert not accepted
    assert any("capability.vision" in msg for msg in failures)


@pytest.mark.unit
def test_evaluate_receipt_fails_when_unsupported_capability_negative_not_ok() -> None:
    """Negative control must report ``ok`` true."""
    receipt = _minimal_pass_receipt()
    receipt["unsupported_capability_negative"] = {"ok": False}
    accepted, failures = evaluate_consumer_receipt_acceptance(receipt)
    assert not accepted
    assert any("unsupported_capability_negative" in msg for msg in failures)


@pytest.mark.unit
def test_evaluate_receipt_fails_when_matrix_row_missing_answers() -> None:
    """Every matrix row must serialize a non-empty ``answers`` map."""
    receipt = _minimal_pass_receipt()
    receipt["matrix_rows"] = [{"leg": "annotation"}]
    accepted, failures = evaluate_consumer_receipt_acceptance(receipt)
    assert not accepted
    assert any("missing answers" in msg for msg in failures)


@pytest.mark.unit
def test_evaluate_receipt_fails_when_paired_ordering_not_ordered() -> None:
    """Paired ordering failures reject the receipt."""
    receipt = _minimal_pass_receipt()
    receipt["paired_ordering"] = [{"unique_data_id": "u1", "ordered": False}]
    accepted, failures = evaluate_consumer_receipt_acceptance(receipt)
    assert not accepted
    assert any("paired_ordering" in msg for msg in failures)


@pytest.mark.unit
def test_evaluate_receipt_passes_minimal_valid_payload() -> None:
    """Minimal valid receipt accepts with no failure messages."""
    accepted, failures = evaluate_consumer_receipt_acceptance(_minimal_pass_receipt())
    assert accepted
    assert failures == []
