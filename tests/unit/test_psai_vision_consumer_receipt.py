"""Unit tests: consumer receipt acceptance checks ([#177][i177]).

Examples:
    ```bash
    uv run pytest -q tests/unit/test_psai_vision_consumer_receipt.py
    ```

See Also:
    - [typevet.evaluation.psai_vision_consumer_receipt][]: acceptance helper
"""

from __future__ import annotations

from pathlib import Path

import pytest

from typevet.evaluation.psai_vision_consumer_harness import run_offline_consumer_proof
from typevet.evaluation.psai_vision_consumer_receipt import (
    evaluate_consumer_receipt_acceptance,
)

FIXTURE_ROOT = (
    Path(__file__).resolve().parents[1] / "fixtures" / "psai" / "vision_smoke"
)


@pytest.mark.unit
def test_evaluate_receipt_fails_when_capability_vision_false() -> None:
    """``capability.vision`` false adds a failure message."""
    result = run_offline_consumer_proof(fixture_root=FIXTURE_ROOT)
    receipt = dict(result.receipt)
    receipt["capability"] = {"vision": False}
    accepted, failures = evaluate_consumer_receipt_acceptance(receipt)
    assert not accepted
    assert any("capability.vision" in msg for msg in failures)


@pytest.mark.unit
def test_evaluate_receipt_fails_when_unsupported_capability_negative_not_ok() -> None:
    """Negative control must report ``ok`` true."""
    result = run_offline_consumer_proof(fixture_root=FIXTURE_ROOT)
    receipt = dict(result.receipt)
    receipt["unsupported_capability_negative"] = {"ok": False}
    accepted, failures = evaluate_consumer_receipt_acceptance(receipt)
    assert not accepted
    assert any("unsupported_capability_negative" in msg for msg in failures)


@pytest.mark.unit
def test_evaluate_receipt_fails_when_matrix_row_missing_answers() -> None:
    """Every matrix row must serialize a non-empty ``answers`` map."""
    result = run_offline_consumer_proof(fixture_root=FIXTURE_ROOT)
    receipt = dict(result.receipt)
    receipt["matrix_rows"] = list(receipt["matrix_rows"])
    receipt["matrix_rows"][0] = {"leg": "visual", "unique_data_id": "x"}
    accepted, failures = evaluate_consumer_receipt_acceptance(receipt)
    assert not accepted
    assert failures


@pytest.mark.unit
def test_evaluate_receipt_fails_when_paired_ordering_not_ordered() -> None:
    """Paired ordering failures reject the receipt."""
    result = run_offline_consumer_proof(fixture_root=FIXTURE_ROOT)
    receipt = dict(result.receipt)
    pairs = list(receipt["paired_ordering"])
    pairs[0] = dict(pairs[0])
    pairs[0]["ordered"] = False
    receipt["paired_ordering"] = pairs
    accepted, failures = evaluate_consumer_receipt_acceptance(receipt)
    assert not accepted
    assert any("paired_ordering" in msg for msg in failures)


@pytest.mark.unit
def test_evaluate_receipt_passes_offline_harness_payload() -> None:
    """Offline harness receipt accepts with no failure messages."""
    result = run_offline_consumer_proof(fixture_root=FIXTURE_ROOT)
    accepted, failures = evaluate_consumer_receipt_acceptance(result.receipt)
    assert accepted
    assert failures == []
