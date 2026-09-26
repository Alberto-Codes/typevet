"""Unit tests: consumer expected-outcome checks ([#177][i177]).

Examples:
    ```bash
    uv run pytest -q tests/unit/test_psai_vision_consumer_outcomes.py
    ```

See Also:
    - [typevet.evaluation.psai_vision_consumer_outcomes][]: outcome helper
"""

from __future__ import annotations

from pathlib import Path

import pytest

from typevet.evaluation.psai_vision_consumer_harness import run_offline_consumer_proof
from typevet.evaluation.psai_vision_consumer_outcomes import expected_outcome_failures
from typevet.evaluation.psai_vision_consumer_receipt import (
    evaluate_consumer_receipt_acceptance,
)

FIXTURE_ROOT = (
    Path(__file__).resolve().parents[1] / "fixtures" / "psai" / "vision_smoke"
)


@pytest.mark.unit
def test_offline_receipt_expected_outcomes_pass() -> None:
    """Scripted offline matrix satisfies gold polarity and annotation answers."""
    result = run_offline_consumer_proof(fixture_root=FIXTURE_ROOT)
    assert expected_outcome_failures(result.receipt) == []
    accepted, failures = evaluate_consumer_receipt_acceptance(result.receipt)
    assert accepted
    assert failures == []
