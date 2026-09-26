"""Unit tests: fail-closed consumer receipt acceptance ([#177][i177]).

Examples:
    ```bash
    uv run pytest -q tests/unit/test_psai_vision_consumer_failclosed.py
    ```

See Also:
    - [typevet.evaluation.psai_vision_consumer_receipt_structure][]: structural checks
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from typevet.domain.candidate_scoring_request import (
    CandidateScoringRequest,
    CandidateTokenSpec,
)
from typevet.domain.scoring_stage import ScoreStage
from typevet.evaluation.psai_vision_consumer_dispatch import (
    ConsumerDispatchLedger,
    wrap_scoring_port,
)
from typevet.evaluation.psai_vision_consumer_harness import run_offline_consumer_proof
from typevet.evaluation.psai_vision_consumer_outcomes import expected_outcome_failures
from typevet.evaluation.psai_vision_consumer_receipt import (
    evaluate_consumer_receipt_acceptance,
)
from typevet.evaluation.psai_vision_consumer_receipt_structure import (
    protocol_structural_failures,
)
from typevet.testing import ScriptedScoringFake

FIXTURE_ROOT = (
    Path(__file__).resolve().parents[1] / "fixtures" / "psai" / "vision_smoke"
)
REV2_RECEIPT = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "consumer"
    / "consumer-receipt-p2-b01ef692945a65e3-gemma-4-31b-kv9-q4km-mm.json"
)


@pytest.mark.unit
def test_empty_receipt_rejects() -> None:
    """Empty mapping fails structural acceptance."""
    accepted, failures = evaluate_consumer_receipt_acceptance({})
    assert not accepted
    assert "receipt is empty" in failures


@pytest.mark.unit
def test_rev2_saved_receipt_structural_checks_only() -> None:
    """Committed rev2 wheel receipt passes structural checks (pins on disk)."""
    receipt = json.loads(REV2_RECEIPT.read_text(encoding="utf-8"))
    assert protocol_structural_failures(receipt) == []


@pytest.mark.unit
def test_rev2_receipt_outcomes_without_local_fixture_root() -> None:
    """Gold checks use manifest pins when fixture_root is machine-local."""
    receipt = json.loads(REV2_RECEIPT.read_text(encoding="utf-8"))
    receipt["fixture_root"] = "/nonexistent/path"
    outcome_failures = expected_outcome_failures(receipt)
    assert outcome_failures == []


@pytest.mark.unit
def test_rev2_full_acceptance_without_local_fixture_root() -> None:
    """End-to-end acceptance uses pins when ``fixture_root`` is not on disk."""
    receipt = json.loads(REV2_RECEIPT.read_text(encoding="utf-8"))
    receipt["fixture_root"] = "/nonexistent/path"
    accepted, failures = evaluate_consumer_receipt_acceptance(receipt)
    assert accepted
    assert failures == []


@pytest.mark.unit
def test_missing_manifest_pin_rejects_via_evaluate() -> None:
    """Structural acceptance rejects receipts with no ``manifest_sha256`` pin."""
    result = run_offline_consumer_proof(fixture_root=FIXTURE_ROOT)
    receipt = copy.deepcopy(result.receipt)
    receipt.pop("manifest_sha256", None)
    accepted, failures = evaluate_consumer_receipt_acceptance(receipt)
    assert not accepted
    assert any("manifest_sha256" in msg for msg in failures)


@pytest.mark.unit
def test_scoring_observed_mismatch_rejects() -> None:
    """Declared scoring count must match instrumented ``scoring_requests_observed``."""
    result = run_offline_consumer_proof(fixture_root=FIXTURE_ROOT)
    receipt = copy.deepcopy(result.receipt)
    receipt["scoring_requests_observed"] = 1
    accepted, failures = evaluate_consumer_receipt_acceptance(receipt)
    assert not accepted
    assert any("scoring_requests_observed" in msg for msg in failures)


@pytest.mark.unit
def test_unknown_matrix_uid_rejects() -> None:
    """Replacing matrix uids with unknown ids fails acceptance."""
    result = run_offline_consumer_proof(fixture_root=FIXTURE_ROOT)
    receipt = copy.deepcopy(result.receipt)
    receipt["matrix_rows"][0]["unique_data_id"] = "not-a-frozen-case"
    accepted, failures = evaluate_consumer_receipt_acceptance(receipt)
    assert not accepted
    assert any("unknown unique_data_id" in msg for msg in failures)


@pytest.mark.unit
def test_stripped_matrix_with_zero_observed_rejects() -> None:
    """Empty matrix with zero observed counts fails closed."""
    result = run_offline_consumer_proof(fixture_root=FIXTURE_ROOT)
    receipt = copy.deepcopy(result.receipt)
    receipt["matrix_rows"] = []
    receipt["paired_ordering"] = []
    receipt["judgment_call_count"] = 0
    receipt["scoring_request_count"] = 0
    receipt["scoring_requests_observed"] = 0
    accepted, failures = evaluate_consumer_receipt_acceptance(receipt)
    assert not accepted
    assert failures


@pytest.mark.unit
def test_scoring_wrapper_increments_ledger() -> None:
    """Wrapped scoring port increments the dispatch ledger."""
    ledger = ConsumerDispatchLedger()
    fake = ScriptedScoringFake(logprobs={"True": -0.2, "False": -1.0})
    port = wrap_scoring_port(fake, ledger)
    request = CandidateScoringRequest(
        model="m",
        prefix="p",
        candidates=(CandidateTokenSpec("True", (1,)),),
        stage=ScoreStage.PRE_SAMPLING,
    )
    port.score_candidates(request)
    assert ledger.scoring_requests == 1
