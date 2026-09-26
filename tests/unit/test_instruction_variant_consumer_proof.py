"""Unit tests: instruction-variant consumer proof ([#177][i177]).

Examples:
    ```bash
    uv run pytest -q tests/unit/test_instruction_variant_consumer_proof.py
    ```
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from typevet.domain.errors import JudgmentValidationError
from typevet.domain.judgment_response import JudgmentResponse
from typevet.evaluation.instruction_variant_consumer_live import (
    run_live_instruction_variant_proof,
)
from typevet.evaluation.instruction_variant_consumer_matrix import (
    _LedgerJudgmentPort,
    probe_invalid_model,
)
from typevet.evaluation.instruction_variant_consumer_offline import (
    run_offline_instruction_variant_proof,
)
from typevet.evaluation.instruction_variant_consumer_proof import proof_main
from typevet.evaluation.instruction_variant_consumer_protocol import (
    FROZEN_VARIANT_JUDGMENT_CALLS,
    FROZEN_VARIANT_SCORING_REQUESTS,
    VariantDispatchLedger,
)
from typevet.evaluation.instruction_variant_consumer_receipt import (
    acceptance_failures,
    comparison_verdict,
)
from typevet.evaluation.instruction_variant_consumer_run import run_variant_matrix
from typevet.evaluation.outcome_replay_metrics import replay_identical_reports
from typevet.ports.judgment import JudgmentPort

FIXTURE_ROOT = (
    Path(__file__).resolve().parents[1] / "fixtures" / "psai" / "vision_smoke"
)


@pytest.mark.unit
def test_offline_instruction_variant_proof_passes() -> None:
    """Offline slice retains probes, replay metrics, and honest verdict."""
    result = run_offline_instruction_variant_proof(fixture_root=FIXTURE_ROOT)
    assert result.exit_code == 0
    receipt = result.receipt
    assert receipt["failed_attempts"] >= 1
    assert receipt["invalid_model_probe"]["ok"] is True
    assert receipt["unsupported_template_probe"]["ok"] is True
    assert receipt["scoring_requests_observed"] == 4
    assert receipt["acceptance_failures"] == []
    assert replay_identical_reports(
        result.replay_report,
        receipt["replay_report"],
    )
    assert receipt["comparison_verdict"] in {
        "candidate_improved",
        "tie",
        "candidate_worse_or_mixed",
        "insufficient_valid_metrics",
    }


@pytest.mark.unit
def test_replay_does_not_claim_improvement_without_both_metrics() -> None:
    """Scripted offline arms use mixed deltas; verdict must stay honest."""
    result = run_offline_instruction_variant_proof(fixture_root=FIXTURE_ROOT)
    improved = result.replay_report.get("candidate_improved")
    if improved is False:
        assert result.receipt["comparison_verdict"] != "candidate_improved"


@pytest.mark.unit
def test_comparison_verdict_branches() -> None:
    """Verdict helper stays honest across replay shapes."""
    assert comparison_verdict({"candidate_improved": True}) == "candidate_improved"
    assert (
        comparison_verdict({"delta_candidate_minus_seed": {}})
        == "insufficient_valid_metrics"
    )
    assert (
        comparison_verdict(
            {"delta_candidate_minus_seed": {"mean_brier": 0.0, "mean_log_loss": 0.0}}
        )
        == "tie"
    )
    assert (
        comparison_verdict(
            {"delta_candidate_minus_seed": {"mean_brier": 0.1, "mean_log_loss": 0.0}}
        )
        == "candidate_worse_or_mixed"
    )


@pytest.mark.unit
def test_acceptance_failures_lists_each_blocker() -> None:
    """Acceptance failures name every failed gate."""
    ledger = VariantDispatchLedger()
    failures = acceptance_failures(
        invalid={"ok": False},
        negative={"ok": False},
        budget_ok=False,
        budget_error="budget",
        ledger=ledger,
        replay_check=False,
    )
    assert "invalid model probe" in failures[0]
    assert "unsupported template probe" in failures[1]
    assert failures[2] == "budget"
    assert "replay metrics not idempotent" in failures[3]
    assert "failed_attempt" in failures[4]


@pytest.mark.unit
def test_variant_ledger_enforces_budgets() -> None:
    """Ledger rejects calls beyond frozen caps."""
    ledger = VariantDispatchLedger()
    for _ in range(FROZEN_VARIANT_JUDGMENT_CALLS):
        ledger.before_judgment()
    with pytest.raises(ValueError, match="judgment calls"):
        ledger.before_judgment()
    for _ in range(FROZEN_VARIANT_SCORING_REQUESTS):
        ledger.before_scoring()
    with pytest.raises(ValueError, match="scoring requests"):
        ledger.before_scoring()


@pytest.mark.unit
def test_ledger_judgment_port_records_failures() -> None:
    """Wrapped port counts validation failures before re-raising."""

    class _RaisingPort(JudgmentPort):
        def judge(self, state, questions, model, *, media=None):
            """Always fail validation for ledger accounting tests.

            Raises:
                JudgmentValidationError: Always raised for this fake port.
            """
            raise JudgmentValidationError("probe")

    ledger = VariantDispatchLedger()
    port = _LedgerJudgmentPort(_RaisingPort(), ledger)
    with pytest.raises(JudgmentValidationError):
        port.judge("s", {}, "m")
    assert ledger.failed_attempts == 1


@pytest.mark.unit
def test_proof_main_writes_receipt(tmp_path: Path) -> None:
    """CLI shim emits JSON summary and optional receipt file."""
    out = tmp_path / "receipt.json"
    code = proof_main(
        [
            "--fixture-root",
            str(FIXTURE_ROOT),
            "--out",
            str(out),
            "--wheel-sha256",
            "abc",
        ]
    )
    assert code == 0
    assert out.is_file()
    summary = json.loads(out.read_text(encoding="utf-8"))
    assert summary["comparison_verdict"]


@pytest.mark.unit
def test_run_live_instruction_variant_proof_requires_live_env() -> None:
    """Live path exits invalid when ``TYPEVET_REQUIRE_LIVE`` is unset."""
    with patch(
        "typevet.evaluation.instruction_variant_consumer_live.require_live_enabled",
        return_value=False,
    ):
        result = run_live_instruction_variant_proof(fixture_root=FIXTURE_ROOT)
    assert result.exit_code == 2
    assert result.receipt["acceptance_failures"]


@pytest.mark.unit
def test_run_live_instruction_variant_proof_when_gate_blocks() -> None:
    """Live path exits invalid when router gate skips."""
    with (
        patch(
            "typevet.evaluation.instruction_variant_consumer_live.require_live_enabled",
            return_value=True,
        ),
        patch(
            "typevet.evaluation.instruction_variant_consumer_live.run_live_variant_matrix",
            return_value=None,
        ),
        patch(
            "typevet.evaluation.instruction_variant_consumer_live.live_skip_reason",
            return_value="live gate blocked",
        ),
    ):
        result = run_live_instruction_variant_proof(fixture_root=FIXTURE_ROOT)
    assert result.exit_code == 2


@pytest.mark.unit
def test_run_live_instruction_variant_proof_offline_shaped_matrix() -> None:
    """Successful live run tags receipt and reuses finalize path."""
    matrix = run_variant_matrix(
        fixture_root=FIXTURE_ROOT,
        seed_instruction="seed q",
        candidate_instruction="candidate q",
        model_id="unit-fake",
    )
    with (
        patch(
            "typevet.evaluation.instruction_variant_consumer_live.require_live_enabled",
            return_value=True,
        ),
        patch(
            "typevet.evaluation.instruction_variant_consumer_live.run_live_variant_matrix",
            return_value=matrix,
        ),
    ):
        result = run_live_instruction_variant_proof(fixture_root=FIXTURE_ROOT)
    assert result.exit_code == 0
    assert result.receipt["evidence_kind"] == "instruction_variant_live"
    assert result.receipt["acceptance_failures"] == []


@pytest.mark.unit
def test_probe_invalid_model_fails_open_without_validation_error() -> None:
    """Invalid-model probe reports failure when port accepts empty model."""

    class _AcceptingPort(JudgmentPort):
        def judge(self, state, questions, model, *, media=None):
            """Accept empty model to simulate fail-open probe.

            Returns:
                Empty judgment response without validation failure.
            """
            return JudgmentResponse(model=model, answers={})

    probe = probe_invalid_model(_AcceptingPort())
    assert probe["ok"] is False
