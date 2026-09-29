"""Shared TPJEP live smoke acceptance checks (#149)."""

from __future__ import annotations

from typevet_evals.tpjep.records import TpjepRunSummary
from typevet_evals.tpjep.runner import TpjepRunReceipt

_EIGHT_TASKS = 8


def assert_tpjep_live_smoke_acceptance(summary: TpjepRunSummary) -> None:
    """Require eight answered, probability-valid attempts and zero failures."""
    failures: list[str] = []
    if summary.n_scheduled != _EIGHT_TASKS:
        failures.append(
            f"n_scheduled={summary.n_scheduled}, want {_EIGHT_TASKS}",
        )
    if summary.n_answered != _EIGHT_TASKS:
        failures.append(
            f"n_answered={summary.n_answered}, want {_EIGHT_TASKS}",
        )
    if summary.n_prob_valid != _EIGHT_TASKS:
        failures.append(
            f"n_prob_valid={summary.n_prob_valid}, want {_EIGHT_TASKS}",
        )
    if summary.n_schema_invalid:
        failures.append(f"n_schema_invalid={summary.n_schema_invalid}, want 0")
    if summary.n_prob_invalid:
        failures.append(f"n_prob_invalid={summary.n_prob_invalid}, want 0")
    if summary.n_transport_failed:
        failures.append(f"n_transport_failed={summary.n_transport_failed}, want 0")
    if summary.n_skipped:
        failures.append(f"n_skipped={summary.n_skipped}, want 0")
    if failures:
        msg = "TPJEP live smoke acceptance failed: " + "; ".join(failures)
        raise AssertionError(msg)


def assert_tpjep_live_smoke_receipt(receipt: TpjepRunReceipt) -> None:
    """Apply live smoke acceptance to a full run receipt."""
    assert_tpjep_live_smoke_acceptance(receipt.summary)
