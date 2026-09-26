"""Non-live regression for TPJEP live smoke acceptance (#149)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import pytest

from tests.fixtures.tpjep.live_acceptance import assert_tpjep_live_smoke_acceptance
from typevet.eval_tpjep_records import (
    TPJEP_PROTOCOL_V0,
    TpjepAttemptRecord,
    TpjepRunSummary,
    summarize_tpjep_records,
)

PINS = {
    "dataset_git_commit": "f8ce71361165846101d02ebc83ad44e47ae44fc3",
    "dataset_hash_recipe": "typellm_manifest_sha256",
    "dataset_hash": "dc3995d8ae1e2fc8e81ce38431add300eb8bb39b85aadfd0c7c32079382dde51",
    "model": "gemma-test",
}

_ANSWERED = TpjepAttemptRecord(
    task_id="task-0",
    source_tier="easy",
    question_type="Noul",
    model=PINS["model"],
    dataset_git_commit=PINS["dataset_git_commit"],
    dataset_hash_recipe=PINS["dataset_hash_recipe"],
    dataset_hash=PINS["dataset_hash"],
    protocol=TPJEP_PROTOCOL_V0,
    outcome="answered",
    predicted=True,
    expected=True,
    probabilities={"yes": 0.6, "no": 0.4},
    prob_valid=True,
    correct=True,
    error_type=None,
    error_message=None,
    usage={"prompt_tokens": 10, "completion_tokens": 2},
    duration_ms=120,
    server_build="build-1",
    template_class="gemma",
)


def _answered_row(index: int, *, correct: bool = True) -> TpjepAttemptRecord:
    return replace(
        _ANSWERED,
        task_id=f"task-{index}",
        correct=correct,
        predicted=correct,
    )


def _summary_from(*rows: TpjepAttemptRecord) -> TpjepRunSummary:
    return summarize_tpjep_records(list(rows))


@pytest.mark.unit
def test_live_acceptance_accepts_eight_valid_imperfect_accuracy() -> None:
    rows = [_answered_row(i, correct=(i % 3 == 0)) for i in range(8)]
    summary = _summary_from(*rows)
    assert summary.n_correct == 3
    assert summary.accuracy_on_prob_valid == 0.375
    assert_tpjep_live_smoke_acceptance(summary)


@pytest.mark.unit
@pytest.mark.parametrize(
    "factory",
    [
        lambda: [
            replace(
                _ANSWERED,
                task_id=f"task-{i}",
                outcome="transport_failed",
                predicted=None,
                prob_valid=False,
                correct=None,
                error_type="TransportError",
                error_message="reset",
            )
            for i in range(8)
        ],
        lambda: (
            [_answered_row(i) for i in range(7)]
            + [
                replace(
                    _ANSWERED,
                    task_id="task-7",
                    outcome="schema_invalid",
                    predicted=None,
                    prob_valid=False,
                    correct=None,
                    error_type="SchemaValidationError",
                    error_message="bad",
                )
            ]
        ),
        lambda: (
            [_answered_row(i) for i in range(7)]
            + [
                replace(
                    _ANSWERED,
                    task_id="task-7",
                    outcome="prob_invalid",
                    predicted="x",
                    prob_valid=False,
                    correct=None,
                    error_type="ProbInvalid",
                    error_message="sum",
                )
            ]
        ),
    ],
    ids=["all-failed", "one-failed", "probability-invalid"],
)
def test_live_acceptance_rejects_bad_receipts(
    factory: Callable[[], list[TpjepAttemptRecord]],
) -> None:
    summary = _summary_from(*factory())
    with pytest.raises(AssertionError):
        assert_tpjep_live_smoke_acceptance(summary)
