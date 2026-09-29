"""Unit tests: PSAI consumer judgment vs scoring accounting ([#177][i177])."""

from __future__ import annotations

import math

import pytest

from tests.fixtures.judgment_scoring_contract import adapter_for
from typevet_evals.datasets.psai_vision_controls import annotation_questions
from typevet_evals.psai_vision_consumer.accounting import (
    ANNOTATION_QUESTIONS_PER_JUDGE_CALL,
    ConsumerCallBudgetError,
    ConsumerCallCounts,
    enforce_consumer_call_budget,
    plan_frozen_consumer_calls,
)


@pytest.mark.unit
def test_frozen_protocol_schedules_fourteen_judgments_and_sixteen_scoring_requests() -> (
    None
):
    """Rev 1 matrix uses 14 judge calls and 16 scoring requests."""
    plan = plan_frozen_consumer_calls(visual_control_rows=12)
    assert plan.judgment_calls == 14
    assert plan.scoring_requests == 16
    assert ANNOTATION_QUESTIONS_PER_JUDGE_CALL == 2


@pytest.mark.unit
def test_enforce_budget_raises_before_dispatch_when_scoring_exceeds() -> None:
    """Budget guard runs before dispatch when scoring_requests would exceed."""
    scheduled = ConsumerCallCounts(judgment_calls=14, scoring_requests=20)
    budget = ConsumerCallCounts(judgment_calls=14, scoring_requests=16)
    with pytest.raises(ConsumerCallBudgetError, match="scoring_requests"):
        enforce_consumer_call_budget(scheduled, budget)


@pytest.mark.unit
def test_enforce_budget_raises_before_dispatch_when_judgment_exceeds() -> None:
    """Budget guard rejects extra judgment calls before dispatch."""
    scheduled = ConsumerCallCounts(judgment_calls=15, scoring_requests=16)
    budget = ConsumerCallCounts(judgment_calls=14, scoring_requests=16)
    with pytest.raises(ConsumerCallBudgetError, match="judgment_calls"):
        enforce_consumer_call_budget(scheduled, budget)


@pytest.mark.unit
def test_enforce_budget_raises_before_dispatch_when_auxiliary_exceeds() -> None:
    """Auxiliary HTTP probes are budgeted separately from judgment and scoring."""
    scheduled = ConsumerCallCounts(
        judgment_calls=14,
        scoring_requests=16,
        auxiliary_http=4,
    )
    budget = ConsumerCallCounts(
        judgment_calls=14,
        scoring_requests=16,
        auxiliary_http=3,
    )
    with pytest.raises(ConsumerCallBudgetError, match="auxiliary_http"):
        enforce_consumer_call_budget(scheduled, budget)


@pytest.mark.unit
def test_annotation_judge_invokes_score_candidates_twice() -> None:
    """One annotation judge call schedules two score_candidates requests."""
    adapter, fake = adapter_for(
        logprobs_by_call=[
            {"BROWSER_TASK": math.log(0.7), "COMPUTER_TASK": math.log(0.3)},
            {"True": math.log(0.2), "False": math.log(0.8)},
        ],
    )
    adapter.judge("task text", annotation_questions(), "fake")
    assert len(fake.calls) == 2
