"""Judgment vs scoring call accounting for the PSAI consumer matrix ([#177][i177]).

One ``ScoringJudgmentAdapter.judge`` invocation over ``annotation_questions()``
schedules two ``score_candidates`` calls (``category`` Choice + ``requires_login``
Noul). The frozen live protocol runs twelve visual judgments (four rows times
three conditions) plus two annotation judgments: **14 judgment calls** and
**16 scoring requests**. Auxiliary HTTP (health, capability, template probe)
is counted separately.

Examples:
    ```python
    from typevet_evals.psai_vision_consumer.accounting import (
        ANNOTATION_QUESTIONS_PER_JUDGE_CALL,
        ConsumerCallCounts,
        enforce_consumer_call_budget,
        plan_frozen_consumer_calls,
    )
    from typevet_evals.datasets.psai_vision_controls import (
        TEXT_NOUL_UIDS,
        control_matrix,
    )

    plan = plan_frozen_consumer_calls(visual_control_rows=12)
    assert plan.judgment_calls == 14
    assert plan.scoring_requests == 16
    enforce_consumer_call_budget(plan, plan)
    assert ANNOTATION_QUESTIONS_PER_JUDGE_CALL == 2
    ```

See Also:
    - [typevet_evals.psai_vision_consumer.harness][]: receipt writer
    - [typevet_evals.datasets.psai_vision_controls][]: control matrix

[i177]: https://github.com/Alberto-Codes/typevet/issues/177

Attributes:
    ANNOTATION_QUESTIONS_PER_JUDGE_CALL (int): Scoring requests per annotation
        ``judge`` call (= ``len(annotation_questions())``).
    FROZEN_CONSUMER_CASE_UIDS (frozenset[str]): Protocol rev 1 row pins.
    TEXT_ANNOTATION_JUDGE_UIDS (tuple[str, ...]): Rows that run text-only
        annotation regressions.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from typevet_evals.datasets.psai_vision_controls import annotation_questions

ANNOTATION_QUESTIONS_PER_JUDGE_CALL: int = len(annotation_questions())

FROZEN_CONSUMER_CASE_UIDS: frozenset[str] = frozenset(
    {
        "cmcc8u6yc00v91p1yw2eruz95",  # C01
        "cmcc8u6yd00wv1p1yy8guorre",  # C03
        "cmcc8u6ym018l1p1yxhf18gc2",  # C08
        "cmcc8u6yc00va1p1ydsdu52zy",  # C10
    }
)
TEXT_ANNOTATION_JUDGE_UIDS: tuple[str, ...] = (
    "cmcc8u6yc00v91p1yw2eruz95",
    "cmcc8u6ym018l1p1yxhf18gc2",
)


@dataclass(frozen=True, slots=True)
class ConsumerCallCounts:
    """Scheduled judgment, scoring and auxiliary HTTP counts for one matrix run.

    Attributes:
        judgment_calls (int): ``ScoringJudgmentAdapter.judge`` invocations.
        scoring_requests (int): Underlying ``score_candidates`` calls.
        auxiliary_http (int): Non-judgment HTTP probes (health, capability,
            template classification).
        failed_attempts (int): Judgment or probe attempts that raised before
            a durable row was recorded.

    Examples:
        ```python
        counts = ConsumerCallCounts(
            judgment_calls=14,
            scoring_requests=16,
            auxiliary_http=3,
        )
        assert counts.judgment_calls < counts.scoring_requests
        ```
    """

    judgment_calls: int
    scoring_requests: int
    auxiliary_http: int = 0
    failed_attempts: int = 0


class ConsumerCallBudgetError(ValueError):
    """Raised when a schedule would exceed a declared call budget.

    Examples:
        ```python
        from typevet_evals.psai_vision_consumer.accounting import (
            ConsumerCallBudgetError,
        )

        raise ConsumerCallBudgetError("scoring_requests 20 exceed budget 16")
        ```
    """


def plan_frozen_consumer_calls(
    *,
    visual_control_rows: int,
    annotation_judge_rows: int = len(TEXT_ANNOTATION_JUDGE_UIDS),
    questions_per_annotation_judge: int = ANNOTATION_QUESTIONS_PER_JUDGE_CALL,
    auxiliary_http: int = 0,
) -> ConsumerCallCounts:
    """Return judgment and scoring totals for the frozen consumer matrix.

    Args:
        visual_control_rows: Present, omitted and swapped rows scheduled
            (twelve for protocol rev 1).
        annotation_judge_rows: Text-only annotation regressions (two for rev 1).
        questions_per_annotation_judge: Scoring requests per annotation judge
            (two when using ``annotation_questions()``).
        auxiliary_http: Expected non-judgment HTTP probes for live runs.

    Returns:
        Counts before dispatch; use ``enforce_consumer_call_budget`` to guard.
    """
    judgment = visual_control_rows + annotation_judge_rows
    scoring = visual_control_rows + (
        annotation_judge_rows * questions_per_annotation_judge
    )
    return ConsumerCallCounts(
        judgment_calls=judgment,
        scoring_requests=scoring,
        auxiliary_http=auxiliary_http,
    )


def enforce_consumer_call_budget(
    scheduled: ConsumerCallCounts,
    budget: ConsumerCallCounts,
) -> None:
    """Raise when ``scheduled`` would exceed any ``budget`` limit.

    Args:
        scheduled: Counts computed for the upcoming matrix.
        budget: Maximum allowed counts (typically equals ``scheduled`` for rev 1).

    Raises:
        ConsumerCallBudgetError: When any scheduled total exceeds the budget.
    """
    if scheduled.judgment_calls > budget.judgment_calls:
        msg = (
            f"judgment_calls {scheduled.judgment_calls} "
            f"exceed budget {budget.judgment_calls}"
        )
        raise ConsumerCallBudgetError(msg)
    if scheduled.scoring_requests > budget.scoring_requests:
        msg = (
            f"scoring_requests {scheduled.scoring_requests} "
            f"exceed budget {budget.scoring_requests}"
        )
        raise ConsumerCallBudgetError(msg)
    if scheduled.auxiliary_http > budget.auxiliary_http:
        msg = (
            f"auxiliary_http {scheduled.auxiliary_http} "
            f"exceed budget {budget.auxiliary_http}"
        )
        raise ConsumerCallBudgetError(msg)


def summarize_receipt_call_counts(receipt: Mapping[str, Any]) -> ConsumerCallCounts:
    """Read explicit accounting fields from a consumer receipt dict.

    Args:
        receipt: Serialized consumer proof receipt.

    Returns:
        Counts stored under ``judgment_call_count``, ``scoring_request_count``,
        ``auxiliary_http_count``, and ``failed_attempts``.
    """
    return ConsumerCallCounts(
        judgment_calls=int(receipt.get("judgment_call_count", 0)),
        scoring_requests=int(receipt.get("scoring_request_count", 0)),
        auxiliary_http=int(receipt.get("auxiliary_http_count", 0)),
        failed_attempts=int(receipt.get("failed_attempts", 0)),
    )
