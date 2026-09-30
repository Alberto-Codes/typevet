"""Check-versus-register run and its key-free receipt (#316).

``run_check_match`` sends one judgment per case, optionally several at one
time (``concurrency``), and stops at the first
backend failure. ``check_match_metrics`` turns the typed answers into
accuracy by class and by variant, the false-clear rate, ROC-AUC and ECE of
each ``Noul``, the ``cannot_tell`` rate, the ``Score`` summary by variant and
Noul-Choice agreement. ``build_check_match_receipt`` writes case ids, typed
answers, render digests, metrics and pins, never image bytes.

``Noul`` probabilities are model confidence. The checks are generated, so
these metrics are not evidence about real checks, fraud or counterfeits.

Attributes:
    RECEIPT_ISSUE (int): Issue number recorded in every receipt.
    CHECK_CONFIDENCE_NOTE (str): Honesty note stored with the metrics.

Examples:
    ```python
    from typevet_evals.check_match import check_match_metrics, run_check_match

    run = run_check_match(port, requests, "model-id")
    metrics = check_match_metrics(run.outcomes)
    ```

See Also:
    - [typevet_evals.check_match.metrics][]: the check-specific rules
    - [typevet_evals.face_match.runner][]: the two-image run this follows
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from typevet.domain import JudgmentResponse
from typevet.ports import JudgmentPort
from typevet_evals.check_match.cases import CheckVariant, ExpectedLabels
from typevet_evals.check_match.metrics import (
    accuracy_by_group,
    agreement_rate,
    class_key,
    counts_for_false_clear,
    false_clear_rate,
    legibility_gap,
    noul_choice_agreement,
    score_summary,
    verdict_correct,
)
from typevet_evals.check_match.request import (
    AMOUNTS_MATCH,
    LEGIBILITY,
    PAYEE_MATCHES,
    VERDICT,
    CheckMatchRequest,
    judge_check_match,
)
from typevet_evals.face_match.metrics import (
    cannot_tell_rate,
    expected_calibration_error,
    reliability_table,
    roc_auc,
)
from typevet_evals.face_match.pool import judge_in_order

RECEIPT_ISSUE: Final[int] = 316
CHECK_CONFIDENCE_NOTE: Final[str] = (
    "Noul values are model confidence, not calibrated match percentages. "
    "The checks are generated; the results are not evidence about real checks. "
    "VOID crosses the signature line on every render, including unsigned."
)
_LEGIBILITY_LEVELS: Final[int] = 5


@dataclass(frozen=True, slots=True)
class CheckMatchOutcome:
    """Typed answers, expected labels and timing for one case.

    Attributes:
        case_id (str): Stable case id, ``r<row>:<variant>``.
        variant (str): Variant name.
        expected (ExpectedLabels): Expected labels (amendment A1).
        verdict (str): ``Choice`` label.
        verdict_probabilities (Mapping[str, float]): ``Choice`` distribution.
        payee_confidence (float): Payee ``Noul`` probability.
        amounts_confidence (float): Amounts ``Noul`` probability.
        legibility_score (float): Expected ``Score`` value.
        legibility_level (int): Most probable ``Score`` level.
        legibility_probabilities (Mapping[int, float]): ``Score`` distribution.
        render_sha256 (str): SHA-256 of the image bytes that were sent.
        latency_seconds (float): Wall time of the judgment call.
        input_tokens (int | None): Prompt tokens the backend reported.

    Examples:
        ```python
        outcome.to_receipt()["case_id"]
        ```
    """

    case_id: str
    variant: str
    expected: ExpectedLabels
    verdict: str
    verdict_probabilities: Mapping[str, float]
    payee_confidence: float
    amounts_confidence: float
    legibility_score: float
    legibility_level: int
    legibility_probabilities: Mapping[int, float]
    render_sha256: str
    latency_seconds: float
    input_tokens: int | None

    @property
    def correct(self) -> bool:
        """Return whether the verdict is in the accepted set.

        Returns:
            ``True`` when the verdict counts as right.
        """
        return verdict_correct(self.verdict, self.expected)

    @property
    def agreement(self) -> bool | None:
        """Return Noul-Choice agreement for this case.

        Returns:
            ``True``, ``False``, or ``None`` for ``cannot_tell``.
        """
        return noul_choice_agreement(
            self.verdict, self.payee_confidence, self.amounts_confidence
        )

    def to_receipt(self) -> dict[str, object]:
        """Return a JSON-ready mapping for the receipt.

        Returns:
            Case id, expected labels, typed answers, digest, latency and tokens.
        """
        return {
            "case_id": self.case_id,
            "variant": self.variant,
            "expected_verdicts": sorted(self.expected.accepted_verdicts),
            "payee_truth": self.expected.payee_matches,
            "amounts_truth": self.expected.amounts_match,
            "verdict": self.verdict,
            "correct": self.correct,
            "verdict_probabilities": dict(self.verdict_probabilities),
            "payee_confidence": self.payee_confidence,
            "amounts_confidence": self.amounts_confidence,
            "noul_choice_agreement": self.agreement,
            "legibility_score": self.legibility_score,
            "legibility_level": self.legibility_level,
            "legibility_probabilities": {
                str(k): v for k, v in sorted(self.legibility_probabilities.items())
            },
            "render_sha256": self.render_sha256,
            "latency_seconds": round(self.latency_seconds, 3),
            "input_tokens": self.input_tokens,
        }


@dataclass(frozen=True, slots=True)
class CheckMatchRun:
    """Outcomes of one run and the failure that stopped it, if any.

    Attributes:
        outcomes (tuple[CheckMatchOutcome, ...]): One outcome per judged case.
        stopped (dict[str, object] | None): Index, case id, error class and
            message of the first failure; ``None`` when every case ran.
        wall_seconds (float): Wall time of the whole run.

    Examples:
        ```python
        CheckMatchRun(outcomes=(), stopped=None, wall_seconds=0.0)
        ```
    """

    outcomes: tuple[CheckMatchOutcome, ...]
    stopped: dict[str, object] | None
    wall_seconds: float


def check_outcome_from_response(
    request: CheckMatchRequest, response: JudgmentResponse, latency_seconds: float
) -> CheckMatchOutcome:
    """Read the four typed answers for one case.

    Args:
        request: The request that was judged.
        response: Typed answers from the judgment port.
        latency_seconds: Wall time of the call.

    Returns:
        The outcome for the receipt and the metrics.
    """
    choice = response.choices[VERDICT]
    score = response.scores[LEGIBILITY]
    level = max(score.probabilities, key=lambda k: score.probabilities[k])
    return CheckMatchOutcome(
        case_id=request.case_id,
        variant=request.case.variant.value,
        expected=request.case.expected,
        verdict=choice.choice,
        verdict_probabilities=dict(choice.probabilities),
        payee_confidence=response.nouls[PAYEE_MATCHES].noul,
        amounts_confidence=response.nouls[AMOUNTS_MATCH].noul,
        legibility_score=score.score,
        legibility_level=level,
        legibility_probabilities=dict(score.probabilities),
        render_sha256=hashlib.sha256(request.media[0].data).hexdigest(),
        latency_seconds=latency_seconds,
        input_tokens=response.usage.input_tokens,
    )


def run_check_match(
    port: JudgmentPort,
    requests: Iterable[CheckMatchRequest],
    model: str,
    *,
    concurrency: int = 1,
    clock: Callable[[], float] = time.perf_counter,
) -> CheckMatchRun:
    """Judge each request once and stop at the first failure.

    A backend failure (``GenerationError``, for example ``TransportError`` or
    ``BackendHttpError``) stops the run. The run records the first failure in
    slice order and keeps the outcomes that came before it. With
    ``concurrency`` above 1, that many judgments run at one time on a thread
    pool; no request is sent after a failure and the outcomes keep slice
    order.

    Args:
        port: Judgment port for the backend. It must be safe to call from
            several threads when ``concurrency`` is above 1.
        requests: Requests in slice order.
        model: Backend model id or alias.
        concurrency: Most judgments in flight at one time; ``1`` judges one
            request at a time, as before.
        clock: Monotonic clock in seconds.

    Returns:
        The outcomes, the stopping failure and the wall time.

    Raises:
        ValueError: When ``concurrency`` is less than 1.
    """
    started = clock()
    batch = judge_in_order(
        requests,
        lambda request: judge_check_match(port, request, model),
        concurrency=concurrency,
        clock=clock,
    )
    outcomes = tuple(
        check_outcome_from_response(j.request, j.response, j.latency_seconds)
        for j in batch.judged
    )
    stopped = (
        None
        if batch.failure is None
        else batch.failure.record("case_id", batch.failure.request.case_id)
    )
    return CheckMatchRun(outcomes, stopped, clock() - started)


def _noul_metrics(confidences: Sequence[float], truth: Sequence[bool]) -> dict:
    table = reliability_table(confidences, truth)
    return {
        "positives": sum(truth),
        "negatives": len(truth) - sum(truth),
        "roc_auc": roc_auc(confidences, truth),
        "ece": expected_calibration_error(confidences, truth) if truth else None,
        "reliability": [
            {
                "lower": b.lower,
                "upper": b.upper,
                "count": b.count,
                "mean_confidence": b.mean_confidence,
                "fraction_true": b.fraction_same_person,
            }
            for b in table
        ],
    }


def _by_variant(outcomes: Sequence[CheckMatchOutcome]) -> dict[str, list]:
    groups: dict[str, list[CheckMatchOutcome]] = {v.value: [] for v in CheckVariant}
    for outcome in outcomes:
        groups[outcome.variant].append(outcome)
    return {name: items for name, items in groups.items() if items}


def _verdict_metrics(
    outcomes: Sequence[CheckMatchOutcome], groups: Mapping[str, list]
) -> dict[str, Any]:
    verdicts = [o.verdict for o in outcomes]
    expected = [o.expected for o in outcomes]
    correct = [o.correct for o in outcomes]
    false_clear = {
        name: false_clear_rate([o.verdict for o in items], [o.expected for o in items])
        for name, items in groups.items()
    }
    return {
        "accuracy": sum(correct) / len(correct) if outcomes else None,
        "accuracy_by_class": accuracy_by_group(
            [class_key(e) for e in expected], correct
        ),
        "accuracy_by_variant": accuracy_by_group(
            [o.variant for o in outcomes], correct
        ),
        "false_clear_rate": false_clear_rate(verdicts, expected),
        "false_clear_cases": sum(1 for e in expected if counts_for_false_clear(e)),
        "false_clear_by_variant": {
            name: rate for name, rate in false_clear.items() if rate is not None
        },
        "cannot_tell_rate": cannot_tell_rate(verdicts) if outcomes else None,
        "cannot_tell_by_variant": {
            name: cannot_tell_rate([o.verdict for o in items])
            for name, items in groups.items()
        },
    }


def _answer_metrics(
    outcomes: Sequence[CheckMatchOutcome], groups: Mapping[str, list]
) -> dict[str, Any]:
    scores = {
        name: score_summary(
            [o.legibility_score for o in items],
            [o.legibility_level for o in items],
            n_levels=_LEGIBILITY_LEVELS,
        )
        for name, items in groups.items()
    }
    agreement = [o.agreement for o in outcomes]
    return {
        "nouls": {
            PAYEE_MATCHES: _noul_metrics(
                [o.payee_confidence for o in outcomes],
                [o.expected.payee_matches for o in outcomes],
            ),
            AMOUNTS_MATCH: _noul_metrics(
                [o.amounts_confidence for o in outcomes],
                [o.expected.amounts_match for o in outcomes],
            ),
        },
        "score_by_variant": scores,
        "legibility_gap_clean_minus_low": legibility_gap(scores),
        "noul_choice_agreement": {
            "rate": agreement_rate(agreement),
            "counted": sum(1 for a in agreement if a is not None),
            "excluded_cannot_tell": sum(1 for a in agreement if a is None),
        },
    }


def check_match_metrics(outcomes: Sequence[CheckMatchOutcome]) -> dict[str, Any]:
    """Compute every check-match metric over the judged cases.

    Args:
        outcomes: Outcomes from ``run_check_match``.

    Returns:
        Case count; accuracy overall, by class and by variant; the false-clear
        rate overall and by variant; the ``cannot_tell`` rate overall and by
        variant; ROC-AUC, ECE and the ten-bin table of each ``Noul``; the
        ``Score`` summary by variant with the clean minus low-legibility gap;
        Noul-Choice agreement; mean latency and the honesty note. Rates are
        ``None`` when no case counts.
    """
    groups = _by_variant(outcomes)
    return {
        "cases": len(outcomes),
        **_verdict_metrics(outcomes, groups),
        **_answer_metrics(outcomes, groups),
        "mean_latency_seconds": (
            sum(o.latency_seconds for o in outcomes) / len(outcomes)
            if outcomes
            else None
        ),
        "confidence_note": CHECK_CONFIDENCE_NOTE,
    }


def build_check_match_receipt(
    run: CheckMatchRun,
    *,
    backend: str,
    model: str,
    pins: Mapping[str, object],
    identity: Mapping[str, object],
) -> dict[str, Any]:
    """Build the receipt body for one run.

    Args:
        run: Result of ``run_check_match``.
        backend: Backend name, for example ``llama_cpp`` or ``vllm``.
        model: Model id sent to the backend.
        pins: Generator, Pillow and server pins.
        identity: Experiment identity mapping.

    Returns:
        JSON-ready receipt with case ids, typed answers, render digests and
        metrics. It holds no image bytes.
    """
    return {
        "issue": RECEIPT_ISSUE,
        "backend": backend,
        "model": model,
        "pins": dict(pins),
        "identity": dict(identity),
        "stopped": run.stopped,
        "wall_seconds": round(run.wall_seconds, 3),
        "metrics": check_match_metrics(run.outcomes),
        "cases": [o.to_receipt() for o in run.outcomes],
    }
