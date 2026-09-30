"""Signature-match run over CEDAR pairs and its key-free receipt (#319).

``run_signature_match`` sends one judgment per pair and stops at the first
backend failure. ``signature_match_metrics`` turns the typed answers into
accuracy, ROC-AUC, ECE with a reliability table, the skilled false-accept
rates, the ``cannot_tell`` rate, the ``Noul`` and ``Choice`` agreement and the
same measures by pair kind. ``build_signature_match_receipt`` writes pair
ids, answers, metrics and pins, never image bytes.

The ``same_writer`` probability is model confidence. It is not a match
percentage or a forensic score.

Attributes:
    RECEIPT_ISSUE (int): Issue number recorded in every receipt.
    CONFIDENCE_NOTE (str): Honesty note stored with the metrics.
    QUALITY_LEVELS (int): Levels of the image-quality ``Score``.

Examples:
    ```python
    from typevet_evals.signature_match.runner import (
        run_signature_match,
        signature_match_metrics,
    )

    run = run_signature_match(port, requests, "model-id")
    metrics = signature_match_metrics(run.outcomes)
    ```

See Also:
    - [typevet_evals.signature_match.metrics][]: the signature metrics
    - [typevet_evals.signature_match.request][]: one request per pair
    - [typevet_evals.face_match.runner][]: the face-match run pattern
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Final

from typevet.domain import JudgmentResponse
from typevet.domain.errors import GenerationError
from typevet.ports import JudgmentPort
from typevet_evals.datasets.cedar import PairKind
from typevet_evals.face_match.metrics import (
    cannot_tell_rate,
    expected_calibration_error,
    reliability_table,
    roc_auc,
)
from typevet_evals.signature_match.metrics import (
    ACCEPT_THRESHOLD,
    kind_accuracy,
    noul_accept_rate,
    noul_choice_agreement,
    skilled_false_accept,
    verdict_accept_rate,
    verdict_accuracy,
)
from typevet_evals.signature_match.request import (
    IMAGE_QUALITY,
    SAME_WRITER,
    VERDICT,
    VERDICT_LABELS,
    SignatureMatchRequest,
    judge_signature_match,
)

RECEIPT_ISSUE: Final[int] = 319
CONFIDENCE_NOTE: Final[str] = (
    "same_writer_confidence is model confidence from the Noul answer. "
    "It is not a match percentage or a forensic score."
)
QUALITY_LEVELS: Final[int] = 5


@dataclass(frozen=True, slots=True)
class SignatureMatchOutcome:
    """Typed answers and timing for one pair.

    Attributes:
        pair_id (str): Stable pair id, see ``SignatureMatchRequest.pair_id``.
        kind (PairKind): Gold pair kind.
        gold_same_writer (bool): Gold same-writer label.
        same_writer_confidence (float): ``Noul`` probability; model confidence.
        verdict (str): ``Choice`` label.
        verdict_probabilities (Mapping[str, float]): ``Choice`` distribution.
        quality_score (float): Expected ``Score`` value.
        quality_level (int): Most probable ``Score`` level.
        quality_probabilities (Mapping[int, float]): ``Score`` distribution.
        latency_seconds (float): Wall time of the judgment call.
        input_tokens (int | None): Prompt tokens the backend reported.

    Examples:
        ```python
        outcome.to_receipt()["pair_id"]
        ```
    """

    pair_id: str
    kind: PairKind
    gold_same_writer: bool
    same_writer_confidence: float
    verdict: str
    verdict_probabilities: Mapping[str, float]
    quality_score: float
    quality_level: int
    quality_probabilities: Mapping[int, float]
    latency_seconds: float
    input_tokens: int | None

    def to_receipt(self) -> dict[str, object]:
        """Return a JSON-ready mapping for the receipt.

        Returns:
            Pair id, kind, gold label, typed answers, latency and tokens.
        """
        return {
            "pair_id": self.pair_id,
            "kind": str(self.kind),
            "gold_same_writer": self.gold_same_writer,
            "same_writer_confidence": self.same_writer_confidence,
            "verdict": self.verdict,
            "verdict_probabilities": dict(self.verdict_probabilities),
            "quality_score": self.quality_score,
            "quality_level": self.quality_level,
            "quality_probabilities": {
                str(k): v for k, v in sorted(self.quality_probabilities.items())
            },
            "latency_seconds": round(self.latency_seconds, 3),
            "input_tokens": self.input_tokens,
        }


@dataclass(frozen=True, slots=True)
class SignatureMatchRun:
    """Outcomes of one run and the failure that stopped it, if any.

    Attributes:
        outcomes (tuple[SignatureMatchOutcome, ...]): One outcome per judged
            pair.
        stopped (dict[str, object] | None): Index, pair id, error class and
            message of the first failure; ``None`` when every pair ran.
        wall_seconds (float): Wall time of the whole run.

    Examples:
        ```python
        SignatureMatchRun(outcomes=(), stopped=None, wall_seconds=0.0)
        ```
    """

    outcomes: tuple[SignatureMatchOutcome, ...]
    stopped: dict[str, object] | None
    wall_seconds: float


def outcome_from_response(
    request: SignatureMatchRequest, response: JudgmentResponse, latency_seconds: float
) -> SignatureMatchOutcome:
    """Read the three typed answers for one pair.

    Args:
        request: The request that was judged.
        response: Typed answers from the judgment port.
        latency_seconds: Wall time of the call.

    Returns:
        The outcome for the receipt and the metrics.
    """
    choice = response.choices[VERDICT]
    score = response.scores[IMAGE_QUALITY]
    level = max(score.probabilities, key=lambda k: score.probabilities[k])
    return SignatureMatchOutcome(
        pair_id=request.pair_id,
        kind=request.gold_kind,
        gold_same_writer=request.gold_same_writer,
        same_writer_confidence=response.nouls[SAME_WRITER].noul,
        verdict=choice.choice,
        verdict_probabilities=dict(choice.probabilities),
        quality_score=score.score,
        quality_level=level,
        quality_probabilities=dict(score.probabilities),
        latency_seconds=latency_seconds,
        input_tokens=response.usage.input_tokens,
    )


def run_signature_match(
    port: JudgmentPort,
    requests: Iterable[SignatureMatchRequest],
    model: str,
    *,
    clock: Callable[[], float] = time.perf_counter,
) -> SignatureMatchRun:
    """Judge each request once, in order, and stop at the first failure.

    A backend failure (``GenerationError``, for example ``TransportError`` or
    ``BackendHttpError``) stops the run. The run records it and keeps the
    outcomes that came before it.

    Args:
        port: Judgment port for the backend.
        requests: Requests in slice order.
        model: Backend model id or alias.
        clock: Monotonic clock in seconds.

    Returns:
        The outcomes, the stopping failure and the wall time.
    """
    started = clock()
    outcomes: list[SignatureMatchOutcome] = []
    stopped: dict[str, object] | None = None
    for index, request in enumerate(requests):
        call_started = clock()
        try:
            response = judge_signature_match(port, request, model)
        except GenerationError as exc:
            stopped = {
                "index": index,
                "pair_id": request.pair_id,
                "error_class": type(exc).__name__,
                "message": str(exc),
            }
            break
        outcomes.append(
            outcome_from_response(request, response, clock() - call_started)
        )
    return SignatureMatchRun(tuple(outcomes), stopped, clock() - started)


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _kind_metrics(outcomes: Sequence[SignatureMatchOutcome]) -> dict[str, Any]:
    verdicts = [o.verdict for o in outcomes]
    confidences = [o.same_writer_confidence for o in outcomes]
    levels = [o.quality_level for o in outcomes]
    empty = not outcomes
    return {
        "pairs": len(outcomes),
        "accuracy": (
            None
            if empty
            else verdict_accuracy(verdicts, [o.gold_same_writer for o in outcomes])
        ),
        "kind_accuracy": (
            None if empty else kind_accuracy(verdicts, [o.kind for o in outcomes])
        ),
        "cannot_tell_rate": None if empty else cannot_tell_rate(verdicts),
        "noul_accept_rate": None if empty else noul_accept_rate(confidences),
        "verdict_accept_rate": None if empty else verdict_accept_rate(verdicts),
        "mean_same_writer_confidence": _mean(confidences),
        "verdict_counts": {label: verdicts.count(label) for label in VERDICT_LABELS},
        "quality_levels": {str(n): levels.count(n) for n in range(QUALITY_LEVELS)},
    }


def _auc_against(
    outcomes: Sequence[SignatureMatchOutcome], kind: PairKind
) -> float | None:
    rows = [o for o in outcomes if o.kind in (PairKind.GENUINE_GENUINE, kind)]
    return roc_auc(
        [o.same_writer_confidence for o in rows], [o.gold_same_writer for o in rows]
    )


def signature_match_metrics(
    outcomes: Sequence[SignatureMatchOutcome],
) -> dict[str, Any]:
    """Compute every signature-match metric over the judged pairs.

    Args:
        outcomes: Outcomes from ``run_signature_match``.

    Returns:
        Overall accuracy, kind accuracy, ROC-AUC (all pairs and against each
        forgery kind), ECE, the ten-bin reliability table, the ``cannot_tell``
        rate, the skilled false-accept rates by both definitions, the ``Noul``
        and ``Choice`` agreement, the measures by pair kind, mean latency and
        the confidence note. Rates are ``None`` when there are no outcomes.
    """
    gold = [o.gold_same_writer for o in outcomes]
    confidences = [o.same_writer_confidence for o in outcomes]
    verdicts = [o.verdict for o in outcomes]
    kinds = [o.kind for o in outcomes]
    empty = not outcomes
    overall = _kind_metrics(outcomes)
    agreement = None if empty else noul_choice_agreement(confidences, verdicts)
    return {
        "pairs": len(outcomes),
        "same_writer_pairs": sum(gold),
        "accuracy": overall["accuracy"],
        "kind_accuracy": overall["kind_accuracy"],
        "roc_auc": roc_auc(confidences, gold),
        "roc_auc_by_negative_kind": {
            str(kind): _auc_against(outcomes, kind)
            for kind in (PairKind.GENUINE_SKILLED, PairKind.GENUINE_RANDOM)
        },
        "ece": None if empty else expected_calibration_error(confidences, gold),
        "reliability": [
            {
                "lower": b.lower,
                "upper": b.upper,
                "count": b.count,
                "mean_confidence": b.mean_confidence,
                "fraction_same_writer": b.fraction_same_person,
            }
            for b in reliability_table(confidences, gold)
        ],
        "cannot_tell_rate": overall["cannot_tell_rate"],
        "skilled_false_accept": {
            "threshold": ACCEPT_THRESHOLD,
            **asdict(skilled_false_accept(confidences, verdicts, kinds)),
        },
        "noul_choice_agreement": None if agreement is None else asdict(agreement),
        "by_kind": {
            str(kind): _kind_metrics([o for o in outcomes if o.kind is kind])
            for kind in PairKind
        },
        "mean_latency_seconds": _mean([o.latency_seconds for o in outcomes]),
        "confidence_note": CONFIDENCE_NOTE,
    }


def build_signature_match_receipt(
    run: SignatureMatchRun,
    *,
    backend: str,
    model: str,
    pins: Mapping[str, object],
    identity: Mapping[str, object],
) -> dict[str, Any]:
    """Build the receipt body for one run.

    Args:
        run: Result of ``run_signature_match``.
        backend: Backend name, for example ``llama_cpp`` or ``vllm``.
        model: Model id sent to the backend.
        pins: Dataset, slice and server pins.
        identity: Experiment identity mapping.

    Returns:
        JSON-ready receipt with pair ids, typed answers and metrics. It holds
        no image bytes.
    """
    return {
        "issue": RECEIPT_ISSUE,
        "backend": backend,
        "model": model,
        "pins": dict(pins),
        "identity": dict(identity),
        "stopped": run.stopped,
        "wall_seconds": round(run.wall_seconds, 3),
        "metrics": signature_match_metrics(run.outcomes),
        "pairs": [o.to_receipt() for o in run.outcomes],
    }
