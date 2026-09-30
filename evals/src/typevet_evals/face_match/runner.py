"""Face-match run over LFW pairs and its key-free receipt (#301).

``run_face_match`` sends one judgment per pair, optionally several at one time
(``concurrency``), and stops at the first backend
failure. ``face_match_metrics`` turns the typed answers into accuracy,
ROC-AUC, ECE with a reliability table, the ``cannot_tell`` rate and the
``Score`` distribution. ``build_face_match_receipt`` writes pair ids, answers,
metrics and pins, never image bytes.

The ``same_person`` probability is model confidence. It is not a calibrated
match percentage.

Attributes:
    RECEIPT_ISSUE (int): Issue number recorded in every receipt.
    CONFIDENCE_NOTE (str): Honesty note stored with the metrics.
    VLLM_REVISION_ENV (str): Variable that names the served vLLM weights
        revision.

Examples:
    ```python
    from typevet_evals.face_match.runner import face_match_metrics, run_face_match

    run = run_face_match(port, requests, "model-id")
    metrics = face_match_metrics(run.outcomes)
    ```

See Also:
    - [typevet_evals.face_match.metrics][]: the metric functions
    - [typevet_evals.face_match.request][]: one request per pair
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from typevet.domain import JudgmentResponse
from typevet.ports import JudgmentPort
from typevet_evals.face_match.metrics import (
    cannot_tell_rate,
    expected_calibration_error,
    reliability_table,
    roc_auc,
    score_distribution,
    verdict_accuracy,
)
from typevet_evals.face_match.pool import judge_in_order
from typevet_evals.face_match.request import (
    FACE_VISIBILITY,
    SAME_PERSON,
    VERDICT,
    FaceMatchRequest,
    judge_face_match,
)

RECEIPT_ISSUE: Final[int] = 301
CONFIDENCE_NOTE: Final[str] = (
    "same_person_confidence is model confidence from the Noul answer. "
    "It is not a calibrated match percentage."
)
VLLM_REVISION_ENV: Final[str] = "TYPEVET_VLLM_MODEL_REVISION"
_VISIBILITY_LEVELS: Final[int] = 5
_AUTH_NEEDLES: Final[tuple[str, ...]] = ("authorization", "bearer ")


@dataclass(frozen=True, slots=True)
class FaceMatchOutcome:
    """Typed answers and timing for one pair.

    Attributes:
        pair_id (str): Stable pair id, see ``FaceMatchRequest.pair_id``.
        gold_same_person (bool): Gold label.
        same_person_confidence (float): ``Noul`` probability; model confidence.
        verdict (str): ``Choice`` label.
        verdict_probabilities (Mapping[str, float]): ``Choice`` distribution.
        visibility_score (float): Expected ``Score`` value.
        visibility_level (int): Most probable ``Score`` level.
        visibility_probabilities (Mapping[int, float]): ``Score`` distribution.
        latency_seconds (float): Wall time of the judgment call.
        input_tokens (int | None): Prompt tokens the backend reported.

    Examples:
        ```python
        outcome.to_receipt()["pair_id"]
        ```
    """

    pair_id: str
    gold_same_person: bool
    same_person_confidence: float
    verdict: str
    verdict_probabilities: Mapping[str, float]
    visibility_score: float
    visibility_level: int
    visibility_probabilities: Mapping[int, float]
    latency_seconds: float
    input_tokens: int | None

    def to_receipt(self) -> dict[str, object]:
        """Return a JSON-ready mapping for the receipt.

        Returns:
            Pair id, gold label, typed answers, latency and tokens.
        """
        return {
            "pair_id": self.pair_id,
            "gold_same_person": self.gold_same_person,
            "same_person_confidence": self.same_person_confidence,
            "verdict": self.verdict,
            "verdict_probabilities": dict(self.verdict_probabilities),
            "visibility_score": self.visibility_score,
            "visibility_level": self.visibility_level,
            "visibility_probabilities": {
                str(k): v for k, v in sorted(self.visibility_probabilities.items())
            },
            "latency_seconds": round(self.latency_seconds, 3),
            "input_tokens": self.input_tokens,
        }


@dataclass(frozen=True, slots=True)
class FaceMatchRun:
    """Outcomes of one run and the failure that stopped it, if any.

    Attributes:
        outcomes (tuple[FaceMatchOutcome, ...]): One outcome per judged pair.
        stopped (dict[str, object] | None): Index, pair id, error class and
            message of the first failure; ``None`` when every pair ran.
        wall_seconds (float): Wall time of the whole run.

    Examples:
        ```python
        FaceMatchRun(outcomes=(), stopped=None, wall_seconds=0.0)
        ```
    """

    outcomes: tuple[FaceMatchOutcome, ...]
    stopped: dict[str, object] | None
    wall_seconds: float


def outcome_from_response(
    request: FaceMatchRequest, response: JudgmentResponse, latency_seconds: float
) -> FaceMatchOutcome:
    """Read the three typed answers for one pair.

    Args:
        request: The request that was judged.
        response: Typed answers from the judgment port.
        latency_seconds: Wall time of the call.

    Returns:
        The outcome for the receipt and the metrics.
    """
    choice = response.choices[VERDICT]
    score = response.scores[FACE_VISIBILITY]
    level = max(score.probabilities, key=lambda k: score.probabilities[k])
    return FaceMatchOutcome(
        pair_id=request.pair_id,
        gold_same_person=request.gold_same_person,
        same_person_confidence=response.nouls[SAME_PERSON].noul,
        verdict=choice.choice,
        verdict_probabilities=dict(choice.probabilities),
        visibility_score=score.score,
        visibility_level=level,
        visibility_probabilities=dict(score.probabilities),
        latency_seconds=latency_seconds,
        input_tokens=response.usage.input_tokens,
    )


def run_face_match(
    port: JudgmentPort,
    requests: Iterable[FaceMatchRequest],
    model: str,
    *,
    concurrency: int = 1,
    clock: Callable[[], float] = time.perf_counter,
) -> FaceMatchRun:
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
        lambda request: judge_face_match(port, request, model),
        concurrency=concurrency,
        clock=clock,
    )
    outcomes = tuple(
        outcome_from_response(j.request, j.response, j.latency_seconds)
        for j in batch.judged
    )
    stopped = (
        None
        if batch.failure is None
        else batch.failure.record("pair_id", batch.failure.request.pair_id)
    )
    return FaceMatchRun(outcomes, stopped, clock() - started)


def face_match_metrics(outcomes: Sequence[FaceMatchOutcome]) -> dict[str, Any]:
    """Compute every face-match metric over the judged pairs.

    Args:
        outcomes: Outcomes from ``run_face_match``.

    Returns:
        Pair count, accuracy, ROC-AUC, ECE, the ten-bin reliability table,
        the ``cannot_tell`` rate, the ``Score`` distribution by gold label,
        mean latency and the confidence note. Rates are ``None`` when there
        are no outcomes.
    """
    gold = [o.gold_same_person for o in outcomes]
    confidences = [o.same_person_confidence for o in outcomes]
    verdicts = [o.verdict for o in outcomes]
    table = reliability_table(confidences, gold)
    empty = not outcomes
    return {
        "pairs": len(outcomes),
        "same_person_pairs": sum(gold),
        "accuracy": None if empty else verdict_accuracy(verdicts, gold),
        "roc_auc": roc_auc(confidences, gold),
        "ece": None if empty else expected_calibration_error(confidences, gold),
        "reliability": [
            {
                "lower": b.lower,
                "upper": b.upper,
                "count": b.count,
                "mean_confidence": b.mean_confidence,
                "fraction_same_person": b.fraction_same_person,
            }
            for b in table
        ],
        "cannot_tell_rate": None if empty else cannot_tell_rate(verdicts),
        "score_distribution": score_distribution(
            [o.visibility_level for o in outcomes], gold, n_levels=_VISIBILITY_LEVELS
        ),
        "mean_latency_seconds": (
            None if empty else sum(o.latency_seconds for o in outcomes) / len(outcomes)
        ),
        "confidence_note": CONFIDENCE_NOTE,
    }


def build_face_match_receipt(
    run: FaceMatchRun,
    *,
    backend: str,
    model: str,
    pins: Mapping[str, object],
    identity: Mapping[str, object],
) -> dict[str, Any]:
    """Build the receipt body for one run.

    Args:
        run: Result of ``run_face_match``.
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
        "metrics": face_match_metrics(run.outcomes),
        "pairs": [o.to_receipt() for o in run.outcomes],
    }


def served_weights_pins(backend: str, environ: Mapping[str, str]) -> dict[str, str]:
    """Read the served weights revision pin for one backend.

    A vLLM run records the weights revision from ``VLLM_REVISION_ENV``. Other
    backends record no revision pin.

    Args:
        backend: Backend name, for example ``llama_cpp`` or ``vllm``.
        environ: Environment variables.

    Returns:
        ``{"model_revision": <revision>}`` for ``vllm``. An empty mapping for
        other backends.

    Raises:
        ValueError: When the backend is ``vllm`` and the variable is missing
            or blank.
    """
    if backend != "vllm":
        return {}
    revision = environ.get(VLLM_REVISION_ENV, "").strip()
    if not revision:
        msg = f"{VLLM_REVISION_ENV} must name the served weights revision"
        raise ValueError(msg)
    return {"model_revision": revision}


def ensure_key_free(text: str, *, secrets: Iterable[str | None]) -> None:
    """Refuse receipt text that holds a key or an auth header.

    Args:
        text: Serialized receipt.
        secrets: Keys that must not appear; empty and ``None`` values are
            skipped.

    Raises:
        ValueError: When a secret or an auth header appears in ``text``.
    """
    lowered = text.lower()
    leaked = [s for s in secrets if s and s in text]
    headers = [n for n in _AUTH_NEEDLES if n in lowered]
    if leaked or headers:
        msg = "receipt is not key-free: it holds a key or an auth header"
        raise ValueError(msg)
