"""Doodle ``Choice`` run, its metrics and its key-free receipt (#412).

``run_doodle_duel`` sends one judgment per doodle, optionally several at one
time, and stops at the first backend failure. ``doodle_metrics`` turns the
answers into accuracy, accuracy per category and the mean probability of the
chosen label. ``build_doodle_receipt`` writes key ids, labels, the full
option distribution, metrics and pins, never image bytes. ``shuffle_rows``
gives the seeded run order.

The option probabilities are model confidence, not calibrated rates.

Attributes:
    IMAGES_PER_JUDGMENT (int): Images sent with each judgment.
    RECEIPT_ISSUE (int): Issue number recorded in every receipt.

Examples:
    ```python
    from typevet_evals.doodle_duel.runner import doodle_metrics, run_doodle_duel

    run = run_doodle_duel(port, requests, "model-id")
    metrics = doodle_metrics(run.outcomes)
    ```

See Also:
    - [typevet_evals.doodle_duel.request][]: one request per doodle
    - [typevet_evals.face_match.pool][]: the shared ordered judgment loop
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from typevet.ports import JudgmentPort
from typevet_evals.datasets.quickdraw import DATASET_CREDIT, DATASET_LICENSE
from typevet_evals.doodle_duel.request import (
    DOODLE_QUESTION,
    DoodleRequest,
    judge_doodle,
)
from typevet_evals.face_match.pool import judge_in_order
from typevet_evals.serving_metrics import run_throughput

IMAGES_PER_JUDGMENT: Final[int] = 1
RECEIPT_ISSUE: Final[int] = 412


@dataclass(frozen=True, slots=True)
class DoodleOutcome:
    """The ``Choice`` answer and timing for one doodle.

    Attributes:
        key_id (str): Dataset drawing id.
        true_label (str): Category the player was asked to draw.
        chosen_label (str): Most probable ``Choice`` label.
        probabilities (Mapping[str, float]): Probability of every option.
        latency_seconds (float): Wall time of the judgment call.

    Examples:
        ```python
        outcome.to_receipt()["chosen_label"]
        ```
    """

    key_id: str
    true_label: str
    chosen_label: str
    probabilities: Mapping[str, float]
    latency_seconds: float

    @property
    def correct(self) -> bool:
        """Return whether the chosen label is the true label.

        Returns:
            ``True`` when the labels are equal.
        """
        return self.chosen_label == self.true_label

    def to_receipt(self) -> dict[str, object]:
        """Return a JSON-ready mapping for the receipt.

        Returns:
            Key id, both labels, every option probability, the result and
            the latency.
        """
        return {
            "key_id": self.key_id,
            "true_label": self.true_label,
            "chosen_label": self.chosen_label,
            "probabilities": dict(self.probabilities),
            "correct": self.correct,
            "latency_seconds": round(self.latency_seconds, 3),
        }


@dataclass(frozen=True, slots=True)
class DoodleRun:
    """Outcomes of one run and the failure that stopped it, if any.

    Attributes:
        outcomes (tuple[DoodleOutcome, ...]): One outcome per judged doodle.
        stopped (dict[str, object] | None): Index, key id, error class and
            message of the first failure; ``None`` when every doodle ran.
        wall_seconds (float): Wall time of the whole run.
        concurrency (int): Most judgments in flight at one time.
        discarded (int): Judgments after the first failure that were
            dropped by design.

    Examples:
        ```python
        DoodleRun(outcomes=(), stopped=None, wall_seconds=0.0)
        ```
    """

    outcomes: tuple[DoodleOutcome, ...]
    stopped: dict[str, object] | None
    wall_seconds: float
    concurrency: int = 1
    discarded: int = 0


def shuffle_rows[T](
    rows: Iterable[T], seed: int, *, key: Callable[[T], str]
) -> list[T]:
    """Return ``rows`` in a seeded order that does not depend on input order.

    Each row sorts by the SHA-256 of ``"<seed>:<key(row)>"``.

    Args:
        rows: Rows to order.
        seed: Order seed.
        key: Stable id of one row, for example the drawing ``key_id``.

    Returns:
        The rows in seeded order.
    """
    return sorted(
        rows, key=lambda row: hashlib.sha256(f"{seed}:{key(row)}".encode()).digest()
    )


def run_doodle_duel(
    port: JudgmentPort,
    requests: Iterable[DoodleRequest],
    model: str,
    *,
    concurrency: int = 1,
    clock: Callable[[], float] = time.perf_counter,
) -> DoodleRun:
    """Judge each request once and stop at the first failure.

    Args:
        port: Judgment port for the backend. It must be safe to call from
            several threads when ``concurrency`` is above 1.
        requests: Requests in run order.
        model: Backend model id or alias.
        concurrency: Most judgments in flight at one time.
        clock: Monotonic clock in seconds.

    Returns:
        The outcomes, the stopping failure, the wall time, the concurrency
        and the discarded count.

    Raises:
        ValueError: When ``concurrency`` is less than 1.
    """
    started = clock()
    batch = judge_in_order(
        requests,
        lambda request: judge_doodle(port, request, model),
        concurrency=concurrency,
        clock=clock,
    )
    outcomes = []
    for judged in batch.judged:
        answer = judged.response.choices[DOODLE_QUESTION]
        outcomes.append(
            DoodleOutcome(
                key_id=judged.request.key_id,
                true_label=judged.request.true_label,
                chosen_label=answer.choice,
                probabilities=dict(answer.probabilities),
                latency_seconds=judged.latency_seconds,
            )
        )
    failure = batch.failure
    return DoodleRun(
        outcomes=tuple(outcomes),
        stopped=None
        if failure is None
        else failure.record("key_id", failure.request.key_id),
        wall_seconds=clock() - started,
        concurrency=concurrency,
        discarded=0 if failure is None else failure.discarded,
    )


def doodle_metrics(outcomes: Sequence[DoodleOutcome]) -> dict[str, Any]:
    """Compute the doodle metrics over the judged doodles.

    Args:
        outcomes: Outcomes from ``run_doodle_duel``.

    Returns:
        ``count``, ``accuracy``, ``per_category_accuracy`` (by true label,
        in first-seen order) and ``mean_chosen_probability``. Rates are
        ``None`` when there are no outcomes.
    """
    by_label: dict[str, list[bool]] = {}
    for outcome in outcomes:
        by_label.setdefault(outcome.true_label, []).append(outcome.correct)
    count = len(outcomes)
    return {
        "count": count,
        "accuracy": sum(o.correct for o in outcomes) / count if count else None,
        "per_category_accuracy": {
            label: sum(results) / len(results) for label, results in by_label.items()
        },
        "mean_chosen_probability": (
            sum(o.probabilities[o.chosen_label] for o in outcomes) / count
            if count
            else None
        ),
    }


def build_doodle_receipt(
    run: DoodleRun,
    *,
    backend: str,
    model: str,
    pins: Mapping[str, object],
    identity: Mapping[str, object],
    server_args: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the receipt body for one run.

    Args:
        run: Result of ``run_doodle_duel``.
        backend: Backend name, for example ``llama_cpp`` or ``vllm``.
        model: Model id sent to the backend.
        pins: Dataset, sample and server pins. The receipt adds the dataset
            licence and credit.
        identity: Experiment identity mapping.
        server_args: Server arguments block; ``None`` when the caller built
            none.

    Returns:
        JSON-ready receipt with rows, metrics, pins and the ``throughput``
        block. It holds no image bytes.
    """
    return {
        "issue": RECEIPT_ISSUE,
        "backend": backend,
        "model": model,
        "pins": {
            **dict(pins),
            "dataset_license": DATASET_LICENSE,
            "credit": DATASET_CREDIT,
        },
        "identity": dict(identity),
        "stopped": run.stopped,
        "wall_seconds": round(run.wall_seconds, 3),
        "throughput": run_throughput(
            [o.latency_seconds for o in run.outcomes],
            images_per_judgment=IMAGES_PER_JUDGMENT,
            wall_seconds=run.wall_seconds,
            concurrency=run.concurrency,
            discarded=run.discarded,
            server=None,
        ),
        "server_args": None if server_args is None else dict(server_args),
        "metrics": doodle_metrics(run.outcomes),
        "rows": [o.to_receipt() for o in run.outcomes],
    }
