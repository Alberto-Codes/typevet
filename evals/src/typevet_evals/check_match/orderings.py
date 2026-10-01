"""Balanced option-order study of the verdict ``Choice`` (#105).

The study asks whether the order of the six verdict options moves the
answer. It follows TypeLLM's rule (v0.1.8, ``permutations="auto"``): a
balanced Latin square of orderings, one ordinary scoring request per
ordering, and the arithmetic mean of the post-softmax probabilities per
label. It never averages logprobs. ``permutations`` stays 1; each ordering
is a ``Choice`` whose options are listed in that order.

``orderings_statistics`` is a pure function of the receipt cases, so the
result comment can be reproduced from the committed receipt. It also
reports seeded bootstrap intervals over cases (the full seed-1 amendment).
``orderings_rows`` reads the register rows of the live study.

Attributes:
    ORDERINGS_RECEIPT_ISSUE (int): Issue number recorded in the receipt.
    POSITION_SPREAD_LIMIT (float): A position spread at or below this value
        means no position bias to fix.
    DO_NOT_ADOPT (str): Decision label for a spread at or below the limit.
    ADOPT_OPT_IN (str): Decision label for a larger spread with averaged
        accuracy at or above the single-order accuracy.
    INCONCLUSIVE (str): Decision label for a larger spread with lower
        averaged accuracy.
    BOOTSTRAP_RESAMPLES (int): Bootstrap resamples over cases.
    BOOTSTRAP_SEED (int): Seed of ``resample_indices``.
    INTERVAL_LEVEL (float): Level of the percentile intervals.
    ROWS_ENV (str): Variable that sets the register rows of the live study.
    DEFAULT_ROWS (int): Register rows when ``ROWS_ENV`` is blank or unset.

Examples:
    ```python
    from typevet_evals.check_match import VERDICT_LABELS
    from typevet_evals.check_match.orderings import balanced_orders

    orders = balanced_orders(len(VERDICT_LABELS))
    assert orders[0] == tuple(range(6))
    ```

See Also:
    - [typevet_evals.check_match.request][]: the verdict ``Choice``
    - [typevet_evals.check_match.runner][]: the single-order run
"""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

from typevet.domain import Choice, GenerationError
from typevet.ports import JudgmentPort
from typevet_evals.check_match.cases import ROW_COUNT
from typevet_evals.check_match.request import VERDICT, CheckMatchRequest
from typevet_evals.wording.metrics import percentile_interval, resample_indices

ORDERINGS_RECEIPT_ISSUE: Final[int] = 105
POSITION_SPREAD_LIMIT: Final[float] = 0.05
DO_NOT_ADOPT: Final[str] = "do_not_adopt"
ADOPT_OPT_IN: Final[str] = "adopt_opt_in"
INCONCLUSIVE: Final[str] = "inconclusive"
BOOTSTRAP_RESAMPLES: Final[int] = 1000
BOOTSTRAP_SEED: Final[int] = 0
INTERVAL_LEVEL: Final[float] = 0.95
ROWS_ENV: Final[str] = "TYPEVET_CHECK_ORDERINGS_ROWS"
DEFAULT_ROWS: Final[int] = 3


def orderings_rows(environ: Mapping[str, str]) -> int:
    """Return the register rows of the live study (#105).

    The rule is that of ``parse_seed``: ASCII digits only, with outer
    spaces. A blank or missing value gives ``DEFAULT_ROWS``, the slice of
    the committed 21-case receipt.

    Args:
        environ: Process environment, or a mapping in its place.

    Returns:
        Register rows, 1 to ``ROW_COUNT``.

    Raises:
        ValueError: When the value is not an integer from 1 to
            ``ROW_COUNT``. The message names the variable, not the value.
    """
    text = environ.get(ROWS_ENV, "").strip()
    if not text:
        return DEFAULT_ROWS
    if not (text.isascii() and text.isdigit() and 1 <= int(text) <= ROW_COUNT):
        msg = f"{ROWS_ENV} must be an integer from 1 to {ROW_COUNT}"
        raise ValueError(msg)
    return int(text)


def balanced_orders(k: int) -> tuple[tuple[int, ...], ...]:
    """Return a balanced Latin square of orderings for ``k`` options.

    The construction is a Williams design. Every option takes every
    position equally often and follows every other option equally often.
    Even ``k`` gives ``k`` orderings; odd ``k`` adds the mirror rows and
    gives ``2k``. The symbols are relabeled so that ordering 0 is the
    canonical order ``0 .. k-1``.

    Args:
        k: Number of options.

    Returns:
        The orderings; each lists option indexes in position order.

    Raises:
        ValueError: When ``k`` is less than 1.
    """
    if k < 1:
        msg = f"k must be at least 1, got {k}"
        raise ValueError(msg)
    base = [0] + [(j + 1) // 2 if j % 2 else k - j // 2 for j in range(1, k)]
    inverse = {symbol: column for column, symbol in enumerate(base)}
    rows = [tuple(inverse[(symbol + r) % k] for symbol in base) for r in range(k)]
    if k % 2:
        rows += [tuple(reversed(row)) for row in rows]
    return tuple(rows)


def reordered_choice(
    choice: Choice, labels: Sequence[str], order: Sequence[int]
) -> Choice:
    """Return ``choice`` with its options listed in ``order``.

    Args:
        choice: The canonical ``Choice``.
        labels: Canonical labels; ``order`` indexes them.
        order: Option indexes in position order.

    Returns:
        A new ``Choice`` with the same descriptions and instructions.
    """
    criteria = {labels[i]: choice.criteria[labels[i]] for i in order}
    return Choice(criteria=criteria, instructions=choice.instructions)


def remap_positions(
    labels: Sequence[str], order: Sequence[int], probabilities: Sequence[float]
) -> dict[str, float]:
    """Map per-position probabilities back to the canonical labels.

    Args:
        labels: Canonical labels.
        order: Option indexes in position order.
        probabilities: Probability at each position.

    Returns:
        Probability by label, in canonical label order.
    """
    by_index = dict(zip(order, probabilities, strict=True))
    return {label: by_index[i] for i, label in enumerate(labels)}


def position_probabilities(
    labels: Sequence[str], order: Sequence[int], probabilities: Mapping[str, float]
) -> tuple[float, ...]:
    """Return the probability of the option at each position.

    Args:
        labels: Canonical labels.
        order: Option indexes in position order.
        probabilities: Probability by label.

    Returns:
        One probability per position.
    """
    return tuple(probabilities[labels[i]] for i in order)


def mean_probabilities(
    per_ordering: Sequence[Mapping[str, float]], labels: Sequence[str]
) -> dict[str, float]:
    """Return the arithmetic mean of post-softmax probabilities per label.

    Args:
        per_ordering: Probability by label for each ordering.
        labels: Canonical labels; they set the key order.

    Returns:
        Mean probability by label.

    Raises:
        ValueError: When ``per_ordering`` is empty.
    """
    if not per_ordering:
        msg = "mean needs at least one ordering"
        raise ValueError(msg)
    n = len(per_ordering)
    return {k: math.fsum(p[k] for p in per_ordering) / n for k in labels}


def argmax_label(probabilities: Mapping[str, float], labels: Sequence[str]) -> str:
    """Return the most probable label; the first in label order wins ties.

    Args:
        probabilities: Probability by label.
        labels: Canonical labels.

    Returns:
        The selected label.
    """
    return max(labels, key=lambda label: probabilities[label])


def decision_label(spread: float, averaged: float, single: float) -> str:
    """Apply the pre-registered decision rule.

    Args:
        spread: Position spread, max minus min of the position means.
        averaged: Accuracy of the averaged verdict.
        single: Single-order accuracy on the same cases.

    Returns:
        ``DO_NOT_ADOPT``, ``ADOPT_OPT_IN`` or ``INCONCLUSIVE``.
    """
    if spread <= POSITION_SPREAD_LIMIT:
        return DO_NOT_ADOPT
    return ADOPT_OPT_IN if averaged >= single else INCONCLUSIVE


def _case_numbers(case: Mapping[str, Any]) -> dict[str, Any]:
    orderings = case["orderings"]
    labels = tuple(orderings[0]["order"])
    probs = [o["probabilities"] for o in orderings]
    positions = [
        [p[label] for label in o["order"]]
        for o, p in zip(orderings, probs, strict=True)
    ]
    argmaxes = [argmax_label(p, labels) for p in probs]
    mean = mean_probabilities(probs, labels)
    single = case["single_order"]
    diff = max(abs(probs[0][k] - single["verdict_probabilities"][k]) for k in labels)
    expected = set(case["expected_verdicts"])
    return {
        "positions": positions,
        "stable": len(set(argmaxes)) == 1,
        "mean_correct": argmax_label(mean, labels) in expected,
        "single_correct": single["verdict"] in expected,
        "agree": argmaxes[0] == single["verdict"],
        "diff": diff,
    }


def _spread(numbers: Sequence[Mapping[str, Any]]) -> tuple[list[float], float]:
    rows = [row for n in numbers for row in n["positions"]]
    means = [math.fsum(col) / len(rows) for col in zip(*rows, strict=True)]
    return means, max(means) - min(means)


def _share(numbers: Sequence[Mapping[str, Any]], key: str) -> float:
    return sum(n[key] for n in numbers) / len(numbers)


def _bootstrap(numbers: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    spreads: list[float] = []
    differences: list[float] = []
    for b in range(BOOTSTRAP_RESAMPLES):
        rows = resample_indices(len(numbers), seed=BOOTSTRAP_SEED, resample=b)
        sample = [numbers[i] for i in rows]
        spreads.append(_spread(sample)[1])
        differences.append(
            _share(sample, "mean_correct") - _share(sample, "single_correct")
        )
    spread = percentile_interval(spreads, level=INTERVAL_LEVEL)
    difference = percentile_interval(differences, level=INTERVAL_LEVEL)
    return {
        "spread_interval": [spread.low, spread.high],
        "accuracy_difference_interval": [difference.low, difference.high],
        "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED},
    }


def orderings_statistics(cases: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute the four pre-registered statistics, intervals and decision.

    The intervals come from ``BOOTSTRAP_RESAMPLES`` resamples over cases,
    drawn by ``resample_indices`` with ``BOOTSTRAP_SEED``. They are the
    95% percentile intervals of the position spread and of averaged minus
    single-order accuracy. They are reported only; the decision uses the
    point estimates.

    Args:
        cases: Receipt cases with ``orderings`` (``order`` labels and
            ``probabilities``), ``expected_verdicts`` and ``single_order``.

    Returns:
        Case count, position means and spread, argmax stability, averaged
        and single-order accuracy, ordering 0 agreement and largest
        probability difference, the spread limit, the decision label, the
        two intervals and the bootstrap settings.

    Raises:
        ValueError: When ``cases`` is empty.
    """
    if not cases:
        msg = "statistics need at least one case"
        raise ValueError(msg)
    numbers = [_case_numbers(case) for case in cases]
    means, spread = _spread(numbers)
    averaged = _share(numbers, "mean_correct")
    single = _share(numbers, "single_correct")
    return {
        "cases": len(numbers),
        "position_means": means,
        "position_spread": spread,
        "argmax_stability": _share(numbers, "stable"),
        "averaged_accuracy": averaged,
        "single_order_accuracy": single,
        "ordering0_agreement": _share(numbers, "agree"),
        "ordering0_max_abs_difference": max(x["diff"] for x in numbers),
        "spread_limit": POSITION_SPREAD_LIMIT,
        "decision": decision_label(spread, averaged, single),
        **_bootstrap(numbers),
    }


def single_order_cases(
    receipt: Mapping[str, Any], requests: Sequence[CheckMatchRequest], *, seed: int
) -> dict[str, Mapping[str, Any]]:
    """Return the single-order receipt case of each request, by case id.

    Args:
        receipt: The committed single-order receipt.
        requests: The requests of this study.
        seed: Generator seed of the requests.

    Returns:
        Receipt case by case id, in request order.

    Raises:
        ValueError: When the receipt seed differs, a case is missing, or a
            render digest differs from the receipt.
    """
    if receipt["pins"].get("generator_seed") != seed:
        msg = f"receipt seed differs from seed {seed}"
        raise ValueError(msg)
    by_id = {case["case_id"]: case for case in receipt["cases"]}
    found: dict[str, Mapping[str, Any]] = {}
    for request in requests:
        case = by_id.get(request.case_id)
        if case is None:
            msg = f"case {request.case_id} missing from the receipt"
            raise ValueError(msg)
        digest = hashlib.sha256(request.media[0].data).hexdigest()
        if case["render_sha256"] != digest:
            msg = f"render of {request.case_id} differs from the receipt"
            raise ValueError(msg)
        found[request.case_id] = case
    return found


@dataclass(frozen=True, slots=True)
class OrderingsRun:
    """Case records of one study run and the failure that stopped it.

    Attributes:
        records (tuple[dict[str, Any], ...]): One record per finished case.
        stopped (dict[str, object] | None): Index, case id, ordering, error
            class and message of the first failure; ``None`` when all ran.
        requests (int): Scoring requests that returned an answer.
        wall_seconds (float): Wall time of the run.

    Examples:
        ```python
        OrderingsRun(records=(), stopped=None, requests=0, wall_seconds=0.0)
        ```
    """

    records: tuple[dict[str, Any], ...]
    stopped: dict[str, object] | None
    requests: int
    wall_seconds: float


def _record(
    request: CheckMatchRequest,
    labels: Sequence[str],
    orders: Sequence[Sequence[int]],
    probs: Sequence[dict[str, float]],
    single: Mapping[str, Any],
) -> dict[str, Any]:
    mean = mean_probabilities(probs, labels)
    argmaxes = [argmax_label(p, labels) for p in probs]
    verdict = argmax_label(mean, labels)
    expected = sorted(request.case.expected.accepted_verdicts)
    return {
        "case_id": request.case_id,
        "variant": request.case.variant.value,
        "expected_verdicts": expected,
        "render_sha256": hashlib.sha256(request.media[0].data).hexdigest(),
        "single_order": {
            "verdict": single["verdict"],
            "verdict_probabilities": dict(single["verdict_probabilities"]),
        },
        "orderings": [
            {"order": [labels[i] for i in o], "probabilities": p, "argmax": a}
            for o, p, a in zip(orders, probs, argmaxes, strict=True)
        ],
        "position_probabilities": [
            list(position_probabilities(labels, o, p))
            for o, p in zip(orders, probs, strict=True)
        ],
        "mean_probabilities": mean,
        "mean_verdict": verdict,
        "mean_correct": verdict in expected,
        "argmax_stable": len(set(argmaxes)) == 1,
    }


def run_orderings(
    port: JudgmentPort,
    requests: Sequence[CheckMatchRequest],
    model: str,
    orders: Sequence[Sequence[int]],
    single: Mapping[str, Mapping[str, Any]],
    *,
    clock: Callable[[], float] = time.perf_counter,
) -> OrderingsRun:
    """Score only the verdict once per ordering, one request at a time.

    Each call sends the request state and image with one reordered verdict
    ``Choice``. The run stops at the first backend failure.

    Args:
        port: Judgment port for the backend.
        requests: Requests in slice order.
        model: Backend model id or alias.
        orders: Orderings from ``balanced_orders``; ordering 0 is canonical.
        single: Single-order receipt case by case id.
        clock: Monotonic clock in seconds.

    Returns:
        The case records, the stopping failure, the request count and the
        wall time.

    Raises:
        TypeError: When a request's verdict question is not a ``Choice``.
    """
    started = clock()
    records: list[dict[str, Any]] = []
    count = 0
    for index, request in enumerate(requests):
        base = request.questions[VERDICT]
        if not isinstance(base, Choice):
            msg = f"{request.case_id}: the verdict question is not a Choice"
            raise TypeError(msg)
        labels = tuple(base.criteria)
        probs: list[dict[str, float]] = []
        for number, order in enumerate(orders):
            question = reordered_choice(base, labels, order)
            try:
                response = port.judge(
                    request.state, {VERDICT: question}, model, media=request.media
                )
            except GenerationError as exc:
                stopped: dict[str, object] = {
                    "index": index,
                    "case_id": request.case_id,
                    "ordering": number,
                    "error_class": type(exc).__name__,
                    "message": str(exc),
                }
                return OrderingsRun(tuple(records), stopped, count, clock() - started)
            count += 1
            answer = response.choices[VERDICT].probabilities
            probs.append({label: answer[label] for label in labels})
        single_case = single[request.case_id]
        records.append(_record(request, labels, orders, probs, single_case))
    return OrderingsRun(tuple(records), None, count, clock() - started)


def build_orderings_receipt(
    run: OrderingsRun,
    *,
    backend: str,
    model: str,
    pins: Mapping[str, object],
    identity: Mapping[str, object],
) -> dict[str, Any]:
    """Build the key-free receipt body of one study run.

    Args:
        run: Result of ``run_orderings``.
        backend: Backend name.
        model: Model id sent to the backend.
        pins: Generator, Pillow and server pins.
        identity: Experiment identity mapping.

    Returns:
        JSON-ready receipt with the statistics and per-case records. The
        statistics are ``None`` when no case finished.
    """
    return {
        "issue": ORDERINGS_RECEIPT_ISSUE,
        "backend": backend,
        "model": model,
        "pins": dict(pins),
        "identity": dict(identity),
        "stopped": run.stopped,
        "requests": run.requests,
        "wall_seconds": round(run.wall_seconds, 3),
        "statistics": orderings_statistics(run.records) if run.records else None,
        "cases": list(run.records),
    }
