"""Post-hoc calibration of Noul probabilities on committed receipts (#343).

Plain Python, offline. Each series is one binary probability column of one
committed receipt. The split puts a row in the calibration half when
``int(sha256(id).hexdigest(), 16) % 2 == 0``, else in the evaluation half.
The fitters see the calibration half only. The metrics use the evaluation
half.

The fitters clip each probability to ``[CLIP, 1 - CLIP]`` first.
Temperature scaling maps ``p`` to ``sigmoid(logit(p) / T)``. A golden-section
search on ``log T`` in ``[-3, 3]`` minimises the negative log-likelihood.
Platt scaling maps ``p`` to ``sigmoid(a * logit(p) + b)``. Newton steps with
step halving minimise the same loss. Isotonic regression pools adjacent
violators over the sorted probabilities. Tied probabilities share one block.
Between knots it returns the value of the nearest lower knot. Below the
first knot it returns the first knot value.

ECE reuses [typevet_evals.face_match.metrics.expected_calibration_error][]
with 10 equal-width bins. The "before" metrics use the unclipped receipt
probabilities. Accuracy reads a probability of 0.5 or more as positive. The
bootstrap reuses the sha256 resampling of [typevet_evals.wording.metrics][]
(seed 0, no ``random``) and the same percentile interval. The interval is
context only and does not change the rule.

The rule is fixed before any fit (issue #343 comment 5919241864). A method
meets it on a series when ECE drops by at least ``MIN_ECE_DROP``, the Brier
score drops, and accuracy drops by at most ``MAX_ACCURACY_DROP``. The
decision uses the six ``DECISION_SERIES`` only.

Attributes:
    CLIP (float): Clip distance from 0 and 1 before a fit.
    MIN_ECE_DROP (float): The smallest ECE drop that meets the rule.
    MAX_ACCURACY_DROP (float): The largest accuracy drop that meets the rule.
    MAX_DECISION_ECE (float): The largest evaluation-half ECE after a fit
        that counts toward the decision; also the exclusion floor.
    MIN_DECISION_SERIES (int): Decision series one method must fix.
    TOLERANCE (float): Float slack at the rule's boundaries.
    BOOTSTRAP_RESAMPLES (int): Resamples of the ECE-change bootstrap.
    POSITIVE_THRESHOLD (float): The probability at or above which a row
        reads positive.
    DECISION_SERIES (tuple[str, ...]): The six series with ECE at or above
        0.10 in the committed receipts.
    TRANSFER_PAIRS (tuple[tuple[str, str], ...]): Source and target series
        of the transfer rows.

Examples:
    ```python
    from typevet_evals.calibration import fit_temperature

    scaler = fit_temperature([0.99, 0.01, 0.99, 0.01], [True, False, False, True])
    assert scaler.temperature > 1.0
    ```

See Also:
    - [typevet_evals.face_match.metrics][]: the ECE implementation
    - [typevet_evals.wording.metrics][]: bootstrap resampling and intervals
"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from typevet_evals.face_match.metrics import expected_calibration_error
from typevet_evals.wording.metrics import percentile_interval, resample_indices

CLIP: Final = 1e-6
MIN_ECE_DROP: Final = 0.03
MAX_ACCURACY_DROP: Final = 0.01
MAX_DECISION_ECE: Final = 0.05
MIN_DECISION_SERIES: Final = 4
TOLERANCE: Final = 1e-9
BOOTSTRAP_RESAMPLES: Final = 1000
POSITIVE_THRESHOLD: Final = 0.5
_CONVERGED: Final = 1e-12
DECISION_SERIES: Final = (
    "difraud/wording252_held_out_gemma_llama_cpp/seed",
    "difraud/wording252_held_out_gemma_vllm/seed",
    "difraud/wording_held_out_vllm/seed",
    "difraud/wording_held_out_vllm/evolved",
    "signatures/signature_match_llama_cpp",
    "signatures/signature_match_vllm",
)
TRANSFER_PAIRS: Final = (
    ("difraud/wording_held_out_llama_cpp/seed", "signatures/signature_match_llama_cpp"),
    ("signatures/signature_match_llama_cpp", "difraud/wording_held_out_llama_cpp/seed"),
)
_METHODS = ("temperature", "platt", "isotonic")


def _logit(p: float) -> float:
    p = min(max(p, CLIP), 1.0 - CLIP)
    return math.log(p / (1.0 - p))


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def _nll(xs: Sequence[float], labels: Sequence[bool], a: float, b: float) -> float:
    total = 0.0
    for x, y in zip(xs, labels, strict=True):
        z = a * x + b
        total += max(z, 0.0) + math.log1p(math.exp(-abs(z))) - (z if y else 0.0)
    return total


def in_calibration_half(item_id: str) -> bool:
    """Return whether a row id falls in the calibration half.

    Args:
        item_id: The row id from the receipt.

    Returns:
        ``True`` when ``int(sha256(id).hexdigest(), 16)`` is even.
    """
    return int(hashlib.sha256(item_id.encode()).hexdigest(), 16) % 2 == 0


def split_indices(ids: Sequence[str]) -> tuple[list[int], list[int]]:
    """Split row positions into the calibration and evaluation halves.

    Args:
        ids: Row ids in row order.

    Returns:
        Calibration positions and evaluation positions, each ascending.
    """
    calibration = [i for i, item in enumerate(ids) if in_calibration_half(item)]
    chosen = set(calibration)
    return calibration, [i for i in range(len(ids)) if i not in chosen]


@dataclass(frozen=True, slots=True)
class TemperatureScaler:
    """Map ``p`` to ``sigmoid(logit(p) / temperature)``.

    Attributes:
        temperature (float): The fitted temperature ``T``.

    Examples:
        ```python
        assert TemperatureScaler(1.0)(0.7) == pytest.approx(0.7)
        ```
    """

    temperature: float

    def __call__(self, p: float) -> float:
        """Return the calibrated probability.

        Args:
            p: A probability in [0, 1].

        Returns:
            The scaled probability.
        """
        return _sigmoid(_logit(p) / self.temperature)

    def params(self) -> dict[str, float]:
        """Return the fitted parameters.

        Returns:
            ``{"temperature": T}``.
        """
        return {"temperature": self.temperature}


@dataclass(frozen=True, slots=True)
class PlattScaler:
    """Map ``p`` to ``sigmoid(slope * logit(p) + intercept)``.

    Attributes:
        slope (float): The fitted ``a``.
        intercept (float): The fitted ``b``.

    Examples:
        ```python
        assert PlattScaler(1.0, 0.0)(0.7) == pytest.approx(0.7)
        ```
    """

    slope: float
    intercept: float

    def __call__(self, p: float) -> float:
        """Return the calibrated probability.

        Args:
            p: A probability in [0, 1].

        Returns:
            The scaled probability.
        """
        return _sigmoid(self.slope * _logit(p) + self.intercept)

    def params(self) -> dict[str, float]:
        """Return the fitted parameters.

        Returns:
            ``{"slope": a, "intercept": b}``.
        """
        return {"slope": self.slope, "intercept": self.intercept}


@dataclass(frozen=True, slots=True)
class IsotonicMap:
    """A non-decreasing step map fitted by pool-adjacent-violators.

    Attributes:
        knots (tuple[float, ...]): Distinct clipped probabilities, ascending.
        values (tuple[float, ...]): The fitted value at each knot.

    Examples:
        ```python
        assert IsotonicMap((0.2, 0.8), (0.0, 1.0))(0.5) == 0.0
        ```
    """

    knots: tuple[float, ...]
    values: tuple[float, ...]

    def __call__(self, p: float) -> float:
        """Return the value of the nearest knot at or below ``p``.

        Args:
            p: A probability in [0, 1].

        Returns:
            The step value; the first knot value below the first knot.
        """
        index = bisect.bisect_right(self.knots, min(max(p, CLIP), 1.0 - CLIP)) - 1
        return self.values[max(index, 0)]

    def params(self) -> dict[str, int]:
        """Return the size of the fitted map.

        Returns:
            ``{"knots": count, "steps": distinct values}``.
        """
        return {"knots": len(self.knots), "steps": len(set(self.values))}


Calibrator = TemperatureScaler | PlattScaler | IsotonicMap


def fit_temperature(
    probabilities: Sequence[float], labels: Sequence[bool]
) -> TemperatureScaler:
    """Fit ``T`` by golden-section search on ``log T`` in [-3, 3].

    Args:
        probabilities: Calibration-half probabilities.
        labels: Calibration-half labels.

    Returns:
        The fitted scaler.
    """
    xs = [_logit(p) for p in probabilities]
    ratio = (math.sqrt(5.0) - 1.0) / 2.0
    low, high = -3.0, 3.0
    for _ in range(100):
        left = high - ratio * (high - low)
        right = low + ratio * (high - low)
        if _nll(xs, labels, math.exp(-left), 0.0) <= _nll(
            xs, labels, math.exp(-right), 0.0
        ):
            high = right
        else:
            low = left
    return TemperatureScaler(math.exp((low + high) / 2.0))


def fit_platt(probabilities: Sequence[float], labels: Sequence[bool]) -> PlattScaler:
    """Fit ``a`` and ``b`` by Newton steps with step halving.

    At most 200 steps run. A step that does not lower the loss after 40
    halvings ends the fit, so separable data gives a large but finite slope.

    Args:
        probabilities: Calibration-half probabilities.
        labels: Calibration-half labels.

    Returns:
        The fitted scaler.
    """
    xs = [_logit(p) for p in probabilities]
    a, b = 1.0, 0.0
    loss = _nll(xs, labels, a, b)
    for _ in range(200):
        ga = gb = haa = hab = hbb = 0.0
        for x, y in zip(xs, labels, strict=True):
            s = _sigmoid(a * x + b)
            w = s * (1.0 - s) + 1e-12
            ga += (s - y) * x
            gb += s - y
            haa, hab, hbb = haa + w * x * x, hab + w * x, hbb + w
        det = haa * hbb - hab * hab
        da, db = (hbb * ga - hab * gb) / det, (haa * gb - hab * ga) / det
        step = 1.0
        for _ in range(40):
            trial = _nll(xs, labels, a - step * da, b - step * db)
            if trial < loss:
                break
            step /= 2.0
        else:
            break
        a, b = a - step * da, b - step * db
        if loss - trial < _CONVERGED:
            break
        loss = trial
    return PlattScaler(a, b)


def fit_isotonic(probabilities: Sequence[float], labels: Sequence[bool]) -> IsotonicMap:
    """Fit a non-decreasing step map by pool-adjacent-violators.

    Args:
        probabilities: Calibration-half probabilities.
        labels: Calibration-half labels.

    Returns:
        The fitted map.
    """
    groups: dict[float, list[float]] = {}
    for p, y in zip(probabilities, labels, strict=True):
        groups.setdefault(min(max(p, CLIP), 1.0 - CLIP), []).append(float(y))
    knots = sorted(groups)
    blocks: list[list[float]] = []  # [label sum, count, knots in block]
    for knot in knots:
        blocks.append([sum(groups[knot]), float(len(groups[knot])), 1.0])
        while len(blocks) > 1 and (
            blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]
        ):
            last = blocks.pop()
            blocks[-1] = [x + y for x, y in zip(blocks[-1], last, strict=True)]
    values = [b[0] / b[1] for b in blocks for _ in range(int(b[2]))]
    return IsotonicMap(tuple(knots), tuple(values))


def brier_score(probabilities: Sequence[float], labels: Sequence[bool]) -> float:
    """Return the mean squared error of the probabilities.

    Args:
        probabilities: One probability per row.
        labels: One label per row.

    Returns:
        The Brier score.
    """
    pairs = zip(probabilities, labels, strict=True)
    return sum((p - float(y)) ** 2 for p, y in pairs) / len(labels)


def accuracy_at_half(probabilities: Sequence[float], labels: Sequence[bool]) -> float:
    """Return the share of rows whose reading at 0.5 matches the label.

    Args:
        probabilities: One probability per row.
        labels: One label per row.

    Returns:
        Accuracy; a probability of 0.5 or more reads positive.
    """
    pairs = zip(probabilities, labels, strict=True)
    return sum(1 for p, y in pairs if (p >= POSITIVE_THRESHOLD) == bool(y)) / len(
        labels
    )


def bootstrap_ece_change(
    probabilities: Sequence[float],
    labels: Sequence[bool],
    calibrator: Callable[[float], float],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> tuple[float, float]:
    """Return the 95% interval of ECE after minus ECE before.

    Args:
        probabilities: Evaluation-half probabilities.
        labels: Evaluation-half labels.
        calibrator: A calibrator already fitted on the calibration half.
        resamples: Resamples drawn with seed 0.

    Returns:
        Lower and upper ends of the percentile interval.
    """
    after = [calibrator(p) for p in probabilities]
    changes = []
    for b in range(resamples):
        rows = resample_indices(len(labels), seed=0, resample=b)
        gold = [labels[i] for i in rows]
        changes.append(
            expected_calibration_error([after[i] for i in rows], gold)
            - expected_calibration_error([probabilities[i] for i in rows], gold)
        )
    interval = percentile_interval(changes, level=0.95)
    return interval.low, interval.high


def rule_met(before: dict[str, float], after: dict[str, float]) -> bool:
    """Return whether one fit meets the pre-registered rule.

    Args:
        before: ``ece``, ``brier`` and ``accuracy`` before the fit.
        after: The same metrics after the fit.

    Returns:
        ``True`` when ECE drops by ``MIN_ECE_DROP`` or more, Brier drops and
        accuracy drops by ``MAX_ACCURACY_DROP`` or less.
    """
    return (
        before["ece"] - after["ece"] >= MIN_ECE_DROP - TOLERANCE
        and after["brier"] < before["brier"]
        and before["accuracy"] - after["accuracy"] <= MAX_ACCURACY_DROP + TOLERANCE
    )


@dataclass(frozen=True, slots=True)
class Series:
    """One binary probability column of one committed receipt.

    The #343 unit test builds the 16 series from the committed receipts.

    Attributes:
        name (str): ``family/receipt[/column]``.
        source (str): Receipt path relative to the fixtures directory.
        ids (tuple[str, ...]): Row ids in receipt order.
        probabilities (tuple[float, ...]): Probability of the positive label.
        labels (tuple[bool, ...]): Gold labels.

    Examples:
        ```python
        Series("x/y", "x/y.json", ("a",), (0.9,), (True,))
        ```
    """

    name: str
    source: str
    ids: tuple[str, ...]
    probabilities: tuple[float, ...]
    labels: tuple[bool, ...]


def _metrics(
    probabilities: Sequence[float], labels: Sequence[bool]
) -> dict[str, float]:
    return {
        "ece": expected_calibration_error(probabilities, labels),
        "brier": brier_score(probabilities, labels),
        "accuracy": accuracy_at_half(probabilities, labels),
    }


def _fit(
    method: str, probabilities: Sequence[float], labels: Sequence[bool]
) -> Calibrator:
    fitters = {
        "temperature": fit_temperature,
        "platt": fit_platt,
        "isotonic": fit_isotonic,
    }
    return fitters[method](probabilities, labels)


def _halves(series: Series) -> tuple[list[float], list[bool], list[float], list[bool]]:
    cal, ev = split_indices(series.ids)
    p, y = series.probabilities, series.labels
    return (
        [p[i] for i in cal],
        [y[i] for i in cal],
        [p[i] for i in ev],
        [y[i] for i in ev],
    )


def _analyse(series: Series) -> dict[str, Any]:
    cal_p, cal_y, ev_p, ev_y = _halves(series)
    before = _metrics(ev_p, ev_y)
    methods = {}
    for method in _METHODS:
        calibrator = _fit(method, cal_p, cal_y)
        after = _metrics([calibrator(p) for p in ev_p], ev_y)
        low, high = bootstrap_ece_change(ev_p, ev_y, calibrator)
        methods[method] = {
            "params": calibrator.params(),
            "after": after,
            "ece_change": after["ece"] - before["ece"],
            "ece_change_interval": {"low": low, "high": high, "level": 0.95},
            "rule_met": rule_met(before, after),
        }
    return {
        "name": series.name,
        "source": series.source,
        "n": len(series.ids),
        "n_calibration": len(cal_p),
        "n_evaluation": len(ev_p),
        "before": before,
        "excluded_from_decision": before["ece"] < MAX_DECISION_ECE,
        "in_decision_set": series.name in DECISION_SERIES,
        "methods": methods,
    }


def _transfer(source: Series, target: Series) -> dict[str, Any]:
    cal_p, cal_y, _, _ = _halves(source)
    _, _, ev_p, ev_y = _halves(target)
    before = _metrics(ev_p, ev_y)
    methods = {}
    for method in _METHODS:
        after = _metrics([_fit(method, cal_p, cal_y)(p) for p in ev_p], ev_y)
        methods[method] = {"after": after, "rule_met": rule_met(before, after)}
    return {
        "fit_on": source.name,
        "applied_to": target.name,
        "before": before,
        "methods": methods,
    }


def _decision(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fixed: dict[str, list[str]] = {}
    for method in _METHODS:
        fixed[method] = [
            r["name"]
            for r in rows
            if r["in_decision_set"]
            and not r["excluded_from_decision"]
            and r["methods"][method]["rule_met"]
            and r["methods"][method]["after"]["ece"] < MAX_DECISION_ECE
        ]
    sufficient = any(len(v) >= MIN_DECISION_SERIES for v in fixed.values())
    return {
        "decision_series": list(DECISION_SERIES),
        "fixed_by_method": fixed,
        "required": MIN_DECISION_SERIES,
        "verdict": "sufficient" if sufficient else "consider_fine_tuning",
    }


def post_hoc_receipt(series: Sequence[Series], fixtures: Path) -> dict[str, Any]:
    """Run the whole #343 analysis on series built from committed receipts.

    Args:
        series: The series, in contract order. Each ``source`` is a path
            relative to ``fixtures``.
        fixtures: The ``evals/fixtures`` directory, for the source digests.

    Returns:
        The receipt: sources with digests, one row per series, the transfer
        rows and the decision.
    """
    by_name = {s.name: s for s in series}
    rows = [_analyse(s) for s in series]
    sources = sorted({s.source for s in series})
    return {
        "issue": 343,
        "contract": "https://github.com/Alberto-Codes/typevet/issues/343#issuecomment-5919241864",
        "sources": {
            src: hashlib.sha256((fixtures / src).read_bytes()).hexdigest()
            for src in sources
        },
        "rule": {
            "min_ece_drop": MIN_ECE_DROP,
            "max_accuracy_drop": MAX_ACCURACY_DROP,
            "brier_must_drop": True,
            "max_decision_ece": MAX_DECISION_ECE,
            "min_decision_series": MIN_DECISION_SERIES,
            "ece_bins": 10,
            "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": 0, "level": 0.95},
        },
        "series": rows,
        "transfer": [_transfer(by_name[a], by_name[b]) for a, b in TRANSFER_PAIRS],
        "decision": _decision(rows),
    }


def _rounded(value: object) -> object:
    if isinstance(value, float):
        return round(value, 12)
    if isinstance(value, dict):
        return {k: _rounded(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_rounded(v) for v in value]
    return value


def render_receipt(receipt: dict[str, Any]) -> str:
    """Return the receipt as stable JSON text.

    Floats are rounded to 12 decimals and keys are sorted, so a rerun on the
    same inputs gives the same bytes.

    Args:
        receipt: The mapping from ``post_hoc_receipt``.

    Returns:
        Indented JSON with a final newline.
    """
    return json.dumps(_rounded(receipt), indent=2, sort_keys=True) + "\n"
