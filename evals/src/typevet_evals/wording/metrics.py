"""Held-out metrics, paired bootstrap and the pre-registered pass rule (#309).

Plain Python. Each function takes one probability per row in row order and
the row's label, 1 for ``scam`` and 0 for ``legit``. A probability of 0.5 or
more reads ``scam``. ECE uses ``ECE_BINS`` equal-width bins. ``cohen_kappa``
gives the agreement of the readings with the labels beyond chance (#329).

The pass rule is fixed before any held-out run: the evolved wording's ECE
drops by at least ``MIN_ECE_DROP`` from the seed wording and ends at most
``MAX_EVOLVED_ECE``; its Brier score drops; its accuracy drops by at most
``MAX_ACCURACY_DROP``. The paired bootstrap interval is context only and
does not change the rule. Resampling hashes the seed, the resample number
and the row position (no ``random``), so the intervals repeat exactly.
``bootstrap_differences`` holds that loop; a ``Choice`` run (#369) reuses it
and the pass rule, which reads any ``RuleMetrics``.

Attributes:
    POSITIVE_THRESHOLD (float): The probability at or above which a row reads scam.
    ECE_BINS (int): Equal-width bins for ECE.
    BOOTSTRAP_RESAMPLES (int): Default resamples of the paired bootstrap.
    BOOTSTRAP_SEED (int): Default seed of the paired bootstrap.
    INTERVAL_LEVEL (float): Default level of the bootstrap interval.
    MIN_ECE_DROP (float): The smallest ECE drop that passes.
    MAX_EVOLVED_ECE (float): The largest evolved ECE that passes.
    MAX_ACCURACY_DROP (float): The largest accuracy drop that passes.
    TOLERANCE (float): Float slack at the rule's boundaries.

Examples:
    ```python
    from typevet_evals.wording.metrics import pass_verdict, wording_metrics

    seed = wording_metrics([0.6, 0.4], [1, 0])
    evolved = wording_metrics([0.9, 0.1], [1, 0])
    assert pass_verdict(seed, evolved).passed
    ```

See Also:
    - [typevet_evals.wording.held_out][]: scores the held-out rows and builds
      the receipt
    - [typevet_evals.face_match.metrics][]: the ECE implementation
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final, Protocol

from typevet_evals.face_match.metrics import expected_calibration_error

POSITIVE_THRESHOLD: Final[float] = 0.5
ECE_BINS: Final[int] = 10
BOOTSTRAP_RESAMPLES: Final[int] = 2000
BOOTSTRAP_SEED: Final[int] = 0
INTERVAL_LEVEL: Final[float] = 0.95
MIN_ECE_DROP: Final[float] = 0.03
MAX_EVOLVED_ECE: Final[float] = 0.10
MAX_ACCURACY_DROP: Final[float] = 0.01
TOLERANCE: Final[float] = 1e-9


@dataclass(frozen=True, slots=True)
class WordingMetrics:
    """Accuracy, Brier score and ECE of one wording over the held-out rows.

    Attributes:
        rows (int): Rows scored.
        positives (int): Rows labelled scam.
        accuracy (float): Share of rows whose reading matches the label.
        brier (float): Mean squared error of the probability.
        ece (float): Expected calibration error over ``ECE_BINS`` bins.

    Examples:
        ```python
        WordingMetrics(rows=2, positives=1, accuracy=1.0, brier=0.01, ece=0.1)
        ```
    """

    rows: int
    positives: int
    accuracy: float
    brier: float
    ece: float

    def to_mapping(self) -> dict[str, float | int]:
        """Return the metrics as a JSON mapping.

        Returns:
            One key per attribute.
        """
        return {
            "rows": self.rows,
            "positives": self.positives,
            "accuracy": self.accuracy,
            "brier": self.brier,
            "ece": self.ece,
        }


def _check_rows(probabilities: Sequence[float], labels: Sequence[int]) -> None:
    if len(probabilities) != len(labels):
        msg = (
            f"length mismatch: {len(probabilities)} probabilities, {len(labels)} labels"
        )
        raise ValueError(msg)
    if not probabilities:
        msg = "cannot measure an empty set of rows"
        raise ValueError(msg)
    for label in labels:
        if label not in (0, 1):
            msg = f"label {label!r} is not 0 or 1"
            raise ValueError(msg)
    for p in probabilities:
        if math.isnan(p) or not 0.0 <= p <= 1.0:
            msg = f"probability {p!r} is not between 0 and 1"
            raise ValueError(msg)


def _brier(probabilities: Sequence[float], labels: Sequence[int]) -> float:
    return sum((p - y) ** 2 for p, y in zip(probabilities, labels, strict=True)) / len(
        labels
    )


def _ece(probabilities: Sequence[float], labels: Sequence[int]) -> float:
    return expected_calibration_error(
        probabilities, [y == 1 for y in labels], n_bins=ECE_BINS
    )


def wording_metrics(
    probabilities: Sequence[float], labels: Sequence[int]
) -> WordingMetrics:
    """Measure one wording over its rows.

    Args:
        probabilities: The scam probability, one per row.
        labels: 1 for scam, 0 for legit, one per row.

    Returns:
        The row counts, accuracy, Brier score and ECE.

    Raises:
        ValueError: When the lengths differ, there are no rows, a label is not
            0 or 1, or a probability is not between 0 and 1.
    """
    _check_rows(probabilities, labels)
    right = sum(
        1
        for p, y in zip(probabilities, labels, strict=True)
        if (p >= POSITIVE_THRESHOLD) == (y == 1)
    )
    return WordingMetrics(
        rows=len(labels),
        positives=sum(labels),
        accuracy=right / len(labels),
        brier=_brier(probabilities, labels),
        ece=_ece(probabilities, labels),
    )


def cohen_kappa(probabilities: Sequence[float], labels: Sequence[int]) -> float | None:
    """Return Cohen's kappa of the readings against the labels.

    A probability of ``POSITIVE_THRESHOLD`` or more reads scam. With observed
    agreement ``p_o`` (the accuracy), reading share ``r`` and label share
    ``s`` of scam, chance agreement is ``p_e = r * s + (1 - r) * (1 - s)``
    and kappa is ``(p_o - p_e) / (1 - p_e)``.

    Args:
        probabilities: The scam probability, one per row.
        labels: 1 for scam, 0 for legit, one per row.

    Returns:
        Kappa, from -1 to 1; None when ``p_e`` is 1 (the readings and the
        labels are one and the same class), where kappa is undefined.

    Raises:
        ValueError: When the lengths differ, there are no rows, a label is not
            0 or 1, or a probability is not between 0 and 1.
    """
    _check_rows(probabilities, labels)
    rows = len(labels)
    readings = [int(p >= POSITIVE_THRESHOLD) for p in probabilities]
    observed = sum(r == y for r, y in zip(readings, labels, strict=True)) / rows
    read_share = sum(readings) / rows
    label_share = sum(labels) / rows
    chance = read_share * label_share + (1 - read_share) * (1 - label_share)
    if chance >= 1.0:
        return None
    return (observed - chance) / (1 - chance)


def resample_indices(n: int, *, seed: int, resample: int) -> list[int]:
    """Return the row positions of one bootstrap resample.

    Position ``i`` of resample ``b`` is the first 8 bytes of
    ``sha256(f"{seed}:{b}:{i}")``, read big-endian, modulo ``n``.

    Args:
        n: Rows in the sample.
        seed: The bootstrap seed.
        resample: The resample number.

    Returns:
        ``n`` positions, each in ``range(n)``, drawn with replacement.
    """
    return [
        int.from_bytes(
            hashlib.sha256(f"{seed}:{resample}:{i}".encode()).digest()[:8], "big"
        )
        % n
        for i in range(n)
    ]


@dataclass(frozen=True, slots=True)
class Interval:
    """A two-sided interval.

    Attributes:
        low (float): Lower end.
        high (float): Upper end.

    Examples:
        ```python
        Interval(-0.05, 0.01)
        ```
    """

    low: float
    high: float


def percentile_interval(values: Sequence[float], *, level: float) -> Interval:
    """Return the percentile interval of ``values`` at ``level``.

    With ``B`` sorted values, the ends are the values at ranks
    ``floor((1 - level) / 2 * B)`` and ``ceil((1 + level) / 2 * B) - 1``
    (from 0). For 2,000 values at 0.95 these are ranks 50 and 1949.

    Args:
        values: Resampled statistics.
        level: The interval level, for example 0.95.

    Returns:
        The lower and upper ends.

    Raises:
        ValueError: When ``values`` is empty.
    """
    if not values:
        msg = "cannot take an interval of an empty sequence"
        raise ValueError(msg)
    ordered = sorted(values)
    count = len(ordered)
    low = math.floor(round((1 - level) / 2 * count, 9))
    high = math.ceil(round((1 + level) / 2 * count, 9)) - 1
    return Interval(ordered[low], ordered[min(high, count - 1)])


@dataclass(frozen=True, slots=True)
class PairedBootstrap:
    """Paired bootstrap intervals of evolved minus seed ECE and Brier score.

    Attributes:
        resamples (int): Resamples drawn.
        seed (int): The bootstrap seed.
        level (float): The interval level.
        ece_difference (Interval): Interval of evolved ECE minus seed ECE.
        brier_difference (Interval): Interval of evolved Brier minus seed Brier.

    Examples:
        ```python
        boot = paired_bootstrap([0.5, 0.5], [0.9, 0.1], [1, 0])
        boot.brier_difference.high < 0
        ```
    """

    resamples: int
    seed: int
    level: float
    ece_difference: Interval
    brier_difference: Interval

    def to_mapping(self) -> dict[str, object]:
        """Return the intervals as a JSON mapping.

        Returns:
            The settings and both intervals as ``{"low": ..., "high": ...}``.
        """
        return {
            "resamples": self.resamples,
            "seed": self.seed,
            "level": self.level,
            "ece_difference": {
                "low": self.ece_difference.low,
                "high": self.ece_difference.high,
            },
            "brier_difference": {
                "low": self.brier_difference.low,
                "high": self.brier_difference.high,
            },
        }


def paired_bootstrap(
    seed_probabilities: Sequence[float],
    evolved_probabilities: Sequence[float],
    labels: Sequence[int],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    level: float = INTERVAL_LEVEL,
) -> PairedBootstrap:
    """Resample rows with replacement and measure both wordings on each resample.

    Each resample takes the same rows for both wordings, so the differences
    are paired. ``bootstrap_differences`` draws the resamples.

    Args:
        seed_probabilities: The seed wording's probability, one per row.
        evolved_probabilities: The evolved wording's probability, one per row.
        labels: 1 for scam, 0 for legit, one per row.
        resamples: Resamples to draw.
        seed: The bootstrap seed.
        level: The interval level.

    Returns:
        Percentile intervals of the ECE and Brier differences, evolved minus seed.

    Raises:
        ValueError: When the three sequences differ in length or hold bad rows.
    """
    _check_rows(seed_probabilities, labels)
    _check_rows(evolved_probabilities, labels)
    arms = (seed_probabilities, evolved_probabilities)

    def ece(rows: Sequence[int], arm: int) -> float:
        """Return one arm's ECE over the rows at ``rows``.

        Args:
            rows: Row positions.
            arm: 0 for the seed wording, 1 for the evolved wording.

        Returns:
            The ECE.
        """
        return _ece([arms[arm][i] for i in rows], [labels[i] for i in rows])

    def brier(rows: Sequence[int], arm: int) -> float:
        """Return one arm's Brier score over the rows at ``rows``.

        Args:
            rows: Row positions.
            arm: 0 for the seed wording, 1 for the evolved wording.

        Returns:
            The Brier score.
        """
        return _brier([arms[arm][i] for i in rows], [labels[i] for i in rows])

    return bootstrap_differences(
        len(labels), ece, brier, resamples=resamples, seed=seed, level=level
    )


def bootstrap_differences(
    n: int,
    ece: Callable[[Sequence[int], int], float],
    brier: Callable[[Sequence[int], int], float],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    level: float = INTERVAL_LEVEL,
) -> PairedBootstrap:
    """Resample row positions and bound the evolved minus seed differences.

    ``paired_bootstrap`` and the ``Choice`` bootstrap share this loop, so
    both draw the same resamples for the same row count and seed.

    Args:
        n: Rows in the sample.
        ece: The ECE of the rows at the given positions, for arm 0 (seed)
            or arm 1 (evolved).
        brier: The Brier score of the rows at the given positions, per arm.
        resamples: Resamples to draw.
        seed: The bootstrap seed.
        level: The interval level.

    Returns:
        Percentile intervals of the ECE and Brier differences, evolved minus seed.
    """
    ece_differences: list[float] = []
    brier_differences: list[float] = []
    for b in range(resamples):
        rows = resample_indices(n, seed=seed, resample=b)
        ece_differences.append(ece(rows, 1) - ece(rows, 0))
        brier_differences.append(brier(rows, 1) - brier(rows, 0))
    return PairedBootstrap(
        resamples=resamples,
        seed=seed,
        level=level,
        ece_difference=percentile_interval(ece_differences, level=level),
        brier_difference=percentile_interval(brier_differences, level=level),
    )


@dataclass(frozen=True, slots=True)
class PassVerdict:
    """The pre-registered pass rule applied to seed and evolved metrics.

    Attributes:
        ece_drop (float): Seed ECE minus evolved ECE.
        ece_drop_ok (bool): ``ece_drop`` is at least ``MIN_ECE_DROP``.
        evolved_ece_ok (bool): Evolved ECE is at most ``MAX_EVOLVED_ECE``.
        brier_drops (bool): Evolved Brier is below seed Brier.
        accuracy_drop (float): Seed accuracy minus evolved accuracy.
        accuracy_ok (bool): ``accuracy_drop`` is at most ``MAX_ACCURACY_DROP``.
        passed (bool): Every clause holds.

    Examples:
        ```python
        verdict = pass_verdict(seed_metrics, evolved_metrics)
        verdict.passed
        ```
    """

    ece_drop: float
    ece_drop_ok: bool
    evolved_ece_ok: bool
    brier_drops: bool
    accuracy_drop: float
    accuracy_ok: bool
    passed: bool

    def to_mapping(self) -> dict[str, object]:
        """Return the verdict and the rule's limits as a JSON mapping.

        Returns:
            One key per attribute and the three limits.
        """
        return {
            "ece_drop": self.ece_drop,
            "ece_drop_ok": self.ece_drop_ok,
            "evolved_ece_ok": self.evolved_ece_ok,
            "brier_drops": self.brier_drops,
            "accuracy_drop": self.accuracy_drop,
            "accuracy_ok": self.accuracy_ok,
            "passed": self.passed,
            "rule": {
                "min_ece_drop": MIN_ECE_DROP,
                "max_evolved_ece": MAX_EVOLVED_ECE,
                "max_accuracy_drop": MAX_ACCURACY_DROP,
                "ece_bins": ECE_BINS,
            },
        }


class RuleMetrics(Protocol):
    """The metrics the pass rule reads: accuracy, Brier score and ECE.

    ``WordingMetrics`` and the ``Choice`` metrics both have this shape.

    Attributes:
        accuracy (float): Share of rows read right.
        brier (float): Mean Brier score.
        ece (float): Expected calibration error.

    Examples:
        ```python
        metrics: RuleMetrics = wording_metrics([0.9], [1])
        ```
    """

    @property
    def accuracy(self) -> float:
        """Return the accuracy."""
        ...

    @property
    def brier(self) -> float:
        """Return the Brier score."""
        ...

    @property
    def ece(self) -> float:
        """Return the ECE."""
        ...


def pass_verdict(seed: RuleMetrics, evolved: RuleMetrics) -> PassVerdict:
    """Apply the pre-registered pass rule.

    The ECE and accuracy limits allow ``TOLERANCE`` of float slack, so a value
    on a boundary passes. The Brier clause is strict: an equal Brier score
    does not drop.

    Args:
        seed: The seed wording's held-out metrics: ``WordingMetrics`` or the
            ``Choice`` metrics.
        evolved: The evolved wording's held-out metrics, of the same kind.

    Returns:
        Each clause and the overall verdict.
    """
    ece_drop = seed.ece - evolved.ece
    accuracy_drop = seed.accuracy - evolved.accuracy
    ece_drop_ok = ece_drop >= MIN_ECE_DROP - TOLERANCE
    evolved_ece_ok = evolved.ece <= MAX_EVOLVED_ECE + TOLERANCE
    brier_drops = evolved.brier < seed.brier
    accuracy_ok = accuracy_drop <= MAX_ACCURACY_DROP + TOLERANCE
    return PassVerdict(
        ece_drop=ece_drop,
        ece_drop_ok=ece_drop_ok,
        evolved_ece_ok=evolved_ece_ok,
        brier_drops=brier_drops,
        accuracy_drop=accuracy_drop,
        accuracy_ok=accuracy_ok,
        passed=ece_drop_ok and evolved_ece_ok and brier_drops and accuracy_ok,
    )
