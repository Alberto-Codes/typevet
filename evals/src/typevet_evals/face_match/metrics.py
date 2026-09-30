"""Face-match metrics over typed answers and gold labels (#301).

Plain Python, no third-party statistics package. Every function takes one
value per pair in pair order and the gold same-person label for that pair.

Attributes:
    CANNOT_TELL (str): ``Choice`` label that never counts as a right verdict.
    DEFAULT_BINS (int): Equal-width bins in the reliability table.

Examples:
    ```python
    from typevet_evals.face_match.metrics import roc_auc, verdict_accuracy

    assert roc_auc([0.9, 0.1], [True, False]) == 1.0
    assert verdict_accuracy(["cannot_tell"], [True]) == 0.0
    ```

See Also:
    - [typevet_evals.face_match.runner][]: collects the answers and the receipt
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

CANNOT_TELL: Final[str] = "cannot_tell"
DEFAULT_BINS: Final[int] = 10

_RIGHT_VERDICT: Final[dict[bool, str]] = {
    True: "same_person",
    False: "different_person",
}
_KNOWN_VERDICTS: Final[frozenset[str]] = frozenset(
    {*_RIGHT_VERDICT.values(), CANNOT_TELL}
)
_LABEL_NAMES: Final[dict[bool, str]] = _RIGHT_VERDICT


@dataclass(frozen=True, slots=True)
class ReliabilityBin:
    """One equal-width bin of model confidence against gold labels.

    Attributes:
        lower (float): Inclusive lower edge.
        upper (float): Upper edge, inclusive only for the last bin.
        count (int): Pairs in the bin.
        mean_confidence (float | None): Mean confidence; ``None`` when empty.
        fraction_same_person (float | None): Share of same-person gold labels;
            ``None`` when empty.

    Examples:
        ```python
        ReliabilityBin(0.9, 1.0, 1, 0.95, 1.0)
        ```
    """

    lower: float
    upper: float
    count: int
    mean_confidence: float | None
    fraction_same_person: float | None


def _check_lengths(values: Sequence[object], gold: Sequence[bool]) -> None:
    if len(values) != len(gold):
        msg = f"values and gold must have equal length: {len(values)} != {len(gold)}"
        raise ValueError(msg)


def _check_not_empty(values: Sequence[object]) -> None:
    if not values:
        msg = "cannot compute a rate over an empty sequence"
        raise ValueError(msg)


def verdict_accuracy(verdicts: Sequence[str], gold: Sequence[bool]) -> float:
    """Return the share of right ``Choice`` verdicts.

    ``cannot_tell`` is always wrong.

    Args:
        verdicts: ``Choice`` labels, one per pair.
        gold: Gold same-person labels.

    Returns:
        Accuracy in [0, 1].

    Raises:
        ValueError: When lengths differ, the input is empty or a label is
            unknown.
    """
    _check_lengths(verdicts, gold)
    _check_not_empty(verdicts)
    unknown = sorted(set(verdicts) - _KNOWN_VERDICTS)
    if unknown:
        msg = f"unknown verdict labels: {', '.join(unknown)}"
        raise ValueError(msg)
    right = sum(
        1 for v, y in zip(verdicts, gold, strict=True) if v == _RIGHT_VERDICT[y]
    )
    return right / len(verdicts)


def cannot_tell_rate(verdicts: Sequence[str]) -> float:
    """Return the share of ``cannot_tell`` verdicts.

    Args:
        verdicts: ``Choice`` labels, one per pair.

    Returns:
        Rate in [0, 1].

    Raises:
        ValueError: When the input is empty.
    """
    _check_not_empty(verdicts)
    return sum(1 for v in verdicts if v == CANNOT_TELL) / len(verdicts)


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and values[order[end + 1]] == values[order[start]]:
            end += 1
        average = (start + end) / 2 + 1
        for position in range(start, end + 1):
            ranks[order[position]] = average
        start = end + 1
    return ranks


def roc_auc(confidences: Sequence[float], gold: Sequence[bool]) -> float | None:
    """Return ROC-AUC by the Mann-Whitney U statistic with average ranks.

    A tie between a same-person and a different-person pair counts one half.

    Args:
        confidences: Same-person confidence, one per pair.
        gold: Gold same-person labels.

    Returns:
        AUC in [0, 1], or ``None`` when one class has no pairs.

    Raises:
        ValueError: When lengths differ.
    """
    _check_lengths(confidences, gold)
    positives = sum(1 for y in gold if y)
    negatives = len(gold) - positives
    if positives == 0 or negatives == 0:
        return None
    ranks = _average_ranks(confidences)
    rank_sum = sum(r for r, y in zip(ranks, gold, strict=True) if y)
    u_statistic = rank_sum - positives * (positives + 1) / 2
    return u_statistic / (positives * negatives)


def reliability_table(
    confidences: Sequence[float], gold: Sequence[bool], n_bins: int = DEFAULT_BINS
) -> tuple[ReliabilityBin, ...]:
    """Bin confidences into equal-width bins against gold labels.

    Args:
        confidences: Same-person confidence in [0, 1], one per pair.
        gold: Gold same-person labels.
        n_bins: Number of bins.

    Returns:
        ``n_bins`` bins in ascending order. A value of 1.0 goes to the last bin.

    Raises:
        ValueError: When lengths differ, ``n_bins`` < 1 or a value is out of range.
    """
    _check_lengths(confidences, gold)
    if n_bins < 1:
        msg = "n_bins must be at least 1"
        raise ValueError(msg)
    buckets: list[list[tuple[float, bool]]] = [[] for _ in range(n_bins)]
    for p, y in zip(confidences, gold, strict=True):
        if not 0.0 <= p <= 1.0:
            msg = f"confidence {p} is not in [0, 1]"
            raise ValueError(msg)
        buckets[min(int(p * n_bins), n_bins - 1)].append((p, y))
    table = []
    for index, items in enumerate(buckets):
        count = len(items)
        mean = sum(p for p, _ in items) / count if count else None
        fraction = sum(1 for _, y in items if y) / count if count else None
        table.append(
            ReliabilityBin(index / n_bins, (index + 1) / n_bins, count, mean, fraction)
        )
    return tuple(table)


def expected_calibration_error(
    confidences: Sequence[float], gold: Sequence[bool], n_bins: int = DEFAULT_BINS
) -> float:
    """Return the count-weighted gap between confidence and gold frequency.

    Args:
        confidences: Same-person confidence in [0, 1], one per pair.
        gold: Gold same-person labels.
        n_bins: Number of equal-width bins.

    Returns:
        ECE in [0, 1]; 0.0 when there are no pairs.
    """
    table = reliability_table(confidences, gold, n_bins)
    if not confidences:
        return 0.0
    gap = sum(
        b.count * abs(b.mean_confidence - b.fraction_same_person)
        for b in table
        if b.mean_confidence is not None and b.fraction_same_person is not None
    )
    return gap / len(confidences)


def score_distribution(
    levels: Sequence[int], gold: Sequence[bool], *, n_levels: int
) -> dict[str, dict[str, int]]:
    """Count ``Score`` levels for each gold label.

    Args:
        levels: Most probable ``Score`` level, one per pair.
        gold: Gold same-person labels.
        n_levels: Number of levels, from 0 to ``n_levels - 1``.

    Returns:
        ``{"same_person": {...}, "different_person": {...}}`` with a count for
        every level, keyed by the level as text.

    Raises:
        ValueError: When lengths differ or a level is out of range.
    """
    _check_lengths(levels, gold)
    table = {
        name: {str(level): 0 for level in range(n_levels)}
        for name in (_LABEL_NAMES[True], _LABEL_NAMES[False])
    }
    for level, y in zip(levels, gold, strict=True):
        if not 0 <= level < n_levels:
            msg = f"level {level} is not in 0..{n_levels - 1}"
            raise ValueError(msg)
        table[_LABEL_NAMES[y]][str(level)] += 1
    return table
