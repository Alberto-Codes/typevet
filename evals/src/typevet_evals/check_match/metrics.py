"""Check-versus-register metric functions over typed answers (#316).

Plain Python, no third-party statistics package. ROC-AUC, ECE and the
reliability table come from :mod:`typevet_evals.face_match.metrics`. This
module adds the check-specific rules: right verdicts against the accepted
set of amendment A1, the false-clear rate, Noul-Choice agreement and the
``Score`` summary by variant.

Noul-Choice agreement rule. A ``Noul`` says "mismatch" when its probability
is below :data:`NOUL_THRESHOLD`. A ``payee_mismatch`` verdict agrees when the
payee ``Noul`` says mismatch. An ``amount_mismatch`` verdict agrees when the
amounts ``Noul`` says mismatch. A ``consistent``, ``date_mismatch`` or
``unsigned`` verdict agrees when neither ``Noul`` says mismatch, because
each of these verdicts implies that payee and amounts agree with the
register. A ``cannot_tell`` verdict is not counted.

Attributes:
    NOUL_THRESHOLD (float): A ``Noul`` below this value says "mismatch".
    CLEAN_VARIANT (str): Variant name of the clean check.
    LOW_VARIANT (str): Variant name of the low-legibility check.

Examples:
    ```python
    from typevet_evals.check_match.metrics import noul_choice_agreement

    assert noul_choice_agreement("payee_mismatch", 0.1, 0.9) is True
    assert noul_choice_agreement("cannot_tell", 0.1, 0.9) is None
    ```

See Also:
    - [typevet_evals.check_match.runner][]: collects answers and the receipt
    - [typevet_evals.face_match.metrics][]: the shared rank and bin metrics
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final

from typevet_evals.check_match.cases import CheckVariant, ExpectedLabels
from typevet_evals.check_match.request import VERDICT_LABELS

NOUL_THRESHOLD: Final[float] = 0.5
CLEAN_VARIANT: Final[str] = CheckVariant.CLEAN.value
LOW_VARIANT: Final[str] = CheckVariant.LOW_LEGIBILITY.value

_CONSISTENT: Final[str] = "consistent"
_CANNOT_TELL: Final[str] = "cannot_tell"
_PAYEE_MISMATCH: Final[str] = "payee_mismatch"
_AMOUNT_MISMATCH: Final[str] = "amount_mismatch"
_KNOWN: Final[frozenset[str]] = frozenset(VERDICT_LABELS)


def _check_label(verdict: str) -> None:
    if verdict not in _KNOWN:
        msg = f"unknown verdict label: {verdict}"
        raise ValueError(msg)


def _check_lengths(left: Sequence[object], right: Sequence[object]) -> None:
    if len(left) != len(right):
        msg = f"inputs must have equal length: {len(left)} != {len(right)}"
        raise ValueError(msg)


def verdict_correct(verdict: str, expected: ExpectedLabels) -> bool:
    """Return whether ``verdict`` is in the accepted set for its variant.

    Low legibility accepts ``consistent`` and ``cannot_tell``.

    Args:
        verdict: ``Choice`` label.
        expected: Expected labels of the case.

    Returns:
        ``True`` when the verdict counts as right.

    Raises:
        ValueError: When the label is not a check-match verdict.
    """
    _check_label(verdict)
    return verdict in expected.accepted_verdicts


def class_key(expected: ExpectedLabels) -> str:
    """Return the accuracy class name of one expected label set.

    Args:
        expected: Expected labels of the case.

    Returns:
        The accepted verdicts, sorted and joined with ``|``.
    """
    return "|".join(sorted(expected.accepted_verdicts))


def accuracy_by_group(keys: Sequence[str], correct: Sequence[bool]) -> dict[str, float]:
    """Return the share of right answers in each group.

    Args:
        keys: Group name, one per case.
        correct: Whether each case is right.

    Returns:
        Accuracy per group, keyed in sorted order.

    Raises:
        ValueError: When lengths differ.
    """
    _check_lengths(keys, correct)
    totals: dict[str, list[int]] = {}
    for key, right in zip(keys, correct, strict=True):
        tally = totals.setdefault(key, [0, 0])
        tally[0] += int(right)
        tally[1] += 1
    return {key: totals[key][0] / totals[key][1] for key in sorted(totals)}


def counts_for_false_clear(expected: ExpectedLabels) -> bool:
    """Return whether a case enters the false-clear rate.

    A case counts when ``consistent`` is not an accepted verdict and the
    variant counts for false clear. So clean and low legibility never count.

    Args:
        expected: Expected labels of the case.

    Returns:
        ``True`` for the mismatch and unsigned variants.
    """
    return expected.counts_for_false_clear and _CONSISTENT not in (
        expected.accepted_verdicts
    )


def false_clear_rate(
    verdicts: Sequence[str], expected: Sequence[ExpectedLabels]
) -> float | None:
    """Return the share of mismatch cases answered ``consistent``.

    :func:`counts_for_false_clear` selects the cases.

    Args:
        verdicts: ``Choice`` labels, one per case.
        expected: Expected labels, one per case.

    Returns:
        Rate in [0, 1], or ``None`` when no case counts.

    Raises:
        ValueError: When lengths differ.
    """
    _check_lengths(verdicts, expected)
    counted = [
        v for v, e in zip(verdicts, expected, strict=True) if counts_for_false_clear(e)
    ]
    if not counted:
        return None
    return sum(1 for v in counted if v == _CONSISTENT) / len(counted)


def noul_choice_agreement(
    verdict: str, payee_confidence: float, amounts_confidence: float
) -> bool | None:
    """Return whether the two ``Noul`` answers support the verdict.

    The module docstring states the rule.

    Args:
        verdict: ``Choice`` label.
        payee_confidence: Payee ``Noul`` probability.
        amounts_confidence: Amounts ``Noul`` probability.

    Returns:
        ``True`` or ``False``; ``None`` for ``cannot_tell``.

    Raises:
        ValueError: When the label is not a check-match verdict.
    """
    _check_label(verdict)
    payee_mismatch = payee_confidence < NOUL_THRESHOLD
    amounts_mismatch = amounts_confidence < NOUL_THRESHOLD
    if verdict == _CANNOT_TELL:
        return None
    if verdict == _PAYEE_MISMATCH:
        return payee_mismatch
    if verdict == _AMOUNT_MISMATCH:
        return amounts_mismatch
    return not (payee_mismatch or amounts_mismatch)


def agreement_rate(flags: Sequence[bool | None]) -> float | None:
    """Return the share of ``True`` among the counted cases.

    Args:
        flags: Agreement per case; ``None`` is not counted.

    Returns:
        Rate in [0, 1], or ``None`` when no case counts.
    """
    counted = [f for f in flags if f is not None]
    if not counted:
        return None
    return sum(1 for f in counted if f) / len(counted)


def score_summary(
    scores: Sequence[float], levels: Sequence[int], *, n_levels: int
) -> dict[str, object]:
    """Summarize ``Score`` answers for one group of cases.

    Args:
        scores: Expected ``Score`` value, one per case.
        levels: Most probable ``Score`` level, one per case.
        n_levels: Number of levels, from 0 to ``n_levels - 1``.

    Returns:
        ``count``, ``mean`` (``None`` when empty) and ``distribution``, the
        level counts keyed by the level as text.

    Raises:
        ValueError: When lengths differ or a level is out of range.
    """
    _check_lengths(scores, levels)
    distribution = {str(level): 0 for level in range(n_levels)}
    for level in levels:
        if not 0 <= level < n_levels:
            msg = f"level {level} is not in 0..{n_levels - 1}"
            raise ValueError(msg)
        distribution[str(level)] += 1
    return {
        "count": len(scores),
        "mean": sum(scores) / len(scores) if scores else None,
        "distribution": distribution,
    }


def legibility_gap(by_variant: Mapping[str, Mapping[str, object]]) -> float | None:
    """Return the clean mean ``Score`` minus the low-legibility mean.

    Args:
        by_variant: ``Score`` summaries keyed by variant name.

    Returns:
        The gap, or ``None`` when either mean is missing.
    """
    clean = by_variant.get(CLEAN_VARIANT, {}).get("mean")
    low = by_variant.get(LOW_VARIANT, {}).get("mean")
    if not isinstance(clean, float | int) or not isinstance(low, float | int):
        return None
    return clean - low
