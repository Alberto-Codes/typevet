"""Signature-match metrics over typed answers and gold pair kinds (#319).

Plain Python, no third-party statistics package. Every function takes one
value per pair in pair order. ROC-AUC, ECE and the reliability table come
from :mod:`typevet_evals.face_match.metrics`; this module adds the measures
that need the three pair kinds or the four ``Choice`` labels.

A verdict of ``different_writer`` or ``skilled_forgery_suspected`` says
"different writer". ``cannot_tell`` says neither.

Attributes:
    CANNOT_TELL (str): ``Choice`` label that never counts as a right verdict.
    SAME_WRITER_LABEL (str): ``Choice`` label that accepts the pair.
    ACCEPT_THRESHOLD (float): ``Noul`` value at or above which the pair is
        accepted as one writer.

Examples:
    ```python
    from typevet_evals.signature_match.metrics import noul_accept_rate

    assert noul_accept_rate([0.5, 0.1]) == 0.5
    ```

See Also:
    - [typevet_evals.signature_match.runner][]: collects the answers and the
      receipt
    - [typevet_evals.face_match.metrics][]: the shared ranking and
      calibration functions
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from typevet_evals.datasets.cedar import PairKind

CANNOT_TELL: Final[str] = "cannot_tell"
SAME_WRITER_LABEL: Final[str] = "same_writer"
ACCEPT_THRESHOLD: Final[float] = 0.5

_SAYS_SAME: Final[dict[str, bool | None]] = {
    SAME_WRITER_LABEL: True,
    "different_writer": False,
    "skilled_forgery_suspected": False,
    CANNOT_TELL: None,
}
_KIND_LABEL: Final[dict[PairKind, str]] = {
    PairKind.GENUINE_GENUINE: SAME_WRITER_LABEL,
    PairKind.GENUINE_SKILLED: "skilled_forgery_suspected",
    PairKind.GENUINE_RANDOM: "different_writer",
}


@dataclass(frozen=True, slots=True)
class FalseAccept:
    """False-accept rates on skilled forgeries, by two definitions.

    Attributes:
        pairs (int): Skilled-forgery pairs.
        noul_rate (float | None): Share with ``Noul`` at or above the
            threshold; ``None`` when there are no skilled pairs.
        verdict_rate (float | None): Share with the ``same_writer`` verdict;
            ``None`` when there are no skilled pairs.

    Examples:
        ```python
        FalseAccept(pairs=2, noul_rate=0.5, verdict_rate=0.0)
        ```
    """

    pairs: int
    noul_rate: float | None
    verdict_rate: float | None


@dataclass(frozen=True, slots=True)
class Agreement:
    """Agreement of the ``Noul`` side with the ``Choice`` verdict.

    Attributes:
        pairs (int): All pairs.
        decided (int): Pairs whose verdict is not ``cannot_tell``.
        agree (int): Decided pairs where both answers say the same side.
        rate (float | None): ``agree / decided``; ``None`` when nothing is
            decided.
        cannot_tell (int): Pairs with the ``cannot_tell`` verdict.
        cannot_tell_rate (float): ``cannot_tell / pairs``.

    Examples:
        ```python
        Agreement(4, 3, 3, 1.0, 1, 0.25)
        ```
    """

    pairs: int
    decided: int
    agree: int
    rate: float | None
    cannot_tell: int
    cannot_tell_rate: float


def _check_lengths(*columns: Sequence[object]) -> None:
    sizes = {len(column) for column in columns}
    if len(sizes) > 1:
        shown = " != ".join(str(len(column)) for column in columns)
        msg = f"values must have equal length: {shown}"
        raise ValueError(msg)


def _check_not_empty(values: Sequence[object]) -> None:
    if not values:
        msg = "cannot compute a rate over an empty sequence"
        raise ValueError(msg)


def verdict_says_same_writer(verdict: str) -> bool | None:
    """Map a ``Choice`` verdict to a same-writer side.

    Args:
        verdict: ``Choice`` label.

    Returns:
        ``True`` for ``same_writer``, ``False`` for ``different_writer`` and
        ``skilled_forgery_suspected``, ``None`` for ``cannot_tell``.

    Raises:
        ValueError: When the label is unknown.
    """
    if verdict not in _SAYS_SAME:
        msg = f"unknown verdict label: {verdict}"
        raise ValueError(msg)
    return _SAYS_SAME[verdict]


def verdict_accuracy(verdicts: Sequence[str], gold: Sequence[bool]) -> float:
    """Return the share of verdicts on the right same-writer side.

    ``cannot_tell`` is always wrong.

    Args:
        verdicts: ``Choice`` labels, one per pair.
        gold: Gold same-writer labels.

    Returns:
        Accuracy in [0, 1].

    Raises:
        ValueError: When lengths differ, the input is empty or a label is
            unknown.
    """
    _check_lengths(verdicts, gold)
    _check_not_empty(verdicts)
    sides = [verdict_says_same_writer(v) for v in verdicts]
    return sum(1 for s, y in zip(sides, gold, strict=True) if s is y) / len(sides)


def kind_accuracy(verdicts: Sequence[str], kinds: Sequence[PairKind]) -> float:
    """Return the share of verdicts that name the pair kind.

    The right label is ``same_writer`` for a genuine-genuine pair,
    ``skilled_forgery_suspected`` for a skilled forgery and
    ``different_writer`` for a random forgery.

    Args:
        verdicts: ``Choice`` labels, one per pair.
        kinds: Gold pair kinds.

    Returns:
        Accuracy in [0, 1].

    Raises:
        ValueError: When lengths differ, the input is empty or a label is
            unknown.
    """
    _check_lengths(verdicts, kinds)
    _check_not_empty(verdicts)
    for verdict in verdicts:
        verdict_says_same_writer(verdict)
    right = sum(1 for v, k in zip(verdicts, kinds, strict=True) if v == _KIND_LABEL[k])
    return right / len(verdicts)


def noul_accept_rate(
    confidences: Sequence[float], threshold: float = ACCEPT_THRESHOLD
) -> float:
    """Return the share of ``Noul`` values at or above ``threshold``.

    Args:
        confidences: Same-writer confidence, one per pair.
        threshold: Accept threshold.

    Returns:
        Rate in [0, 1].

    Raises:
        ValueError: When the input is empty.
    """
    _check_not_empty(confidences)
    return sum(1 for p in confidences if p >= threshold) / len(confidences)


def verdict_accept_rate(verdicts: Sequence[str]) -> float:
    """Return the share of ``same_writer`` verdicts.

    Args:
        verdicts: ``Choice`` labels, one per pair.

    Returns:
        Rate in [0, 1].

    Raises:
        ValueError: When the input is empty.
    """
    _check_not_empty(verdicts)
    return sum(1 for v in verdicts if v == SAME_WRITER_LABEL) / len(verdicts)


def skilled_false_accept(
    confidences: Sequence[float],
    verdicts: Sequence[str],
    kinds: Sequence[PairKind],
) -> FalseAccept:
    """Return the false-accept rates on the skilled-forgery pairs only.

    One rate uses the ``Noul`` at or above :data:`ACCEPT_THRESHOLD`. The
    other uses the ``same_writer`` verdict. The two are kept apart.

    Args:
        confidences: Same-writer confidence, one per pair.
        verdicts: ``Choice`` labels, one per pair.
        kinds: Gold pair kinds.

    Returns:
        The skilled pair count and both rates.

    Raises:
        ValueError: When lengths differ.
    """
    _check_lengths(confidences, verdicts, kinds)
    rows = [
        (p, v)
        for p, v, k in zip(confidences, verdicts, kinds, strict=True)
        if k is PairKind.GENUINE_SKILLED
    ]
    if not rows:
        return FalseAccept(0, None, None)
    return FalseAccept(
        pairs=len(rows),
        noul_rate=noul_accept_rate([p for p, _ in rows]),
        verdict_rate=verdict_accept_rate([v for _, v in rows]),
    )


def noul_choice_agreement(
    confidences: Sequence[float], verdicts: Sequence[str]
) -> Agreement:
    """Compare the ``Noul`` side with the verdict side for each pair.

    The ``Noul`` side is "same writer" at or above :data:`ACCEPT_THRESHOLD`.
    ``cannot_tell`` pairs are counted apart and left out of the rate.

    Args:
        confidences: Same-writer confidence, one per pair.
        verdicts: ``Choice`` labels, one per pair.

    Returns:
        Decided, agreeing and ``cannot_tell`` counts with their rates.

    Raises:
        ValueError: When lengths differ, the input is empty or a label is
            unknown.
    """
    _check_lengths(confidences, verdicts)
    _check_not_empty(verdicts)
    decided = 0
    agree = 0
    for p, v in zip(confidences, verdicts, strict=True):
        side = verdict_says_same_writer(v)
        if side is None:
            continue
        decided += 1
        agree += int((p >= ACCEPT_THRESHOLD) is side)
    cannot_tell = len(verdicts) - decided
    return Agreement(
        pairs=len(verdicts),
        decided=decided,
        agree=agree,
        rate=agree / decided if decided else None,
        cannot_tell=cannot_tell,
        cannot_tell_rate=cannot_tell / len(verdicts),
    )
