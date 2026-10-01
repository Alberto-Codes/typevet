"""Held-out metrics of a ``Choice`` seed run (#369).

A ``ChoicePair`` holds both wordings' label distributions for one held-out
row and its gold label name. ``choice_metrics`` measures one wording:

- accuracy: the share of rows whose most probable label is the gold label;
  a tie goes to the first label in seed order;
- Brier: the mean of ``choice_brier``, the multi-class Brier scaled to 0..1;
- ECE: over ``ECE_BINS`` bins, on the chosen label's probability against
  whether the chosen label is the gold label;
- kappa: Cohen's kappa of the chosen labels against the gold labels.

``choice_bootstrap`` and the pass rule reuse the ``Noul`` code paths in
``metrics``. The #133 ECE reference applies to ``Noul`` runs only.

Examples:
    ```python
    pairs = [ChoicePair("id", "yes", {"yes": 0.5, "no": 0.5}, {"yes": 0.9, "no": 0.1})]
    seed, evolved = choice_arm_metrics(pairs, ("yes", "no"))
    ```

See Also:
    - [typevet_evals.wording.metrics][]: the ``Noul`` metrics, bootstrap and rule
    - [typevet_evals.wording.scorers][]: ``choice_brier``
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from typevet_evals.face_match.metrics import expected_calibration_error
from typevet_evals.wording.metrics import (
    ECE_BINS,
    PairedBootstrap,
    bootstrap_differences,
)
from typevet_evals.wording.scorers import check_distribution, choice_brier


@dataclass(frozen=True, slots=True)
class ChoicePair:
    """Both wordings' label distributions for one held-out row.

    Attributes:
        record_id (str): The record id.
        gold (str): The gold label name.
        seed (Mapping[str, float]): The seed wording's distribution.
        evolved (Mapping[str, float]): The evolved wording's distribution.

    Examples:
        ```python
        ChoicePair(
            "pubmedqa:1", "yes", {"yes": 0.6, "no": 0.4}, {"yes": 0.9, "no": 0.1}
        )
        ```
    """

    record_id: str
    gold: str
    seed: Mapping[str, float]
    evolved: Mapping[str, float]

    def to_mapping(self) -> dict[str, object]:
        """Return the pair as a JSON mapping.

        Returns:
            The record id, the gold label and both distributions.
        """
        return {
            "record_id": self.record_id,
            "gold": self.gold,
            "seed": dict(self.seed),
            "evolved": dict(self.evolved),
        }


@dataclass(frozen=True, slots=True)
class ChoiceMetrics:
    """Accuracy, Brier score, ECE and kappa of one wording on a ``Choice`` seed.

    Attributes:
        rows (int): Rows scored.
        accuracy (float): Share of rows whose chosen label is the gold label.
        brier (float): Mean scaled multi-class Brier score.
        ece (float): ECE of the chosen label's probability.
        kappa (float | None): Cohen's kappa of the chosen labels; None when
            chance agreement is 1.

    Examples:
        ```python
        ChoiceMetrics(rows=3, accuracy=1.0, brier=0.03, ece=0.2, kappa=1.0)
        ```
    """

    rows: int
    accuracy: float
    brier: float
    ece: float
    kappa: float | None

    def to_mapping(self) -> dict[str, float | int | None]:
        """Return the metrics as a JSON mapping.

        Returns:
            One key per attribute.
        """
        return {
            "rows": self.rows,
            "accuracy": self.accuracy,
            "brier": self.brier,
            "ece": self.ece,
            "kappa": self.kappa,
        }


def chosen_label(probabilities: Mapping[str, float], labels: Sequence[str]) -> str:
    """Return the most probable label; a tie goes to the first in seed order.

    Args:
        probabilities: Label name to probability.
        labels: The seed's labels, in seed order.

    Returns:
        The chosen label.
    """
    return max(labels, key=lambda label: (probabilities[label], -labels.index(label)))


def _kappa(
    chosen: Sequence[str], golds: Sequence[str], labels: Sequence[str]
) -> float | None:
    """Return Cohen's kappa of the chosen labels against the gold labels.

    Args:
        chosen: The chosen label of each row.
        golds: The gold label of each row.
        labels: The seed's labels.

    Returns:
        ``(p_o - p_e) / (1 - p_e)``; None when ``p_e`` is 1.
    """
    rows = len(golds)
    observed = sum(c == g for c, g in zip(chosen, golds, strict=True)) / rows
    chance = sum((chosen.count(k) / rows) * (golds.count(k) / rows) for k in labels)
    if chance >= 1.0:
        return None
    return (observed - chance) / (1 - chance)


def _ece(
    dists: Sequence[Mapping[str, float]], golds: Sequence[str], labels: Sequence[str]
) -> float:
    """Return the ECE of the chosen label's probability.

    Args:
        dists: One distribution per row.
        golds: The gold label of each row.
        labels: The seed's labels.

    Returns:
        The ECE over ``ECE_BINS`` bins.
    """
    chosen = [chosen_label(d, labels) for d in dists]
    confidences = [d[c] for d, c in zip(dists, chosen, strict=True)]
    right = [c == g for c, g in zip(chosen, golds, strict=True)]
    return expected_calibration_error(confidences, right, n_bins=ECE_BINS)


def _brier(dists: Sequence[Mapping[str, float]], golds: Sequence[str]) -> float:
    """Return the mean scaled multi-class Brier score.

    Args:
        dists: One distribution per row.
        golds: The gold label of each row.

    Returns:
        The mean of ``choice_brier`` over the rows.
    """
    return sum(choice_brier(d, g) for d, g in zip(dists, golds, strict=True)) / len(
        golds
    )


def choice_metrics(
    dists: Sequence[Mapping[str, float]], golds: Sequence[str], labels: Sequence[str]
) -> ChoiceMetrics:
    """Measure one wording over its rows.

    Args:
        dists: One distribution per row.
        golds: The gold label of each row.
        labels: The seed's labels, in seed order.

    Returns:
        The row count, accuracy, Brier score, ECE and kappa.

    Raises:
        ValueError: When the lengths differ, there are no rows, a gold label
            is not a seed label, or a distribution does not hold exactly the
            seed labels.
    """
    if len(dists) != len(golds) or not golds:
        raise ValueError("cannot measure rows of unequal length or no rows")
    for gold in golds:
        if gold not in labels:
            raise ValueError(f"gold label {gold!r} is not a label of the seed")
    dists = [check_distribution(d, labels) for d in dists]
    chosen = [chosen_label(d, labels) for d in dists]
    return ChoiceMetrics(
        rows=len(golds),
        accuracy=sum(c == g for c, g in zip(chosen, golds, strict=True)) / len(golds),
        brier=_brier(dists, golds),
        ece=_ece(dists, golds, labels),
        kappa=_kappa(chosen, golds, labels),
    )


def choice_arm_metrics(
    pairs: Sequence[ChoicePair], labels: Sequence[str]
) -> tuple[ChoiceMetrics, ChoiceMetrics]:
    """Return the seed and evolved metrics of the pairs.

    Args:
        pairs: The scored pairs.
        labels: The seed's labels.

    Returns:
        The seed wording's metrics and the evolved wording's metrics.
    """
    golds = [p.gold for p in pairs]
    return (
        choice_metrics([p.seed for p in pairs], golds, labels),
        choice_metrics([p.evolved for p in pairs], golds, labels),
    )


def choice_bootstrap(
    pairs: Sequence[ChoicePair], labels: Sequence[str]
) -> PairedBootstrap:
    """Bound the evolved minus seed ECE and Brier differences of the pairs.

    The resamples are those of ``paired_bootstrap`` for the same row count.

    Args:
        pairs: The scored pairs; ``choice_arm_metrics`` checks them first.
        labels: The seed's labels.

    Returns:
        The paired bootstrap intervals.
    """
    arms = ([p.seed for p in pairs], [p.evolved for p in pairs])
    golds = [p.gold for p in pairs]

    def ece(rows: Sequence[int], arm: int) -> float:
        """Return one arm's ECE over the rows at ``rows``.

        Args:
            rows: Row positions.
            arm: 0 for the seed wording, 1 for the evolved wording.

        Returns:
            The ECE.
        """
        return _ece([arms[arm][i] for i in rows], [golds[i] for i in rows], labels)

    def brier(rows: Sequence[int], arm: int) -> float:
        """Return one arm's Brier score over the rows at ``rows``.

        Args:
            rows: Row positions.
            arm: 0 for the seed wording, 1 for the evolved wording.

        Returns:
            The Brier score.
        """
        return _brier([arms[arm][i] for i in rows], [golds[i] for i in rows])

    return bootstrap_differences(len(pairs), ece, brier)
