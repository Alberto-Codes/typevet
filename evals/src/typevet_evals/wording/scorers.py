"""The gepa-adk reward of a ``Choice`` seed: a scaled multi-class Brier (#369).

The transport answers a ``Choice`` seed with ``{"probabilities": {label: p}}``.
``ChoiceScorer`` scores that body against the gold label name as
``1 - choice_brier``, where ``choice_brier`` is
``1/2 * sum_k (p_k - 1[k = gold])^2``. The half scales the multi-class Brier
to 0..1, so a ``Choice`` reward and a ``Noul`` reward share one range.

Every refusal names a label or the body key, never a wording text.

Examples:
    ```python
    scorer = ChoiceScorer(("yes", "no", "maybe"))
    body = '{"probabilities": {"yes": 0.7, "no": 0.2, "maybe": 0.1}}'
    score, meta = scorer.score("state", body, "yes")
    # score == 0.93, meta["brier"] == 0.07
    ```

See Also:
    - [typevet_evals.wording.runner][]: ``BrierScorer``, the ``Noul`` reward
    - [typevet_evals.wording.transport][]: the transport body of each seed type
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any


def check_distribution(
    probabilities: Mapping[str, Any], labels: Sequence[str]
) -> dict[str, float]:
    """Return a label distribution after checking it holds exactly ``labels``.

    Args:
        probabilities: Label name to probability.
        labels: The seed's labels.

    Returns:
        The distribution as floats, in ``labels`` order.

    Raises:
        ValueError: If a label is missing or extra, or a probability is not
            a number between 0 and 1. The message names the label only.
    """
    for label in labels:
        if label not in probabilities:
            msg = f"the distribution lacks the label {label!r}"
            raise ValueError(msg)
    for label in probabilities:
        if label not in labels:
            msg = f"the distribution holds the unknown label {label!r}"
            raise ValueError(msg)
    checked: dict[str, float] = {}
    for label in labels:
        value = probabilities[label]
        number = isinstance(value, int | float) and not isinstance(value, bool)
        if not number or math.isnan(value) or not 0.0 <= value <= 1.0:
            msg = f"the probability of label {label!r} is not between 0 and 1"
            raise ValueError(msg)
        checked[label] = float(value)
    return checked


def choice_brier(probabilities: Mapping[str, float], gold: str) -> float:
    """Return the scaled multi-class Brier score of one distribution.

    Args:
        probabilities: Label name to probability, every label present.
        gold: The gold label name.

    Returns:
        ``1/2 * sum_k (p_k - 1[k = gold])^2``; 0 is a perfect answer, 1 the
        worst.
    """
    return 0.5 * sum(
        (p - (1.0 if label == gold else 0.0)) ** 2 for label, p in probabilities.items()
    )


def read_distribution(output: str, labels: Sequence[str]) -> dict[str, float]:
    """Read the checked distribution from ``{"probabilities": {...}}``.

    Args:
        output: The agent's final text.
        labels: The seed's labels.

    Returns:
        The distribution, in ``labels`` order.

    Raises:
        ValueError: If the text is not that body, or the distribution does
            not pass ``check_distribution``.
        TypeError: If ``probabilities`` is not a mapping.
    """
    try:
        probabilities = json.loads(output)["probabilities"]
    except (json.JSONDecodeError, TypeError, KeyError) as exc:
        raise ValueError("no probabilities in the transport body") from exc
    if not isinstance(probabilities, Mapping):
        raise TypeError("the probabilities are not a mapping")
    return check_distribution(probabilities, labels)


class ChoiceScorer:
    """A gepa-adk ``Scorer`` for a ``Choice`` seed: one minus ``choice_brier``.

    ``expected`` is the row's gold label name. A body or label it cannot read
    raises ``ValueError``; gepa-adk then counts the row as a failed
    evaluation that scores 0.

    Attributes:
        labels (tuple[str, ...]): The seed's labels, in seed order.

    Examples:
        ```python
        scorer = ChoiceScorer(("yes", "no", "maybe"))
        ```
    """

    def __init__(self, labels: Sequence[str]) -> None:
        """Hold the seed's labels.

        Args:
            labels: The seed's labels, in seed order.
        """
        self.labels = tuple(labels)

    def score(
        self, input_text: str, output: str, expected: str | None = None
    ) -> tuple[float, dict[str, Any]]:
        """Score one transport body against its gold label.

        Args:
            input_text: The state key; not used.
            output: The transport body ``{"probabilities": {label: p}}``.
            expected: The gold label name.

        Returns:
            ``1 - brier`` and the Brier score, distribution and gold label.

        Raises:
            ValueError: If the gold label is not one of ``labels``.
        """
        if expected not in self.labels:
            msg = f"gold label {expected!r} is not a label of the seed"
            raise ValueError(msg)
        probabilities = read_distribution(output, self.labels)
        brier = choice_brier(probabilities, expected)
        meta = {"brier": brier, "probabilities": probabilities, "gold": expected}
        return 1.0 - brier, meta

    async def async_score(
        self, input_text: str, output: str, expected: str | None = None
    ) -> tuple[float, dict[str, Any]]:
        """Score one transport body against its gold label.

        Args:
            input_text: The state key; not used.
            output: The transport body ``{"probabilities": {label: p}}``.
            expected: The gold label name.

        Returns:
            The same result as ``score``.
        """
        return self.score(input_text, output, expected)


def seed_labels(seed: object) -> tuple[str, ...]:
    """Return the labels of a ``Choice`` seed, in seed order.

    Args:
        seed: The seed question.

    Returns:
        Each ``Choice`` label as text; an empty tuple for another seed type.
    """
    if type(seed).__name__ != "Choice":
        return ()
    criteria: Mapping[Any, Any] = getattr(seed, "criteria", {})
    return tuple(str(label) for label in criteria)
