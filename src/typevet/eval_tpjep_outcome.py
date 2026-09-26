"""Map judgment answers to TPJEP attempt outcomes (#106).

Examples:
    ```python
    from typevet.domain.judgment_answers import NoulAnswer
    from typevet.eval_tpjep_outcome import outcome_from_answer

    # task from eval_tpjep_loader ...
    predicted, probs, correct = outcome_from_answer(task, NoulAnswer(noul=0.2))
    ```

See Also:
    - [typevet.eval_tpjep_runner][]: Builds attempt records from outcomes
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from typevet.domain.judgment_answers import ChoiceAnswer, NoulAnswer, ScoreAnswer
from typevet.eval_tpjep_loader import TpjepScheduledTask

_PROB_SUM_TOLERANCE: Final[float] = 0.001
_NOUL_LABEL_PAIR: Final[int] = 2


def prob_valid(probabilities: Mapping[str, float] | None) -> bool:
    """Return true when label probabilities sum to one within tolerance.

    Returns:
        ``True`` when the sum is within ``0.001`` of one.
    """
    if not probabilities:
        return False
    total = sum(probabilities.values())
    return abs(total - 1.0) <= _PROB_SUM_TOLERANCE


def _noul_probabilities(
    task: TpjepScheduledTask, answer: NoulAnswer
) -> dict[str, float]:
    if len(task.labels) == _NOUL_LABEL_PAIR and task.labels[1] in ("true", "yes"):
        yes_key, no_key = task.labels[1], task.labels[0]
        return {yes_key: answer.noul, no_key: 1.0 - answer.noul}
    return {task.labels[0]: 1.0 - answer.noul, task.labels[1]: answer.noul}


def outcome_from_answer(
    task: TpjepScheduledTask,
    answer: NoulAnswer | ChoiceAnswer | ScoreAnswer,
) -> tuple[object, Mapping[str, float] | None, bool | None]:
    """Derive predicted label, probabilities, and correctness for one answer.

    Returns:
        Tuple of predicted value, probability map, and correctness when valid.
    """
    if isinstance(answer, NoulAnswer):
        probs = _noul_probabilities(task, answer)
        predicted = max(probs, key=probs.__getitem__)
        valid = prob_valid(probs)
        correct = valid and predicted == task.expected
        return predicted, probs, correct if valid else None
    if isinstance(answer, ChoiceAnswer):
        probs = dict(answer.probabilities)
        predicted = answer.choice
        valid = prob_valid(probs)
        correct = valid and predicted == task.expected
        return predicted, probs, correct if valid else None
    probs = {str(k): float(v) for k, v in answer.probabilities.items()}
    predicted = answer.score
    valid = prob_valid(probs)
    modal = max(answer.probabilities, key=answer.probabilities.__getitem__)
    correct = valid and modal == task.expected
    return predicted, probs, correct if valid else None
