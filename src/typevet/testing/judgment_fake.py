"""Public offline ``JudgmentPort`` fake built from scripted distributions.

Examples:
    ```python
    from typevet.domain.judgment_questions import Choice, Noul
    from typevet.testing import ScriptedJudgmentFake

    fake = ScriptedJudgmentFake(
        {"billing": 0.8, "route": {"billing": 3.0, "technical": 1.0}}
    )
    response = fake.judge(
        "I was charged twice.",
        {
            "billing": Noul(),
            "route": Choice(criteria={"billing": "Money", "technical": "Bugs"}),
        },
        "fake-judgment",
    )
    assert response.choices["route"].choice == "billing"
    ```

See Also:
    - [typevet.ports.judgment][]: JudgmentPort protocol
    - [typevet.domain.judgment_answers][]: Answer types this fake returns
    - [typevet.testing.fakes][]: Generation and scoring fakes
"""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from typing import Any

from typevet.domain.decision_execute import check_off_option_threshold
from typevet.domain.errors import DecisionExecutionError, JudgmentValidationError
from typevet.domain.judgment_answers import (
    Answer,
    ChoiceAnswer,
    NoulAnswer,
    ScoreAnswer,
)
from typevet.domain.judgment_questions import Choice, Noul, Question, Score
from typevet.domain.judgment_response import JudgmentResponse, TokenUsage
from typevet.domain.media import ImageInput

Distribution = float | Mapping[str, float] | Mapping[int, float]
"""Noul P(True), Choice label weights, or Score level weights."""


def _finite(value: object) -> float | None:
    """Return ``value`` as a float when it is a finite non-bool number.

    Args:
        value: Candidate number.

    Returns:
        The float value, or ``None`` for a bool, a non-number or a
        non-finite float.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if isfinite(value) else None


def _normalise(name: str, weights: object) -> dict[Any, float]:
    """Divide each weight by the weight sum.

    Args:
        name: Question name for error messages.
        weights: Mapping of non-negative finite weights by label or level.

    Returns:
        Weights scaled to sum to one, in the input key order.

    Raises:
        JudgmentValidationError: When ``weights`` is not a mapping, a weight
            is negative or not a finite number, or the sum is not positive.
    """
    if not isinstance(weights, Mapping):
        msg = f"distribution for {name!r} must be a mapping of weights"
        raise JudgmentValidationError(msg)
    checked: dict[Any, float] = {}
    for key, weight in weights.items():
        value = _finite(weight)
        if value is None or value < 0.0:
            msg = f"weight for {key!r} in {name!r} must be finite and >= 0"
            raise JudgmentValidationError(msg)
        checked[key] = value
    total = sum(checked.values())
    if total <= 0.0:
        msg = f"weights for {name!r} must have a positive sum"
        raise JudgmentValidationError(msg)
    return {key: value / total for key, value in checked.items()}


def _noul_answer(name: str, distribution: object) -> NoulAnswer:
    """Build a Noul answer from a scripted P(True).

    Args:
        name: Question name for error messages.
        distribution: Scripted probability of a true answer.

    Returns:
        Noul answer with that probability.

    Raises:
        JudgmentValidationError: When the value is not a number in ``[0, 1]``.
    """
    value = _finite(distribution)
    if value is None or not 0.0 <= value <= 1.0:
        msg = f"P(True) for {name!r} must be a number in [0, 1]"
        raise JudgmentValidationError(msg)
    return NoulAnswer(noul=value)


def _choice_answer(name: str, question: Choice, distribution: object) -> ChoiceAnswer:
    """Build a Choice answer from label weights.

    Labels absent from the weights get probability zero. Ties go to the
    first label in ``question.criteria`` order.

    Args:
        name: Question name for error messages.
        question: Choice question that names the allowed labels.
        distribution: Label to weight mapping.

    Returns:
        Choice answer with the argmax label and normalised probabilities.

    Raises:
        JudgmentValidationError: When a label is outside ``criteria`` or the
            weights are not valid.
    """
    weights = _normalise(name, distribution)
    unknown = [label for label in weights if label not in question.criteria]
    if unknown:
        msg = f"labels {unknown!r} for {name!r} not in criteria {list(question.criteria)!r}"
        raise JudgmentValidationError(msg)
    probabilities = {label: weights.get(label, 0.0) for label in question.criteria}
    choice = max(probabilities, key=lambda label: probabilities[label])
    return ChoiceAnswer(
        choice=choice,
        confidence=probabilities[choice],
        probabilities=probabilities,
    )


def _score_answer(name: str, question: Score, distribution: object) -> ScoreAnswer:
    """Build a Score answer from level weights.

    Levels absent from the weights get probability zero. The legend maps each
    level to its rubric text. Ties go to the lowest level.

    Args:
        name: Question name for error messages.
        question: Score question whose rubric fixes the levels.
        distribution: Level to weight mapping.

    Returns:
        Score answer with the argmax level as ``score``.

    Raises:
        JudgmentValidationError: When a level is outside the rubric or the
            weights are not valid.
    """
    weights = _normalise(name, distribution)
    levels = range(len(question.criteria))
    unknown = [
        level
        for level in weights
        if isinstance(level, bool) or not isinstance(level, int) or level not in levels
    ]
    if unknown:
        msg = f"levels {unknown!r} for {name!r} not in rubric levels {list(levels)!r}"
        raise JudgmentValidationError(msg)
    probabilities = {level: weights.get(level, 0.0) for level in levels}
    legend = {level: str(question.criteria[level]) for level in levels}
    score = max(probabilities, key=lambda level: probabilities[level])
    return ScoreAnswer(
        score=float(score),
        confidence=probabilities[score],
        legend=legend,
        probabilities=probabilities,
    )


class ScriptedJudgmentFake:
    """Offline ``JudgmentPort`` that answers from scripted distributions.

    Each question name maps to one distribution. A Noul takes P(True) as a
    float. A Choice takes label weights. A Score takes level weights, with
    levels counted from zero along the rubric. Weights are normalised by
    their sum. Media is accepted and ignored, and nothing is forwarded.

    Attributes:
        _distributions (dict[str, Distribution]): Scripted distribution per
            question name.

    Examples:
        ```python
        from typevet.domain.judgment_questions import Score
        from typevet.testing import ScriptedJudgmentFake

        fake = ScriptedJudgmentFake({"quality": {0: 1.0, 2: 3.0}})
        rubric = Score(criteria=["Poor", "Fair", "Good"])
        answer = fake.judge("text", {"quality": rubric}, "m").scores["quality"]
        assert answer.score == 2.0
        ```
    """

    def __init__(self, distributions: Mapping[str, Distribution]) -> None:
        """Store the scripted distribution for each question name.

        Args:
            distributions: Question name to Noul P(True), Choice label
                weights, or Score level weights.
        """
        self._distributions: dict[str, Distribution] = dict(distributions)

    def judge(
        self,
        state: str | dict[str, Any] | list[Any],
        questions: Mapping[str, Question | Mapping[str, Any]],
        model: str,
        *,
        media: tuple[ImageInput, ...] | None = None,
        off_option_threshold: float | None = None,
    ) -> JudgmentResponse:
        """Answer each question from its scripted distribution.

        Args:
            state: Content under evaluation; ignored.
            questions: Question names to typed ``Noul``, ``Choice`` or
                ``Score`` questions.
            model: Model id copied onto the response.
            media: Images of any length; ignored.
            off_option_threshold: Checked for range; no answer is flagged.

        Returns:
            Response with one answer per question and empty token usage.

        Raises:
            JudgmentValidationError: For a blank model, an invalid threshold,
                a question name without a distribution, a question that is
                not a typed question, or an invalid distribution.
        """
        del state, media
        if not model.strip():
            raise JudgmentValidationError("model must be non-empty")
        try:
            check_off_option_threshold(off_option_threshold)
        except DecisionExecutionError as exc:
            raise JudgmentValidationError(str(exc)) from exc
        answers = {
            name: self._answer(name, question) for name, question in questions.items()
        }
        return JudgmentResponse(model=model, usage=TokenUsage(), answers=answers)

    def _answer(self, name: str, question: object) -> Answer:
        """Build the answer for one named question.

        Args:
            name: Question name.
            question: Typed question for that name.

        Returns:
            Answer of the kind that matches the question.

        Raises:
            JudgmentValidationError: When the name has no distribution or the
                question is not a ``Noul``, ``Choice`` or ``Score``.
        """
        if name not in self._distributions:
            msg = f"no scripted distribution for question {name!r}"
            raise JudgmentValidationError(msg)
        distribution = self._distributions[name]
        if isinstance(question, Noul):
            return _noul_answer(name, distribution)
        if isinstance(question, Choice):
            return _choice_answer(name, question, distribution)
        if isinstance(question, Score):
            return _score_answer(name, question, distribution)
        msg = f"unsupported wire question for {name!r}"
        raise JudgmentValidationError(msg)
