"""Answer types for System One-shaped judgment results.

Each answer carries ``off_option_flag``, ``False`` unless a caller threshold
on the off-option mass was exceeded (#353).

The domain checks noul, confidence and probabilities against ``[0, 1]`` on
construction. Logprob scoring that fills these fields lives in a future adapter
([issue #26](https://github.com/Alberto-Codes/typevet/issues/26)).

Examples:
    ```python
    from typevet.domain.judgment_answers import NoulAnswer, ChoiceAnswer, ScoreAnswer

    noul = NoulAnswer(noul=0.75)
    choice = ChoiceAnswer(
        choice="yes",
        confidence=0.8,
        probabilities={"yes": 0.8, "no": 0.2},
    )
    score = ScoreAnswer(
        score=2.0,
        confidence=0.9,
        legend={0: "poor", 1: "fair", 2: "good"},
        probabilities={0: 0.1, 1: 0.2, 2: 0.7},
    )
    ```

See Also:
    - [typevet.domain.judgment_questions][]: Question types
    - [typevet.domain.judgment_response][]: Response container
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

PROBABILITY_SUM_TOLERANCE = 0.005
"""Allowed deviation of a probability sum from 1.0, per probability."""


def _validate_finite_number(value: float, name: str) -> None:
    """Require a finite integer or float, excluding booleans.

    Raises:
        TypeError: When ``value`` is a bool or not an int or float.
        ValueError: When ``value`` is a non-finite float.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be float, got {type(value).__name__}")
    if isinstance(value, float) and not isfinite(value):
        raise ValueError(f"{name} must be finite")


def _validate_flag(value: object) -> None:
    """Require a real bool for ``off_option_flag``.

    Raises:
        TypeError: When ``value`` is not a bool.
    """
    if not isinstance(value, bool):
        raise TypeError(f"off_option_flag must be bool, got {type(value).__name__}")


@dataclass(frozen=True, slots=True)
class NoulAnswer:
    """A yes/no answer with probability of true.

    Attributes:
        noul (float): Probability of a yes answer, from 0 to 1.
        off_option_flag (bool): ``True`` when the off-option mass is above
            the caller threshold; ``False`` by default.

    Examples:
        ```python
        answer = NoulAnswer(noul=0.75)
        assert 0.0 <= answer.noul <= 1.0
        ```
    """

    noul: float
    off_option_flag: bool = False

    def __post_init__(self) -> None:
        """Reject non-finite or out-of-range ``noul`` values.

        Raises:
            TypeError: When ``noul`` is not a numeric type, or the flag is
                not a bool.
            ValueError: When ``noul`` is not finite or not in ``[0.0, 1.0]``.
        """
        _validate_flag(self.off_option_flag)
        _validate_finite_number(self.noul, "noul")
        if self.noul < 0.0 or self.noul > 1.0:
            raise ValueError(f"noul must be in [0.0, 1.0], got {self.noul}")


@dataclass(frozen=True, slots=True)
class ChoiceAnswer:
    """A selected choice with probabilities and confidence.

    Attributes:
        choice (str): The name of the selected option.
        confidence (float): Confidence in the selected choice, from 0 to 1.
        probabilities (dict[str, float]): Probability of each choice by name.
        off_option_flag (bool): ``True`` when the off-option mass is above
            the caller threshold; ``False`` by default.

    Examples:
        ```python
        answer = ChoiceAnswer(
            choice="yes",
            confidence=0.8,
            probabilities={"yes": 0.8, "no": 0.2},
        )
        assert answer.choice in answer.probabilities
        ```
    """

    choice: str
    confidence: float
    probabilities: dict[str, float]
    off_option_flag: bool = False

    def __post_init__(self) -> None:
        """Reject invalid confidence, probabilities, or a missing ``choice`` key.

        Raises:
            TypeError: When numeric fields are not numeric types, or the flag
                is not a bool.
            ValueError: When values are out of range or probabilities do not sum to one.
        """
        _validate_flag(self.off_option_flag)
        _validate_finite_number(self.confidence, "confidence")
        if self.confidence < 0.0 or self.confidence > 1.0:
            raise ValueError(f"confidence must be in [0.0, 1.0], got {self.confidence}")

        for key, value in self.probabilities.items():
            _validate_finite_number(value, "probability values")
            if value < 0.0 or value > 1.0:
                raise ValueError(
                    f"probability values must be in [0.0, 1.0], got {value} for key '{key}'"
                )

        if self.choice not in self.probabilities:
            raise ValueError(f"choice '{self.choice}' not in probabilities keys")

        total = sum(self.probabilities.values())
        if abs(total - 1.0) > PROBABILITY_SUM_TOLERANCE * len(self.probabilities):
            raise ValueError(f"probabilities must sum to 1.0, got {total:.6f}")


@dataclass(frozen=True, slots=True)
class ScoreAnswer:
    """A scored response with rubric and probabilities.

    Attributes:
        score (float): Expected score within the legend range.
        confidence (float): Confidence in the score, from 0 to 1.
        legend (dict[int, str]): Rubric descriptions keyed by integer level.
        probabilities (dict[int, float]): Probability of each level.
        off_option_flag (bool): ``True`` when the off-option mass is above
            the caller threshold; ``False`` by default.

    Examples:
        ```python
        answer = ScoreAnswer(
            score=2.0,
            confidence=0.9,
            legend={0: "poor", 1: "fair", 2: "good"},
            probabilities={0: 0.1, 1: 0.2, 2: 0.7},
        )
        assert set(answer.legend) == set(answer.probabilities)
        ```
    """

    score: float
    confidence: float
    legend: dict[int, str]
    probabilities: dict[int, float]
    off_option_flag: bool = False

    def __post_init__(self) -> None:
        """Reject mismatched legend keys, bad probabilities, or out-of-range score.

        Raises:
            TypeError: When numeric fields or dict keys have wrong types, or
                the flag is not a bool.
            ValueError: When values are out of range or probabilities do not sum to one.
        """
        _validate_flag(self.off_option_flag)
        _validate_finite_number(self.confidence, "confidence")
        if self.confidence < 0.0 or self.confidence > 1.0:
            raise ValueError(f"confidence must be in [0.0, 1.0], got {self.confidence}")

        if set(self.legend.keys()) != set(self.probabilities.keys()):
            raise ValueError("legend keys must match probabilities keys")

        for key, value in self.probabilities.items():
            self._validate_numeric_key("probability", key)
            _validate_finite_number(value, "probability values")
            if value < 0.0 or value > 1.0:
                raise ValueError(
                    f"probability values must be in [0.0, 1.0], got {value} for key {key}"
                )

        total = sum(self.probabilities.values())
        if abs(total - 1.0) > PROBABILITY_SUM_TOLERANCE * len(self.probabilities):
            raise ValueError(f"probabilities must sum to 1.0, got {total:.6f}")

        _validate_finite_number(self.score, "score")
        min_score = min(self.legend.keys())
        max_score = max(self.legend.keys())
        if self.score < min_score or self.score > max_score:
            raise ValueError(
                f"score must be in [{min_score}, {max_score}] (legend range), got {self.score}"
            )

    @staticmethod
    def _validate_numeric_key(name: str, key: object) -> None:
        if not isinstance(key, int) or isinstance(key, bool):
            raise TypeError(f"{name} key must be int, got {type(key).__name__}")


Answer = NoulAnswer | ChoiceAnswer | ScoreAnswer
"""Union type for all judgment answers."""
