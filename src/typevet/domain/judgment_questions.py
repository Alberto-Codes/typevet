"""Question types for System One-shaped judgment calls.

``question_types`` maps each question id to its wire type name for diagnostics.

Examples:
    ```python
    from typevet.domain.judgment_questions import Noul, Choice, Score

    noul = Noul(
        instructions="Is this about billing?",
        criteria={"true": "Billing topic", "false": "Not billing"},
    )
    choice = Choice(
        criteria={"a": "Option A", "b": "Option B"},
        instructions="Choose one:",
    )
    score = Score(
        criteria=["Poor", "Fair", "Good"],
        instructions="Rate the response:",
    )
    ```

See Also:
    - [typevet.domain.judgment_answers][]: Answer types
    - [typevet.domain.judgment_response][]: Response container
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


class Noul:
    """A yes/no question with optional descriptions for either outcome.

    Construction is keyword-only.

    Attributes:
        instructions (str | dict | Sequence | None): Question or statement to evaluate.
        criteria (dict | None): Optional ``true`` and ``false`` outcome descriptions.

    Examples:
        ```python
        question = Noul(
            instructions="Is this valid?",
            criteria={"true": "Valid", "false": "Invalid"},
        )
        assert question.instructions is not None
        ```
    """

    __slots__ = ("criteria", "instructions")

    def __init__(
        self,
        *,
        instructions: str | dict[str, Any] | Sequence[Any] | None = None,
        criteria: dict[str, Any] | None = None,
    ) -> None:
        """Initialize a Noul question.

        Args:
            instructions: The yes/no question or statement to evaluate.
            criteria: Optional descriptions of the yes and no outcomes.
        """
        self.instructions = instructions
        self.criteria = criteria

    def __repr__(self) -> str:
        """Return a debug representation."""
        return f"Noul(instructions={self.instructions!r}, criteria={self.criteria!r})"


class Choice:
    """A question that selects between named alternatives.

    Construction is keyword-only.

    Attributes:
        criteria (Mapping[str, str | dict | Sequence | None]): Labels mapped to
            descriptions or ``None`` when undescribed.
        instructions (str | dict | Sequence | None): The question to ask.

    Examples:
        ```python
        question = Choice(
            criteria={"a": "Option A", "b": "Option B"},
            instructions="Choose one:",
        )
        assert len(question.criteria) == 2
        ```
    """

    __slots__ = ("criteria", "instructions")

    def __init__(
        self,
        *,
        criteria: Mapping[str, str | dict[str, Any] | Sequence[Any] | None],
        instructions: str | dict[str, Any] | Sequence[Any] | None = None,
    ) -> None:
        """Initialize a Choice question.

        Args:
            criteria: Labels mapped to descriptions.
            instructions: The question to ask.
        """
        self.criteria = dict(criteria)
        self.instructions = instructions

    def __repr__(self) -> str:
        """Return a debug representation."""
        return f"Choice(criteria={self.criteria!r}, instructions={self.instructions!r})"


class Score:
    """A question that assigns a score using an ordered rubric.

    Construction is keyword-only.

    Attributes:
        criteria (Sequence[str | dict | Sequence]): Ordered rubric descriptions.
        instructions (str | dict | Sequence | None): What the model should rate.

    Examples:
        ```python
        question = Score(
            criteria=["Poor", "Fair", "Good"],
            instructions="Rate the response:",
        )
        assert len(question.criteria) >= 2
        ```
    """

    __slots__ = ("criteria", "instructions")

    def __init__(
        self,
        *,
        criteria: Sequence[str | dict[str, Any] | Sequence[Any]],
        instructions: str | dict[str, Any] | Sequence[Any] | None = None,
    ) -> None:
        """Initialize a Score question.

        Args:
            criteria: Ordered descriptions, one per score from zero.
            instructions: What the model should rate.
        """
        self.criteria = list(criteria)
        self.instructions = instructions

    def __repr__(self) -> str:
        """Return a debug representation."""
        return f"Score(criteria={self.criteria!r}, instructions={self.instructions!r})"


Question = Noul | Choice | Score
"""A typed judgment question."""

_TYPE_NAMES: tuple[tuple[type, str], ...] = (
    (Noul, "noul"),
    (Choice, "choice"),
    (Score, "score"),
)


def question_types(questions: Mapping[str, Any]) -> dict[str, str | None]:
    """Map each question id to its wire type name.

    Typed questions map to ``noul``, ``choice`` or ``score``. Raw wire dictionaries
    map to their ``type`` value when that value is a string.

    Args:
        questions: Question objects or raw wire dictionaries keyed by id.

    Returns:
        The id to type mapping; ``None`` where a raw question has no string type.
    """
    types: dict[str, str | None] = {}
    for name, value in questions.items():
        kind = next((n for cls, n in _TYPE_NAMES if isinstance(value, cls)), None)
        if kind is None and isinstance(value, dict):
            raw = value.get("type")
            kind = raw if isinstance(raw, str) else None
        types[name] = kind
    return types
