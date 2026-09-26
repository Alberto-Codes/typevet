"""Response container for judgment port calls.

Examples:
    ```python
    from typevet.domain.judgment_answers import NoulAnswer
    from typevet.domain.judgment_response import JudgmentResponse, TokenUsage

    response = JudgmentResponse(
        model="local-test",
        usage=TokenUsage(input_tokens=10, output_tokens=5),
        answers={"q1": NoulAnswer(noul=0.75)},
    )
    assert response.nouls["q1"].noul == 0.75
    ```

See Also:
    - [typevet.domain.judgment_answers][]: Answer types
    - [typevet.ports.judgment][]: JudgmentPort protocol
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typevet.domain.judgment_answers import (
    Answer,
    ChoiceAnswer,
    NoulAnswer,
    ScoreAnswer,
)


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Token counts for one judgment call when an adapter reports them.

    Attributes:
        input_tokens (int | None): Prompt tokens, when known.
        output_tokens (int | None): Completion tokens, when known.

    Examples:
        ```python
        usage = TokenUsage(input_tokens=12, output_tokens=3)
        assert usage.input_tokens == 12
        ```
    """

    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class JudgmentResponse:
    """Answers grouped by question type with model and optional usage metadata.

    Attributes:
        model (str): The model id that produced the answers.
        usage (TokenUsage): Token usage metadata for the call.
        answers (dict[str, Answer]): Answer objects keyed by question name.

    Examples:
        ```python
        from typevet.domain.judgment_answers import NoulAnswer

        response = JudgmentResponse(
            model="local-test",
            answers={"q1": NoulAnswer(noul=0.5)},
        )
        assert response.model == "local-test"
        ```
    """

    model: str
    usage: TokenUsage = field(default_factory=TokenUsage)
    answers: dict[str, Answer] = field(default_factory=dict)

    @property
    def nouls(self) -> dict[str, NoulAnswer]:
        """Return yes/no answers keyed by question id.

        Returns:
            Subset of ``answers`` whose values are ``NoulAnswer`` instances.
        """
        return {k: v for k, v in self.answers.items() if isinstance(v, NoulAnswer)}

    @property
    def choices(self) -> dict[str, ChoiceAnswer]:
        """Return choice answers keyed by question id.

        Returns:
            Subset of ``answers`` whose values are ``ChoiceAnswer`` instances.
        """
        return {k: v for k, v in self.answers.items() if isinstance(v, ChoiceAnswer)}

    @property
    def scores(self) -> dict[str, ScoreAnswer]:
        """Return score answers keyed by question id.

        Returns:
            Subset of ``answers`` whose values are ``ScoreAnswer`` instances.
        """
        return {k: v for k, v in self.answers.items() if isinstance(v, ScoreAnswer)}
