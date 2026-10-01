"""Response container for judgment port calls and the off-option receipt.

``OffOptionReceipt`` records the off-option mass, the caller threshold and
the guard flag for each answer (#353).

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
    - [typevet.domain.calibration][]: Calibration records on a response
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typevet.domain.calibration import CalibrationRecord
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
class OffOptionReceipt:
    """Off-option mass, the caller threshold and the guard flag for one answer.

    The flag is ``True`` only when both values are known and the mass is
    above the threshold. A ``None`` mass never sets the flag.

    Attributes:
        off_option_mass (float | None): Probability mass outside the option
            set, or ``None`` when the scorer does not report it.
        off_option_threshold (float | None): Caller threshold, or ``None``
            when the guard is off.
        off_option_flag (bool): ``True`` when the mass is above the threshold.

    Examples:
        ```python
        receipt = OffOptionReceipt.evaluate(mass=0.4, threshold=0.3)
        assert receipt.off_option_flag is True
        unknown = OffOptionReceipt.evaluate(mass=None, threshold=0.0)
        assert unknown.off_option_flag is False
        ```
    """

    off_option_mass: float | None = None
    off_option_threshold: float | None = None
    off_option_flag: bool = False

    @classmethod
    def evaluate(
        cls, *, mass: float | None, threshold: float | None
    ) -> OffOptionReceipt:
        """Build a receipt and set the flag from the mass and the threshold.

        Args:
            mass: Off-option mass from the scoring result, or ``None``.
            threshold: Caller threshold, or ``None`` when the guard is off.

        Returns:
            Receipt whose flag is ``True`` only when ``mass > threshold``.
        """
        flag = mass is not None and threshold is not None and mass > threshold
        return cls(
            off_option_mass=mass,
            off_option_threshold=threshold,
            off_option_flag=flag,
        )

    def as_dict(self) -> dict[str, float | bool | None]:
        """Return the receipt as a JSON-ready mapping with explicit names.

        Returns:
            ``off_option_mass``, ``off_option_threshold`` and
            ``off_option_flag``; an unknown value stays ``None`` (JSON null).
        """
        return {
            "off_option_mass": self.off_option_mass,
            "off_option_threshold": self.off_option_threshold,
            "off_option_flag": self.off_option_flag,
        }


@dataclass(frozen=True, slots=True)
class JudgmentResponse:
    """Answers grouped by question type with model and optional usage metadata.

    Attributes:
        model (str): The model id that produced the answers.
        usage (TokenUsage): Token usage metadata for the call.
        answers (dict[str, Answer]): Answer objects keyed by question name.
        off_option (dict[str, OffOptionReceipt]): Off-option receipt keyed by
            question name; empty when the adapter does not report one.
        calibration (dict[str, CalibrationRecord]): Raw and calibrated values
            keyed by question name, for each answer that a calibration map
            changed. Empty when no map applied.

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
    off_option: dict[str, OffOptionReceipt] = field(default_factory=dict)
    calibration: dict[str, CalibrationRecord] = field(default_factory=dict)

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
