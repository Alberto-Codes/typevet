"""Request types for candidate logprob scoring.

Examples:
    ```python
    from typevet.domain.candidate_scoring_request import (
        CandidateScoringRequest,
        CandidateTokenSpec,
    )
    from typevet.domain.scoring_stage import ScoreStage

    request = CandidateScoringRequest(
        model="local",
        prefix="Pick:",
        candidates=(CandidateTokenSpec("a", (1,)),),
        stage=ScoreStage.PRE_SAMPLING,
    )
    assert request.prefix == "Pick:"
    ```

See Also:
    - [typevet.ports.scoring][]: CandidateScoringPort protocol
    - [typevet.domain.scoring_stage][]: ScoreStage enum
"""

from __future__ import annotations

from dataclasses import dataclass

from typevet.domain.errors import ScoringValidationError
from typevet.domain.scoring_stage import ScoreStage


@dataclass(frozen=True, slots=True)
class CandidateTokenSpec:
    """One labeled candidate as an ordered token-id sequence.

    Attributes:
        label (str): Stable candidate id (for example an enum string).
        token_ids (tuple[int, ...]): Encoded token ids for this candidate.

    Examples:
        ```python
        spec = CandidateTokenSpec("yes", (42,))
        assert spec.token_ids == (42,)
        ```
    """

    label: str
    token_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        """Reject empty labels or invalid token-id sequences.

        Raises:
            ScoringValidationError: When label or token ids are invalid.
        """
        if not self.label.strip():
            msg = "candidate label must be non-empty"
            raise ScoringValidationError(msg)
        if not self.token_ids:
            msg = f"candidate {self.label!r} must have at least one token id"
            raise ScoringValidationError(msg)
        if any(tid < 0 for tid in self.token_ids):
            msg = f"candidate {self.label!r} token ids must be non-negative"
            raise ScoringValidationError(msg)


@dataclass(frozen=True, slots=True)
class CandidateScoringRequest:
    """Identify model, prefix, candidates, and required score stage.

    Complete candidate coverage is required: adapters must return a score for
    every requested label and must not invent scores for omitted labels.

    Attributes:
        model (str): Backend model id or alias.
        prefix (str): Rendered prompt prefix ending before candidate tokens.
        candidates (tuple[CandidateTokenSpec, ...]): Ordered candidate set.
        stage (ScoreStage): Required logprob extraction stage.

    Examples:
        ```python
        request = CandidateScoringRequest(
            model="m",
            prefix="Q:",
            candidates=(CandidateTokenSpec("a", (1,)),),
        )
        assert request.stage is ScoreStage.PRE_SAMPLING
        ```
    """

    model: str
    prefix: str
    candidates: tuple[CandidateTokenSpec, ...]
    stage: ScoreStage = ScoreStage.PRE_SAMPLING

    def __post_init__(self) -> None:
        """Reject empty model, prefix, candidates, or duplicate ids.

        Raises:
            ScoringValidationError: When the ask violates coverage rules.
        """
        if not self.model.strip():
            msg = "model must be non-empty"
            raise ScoringValidationError(msg)
        if not self.prefix:
            msg = "prefix must be non-empty"
            raise ScoringValidationError(msg)
        if not self.candidates:
            msg = "at least one candidate is required"
            raise ScoringValidationError(msg)
        labels = [spec.label for spec in self.candidates]
        if len(set(labels)) != len(labels):
            msg = f"duplicate candidate labels: {labels!r}"
            raise ScoringValidationError(msg)
        sequences = [spec.token_ids for spec in self.candidates]
        if len(set(sequences)) != len(sequences):
            msg = "duplicate candidate token-id sequences are not allowed"
            raise ScoringValidationError(msg)
