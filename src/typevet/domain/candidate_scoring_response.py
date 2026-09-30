"""Result types for candidate logprob scoring.

Examples:
    ```python
    from typevet.domain.candidate_scoring_response import (
        CandidateScoringResult,
        ScoredCandidate,
    )
    from typevet.domain.scoring_stage import ScoreStage

    result = CandidateScoringResult(
        model="local",
        stage=ScoreStage.PRE_SAMPLING,
        candidates=(ScoredCandidate("a", (1,), -0.5),),
    )
    assert result.candidates[0].logprob == -0.5
    ```

See Also:
    - [typevet.domain.candidate_scoring_validate][]: Fail-closed validation
    - [typevet.ports.scoring][]: CandidateScoringPort protocol
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typevet.domain.judgment_response import TokenUsage
from typevet.domain.scoring_stage import ScoreStage


@dataclass(frozen=True, slots=True)
class ScoredCandidate:
    """One candidate label, token ids, and raw logprob.

    Attributes:
        label (str): Candidate id matching the request label.
        token_ids (tuple[int, ...]): Token sequence matching the request.
        logprob (float): Raw pre-sampling logprob for this candidate.

    Examples:
        ```python
        scored = ScoredCandidate("billing", (101,), -0.25)
        assert scored.label == "billing"
        ```
    """

    label: str
    token_ids: tuple[int, ...]
    logprob: float


@dataclass(frozen=True, slots=True)
class ScoringTermination:
    """Optional backend termination metadata for a scoring call.

    Attributes:
        finish_reason (str | None): Backend stop reason when reported.

    Examples:
        ```python
        meta = ScoringTermination(finish_reason="length")
        assert meta.finish_reason == "length"
        ```
    """

    finish_reason: str | None = None


@dataclass(frozen=True, slots=True)
class CandidateScoringResult:
    """Complete identity-preserving scores for every requested candidate.

    Attributes:
        model (str): Model id that produced the scores.
        stage (ScoreStage): Stage used to obtain logprobs.
        candidates (tuple[ScoredCandidate, ...]): Scores in request order.
        usage (TokenUsage): Optional token usage metadata.
        termination (ScoringTermination | None): Optional stop metadata.
        off_option_mass (float | None): Probability mass that the model put
            outside the candidate tokens, in ``[0, 1]``. It is
            ``1 - sum(exp(logprob))`` over the raw candidate logprobs. ``None``
            means the backend did not return the full-vocabulary distribution,
            so the value is unavailable.

    Examples:
        ```python
        result = CandidateScoringResult(
            model="m",
            stage=ScoreStage.PRE_SAMPLING,
            candidates=(ScoredCandidate("a", (1,), -1.0),),
        )
        assert len(result.candidates) == 1
        assert result.off_option_mass is None
        ```
    """

    model: str
    stage: ScoreStage
    candidates: tuple[ScoredCandidate, ...]
    usage: TokenUsage = field(default_factory=TokenUsage)
    termination: ScoringTermination | None = None
    off_option_mass: float | None = None
