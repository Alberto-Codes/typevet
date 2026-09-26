"""Candidate scoring port: prefix and token ids in, logprobs out.

The protocol is IO-free: live llama.cpp pre-sampling adapters are implemented
outside this module ([#27](https://github.com/Alberto-Codes/typevet/issues/27)).

Examples:
    ```python
    from typevet.ports.scoring import CandidateScoringPort


    def use(port: CandidateScoringPort) -> None:
        port.score_candidates  # structural check
    ```

See Also:
    - [typevet.domain.candidate_scoring_request][]: Request types
    - [typevet.domain.candidate_scoring_response][]: Result types
"""

from __future__ import annotations

from typing import Protocol

from typevet.domain.candidate_scoring_request import CandidateScoringRequest
from typevet.domain.candidate_scoring_response import CandidateScoringResult


class CandidateScoringPort(Protocol):
    """Structural protocol for complete candidate logprob scoring.

    Examples:
        ```python
        from typevet.ports.scoring import CandidateScoringPort

        port: CandidateScoringPort
        _ = port.score_candidates
        ```
    """

    def score_candidates(
        self, request: CandidateScoringRequest
    ) -> CandidateScoringResult:
        """Score every requested candidate at the contracted stage.

        Args:
            request: Model id, rendered prefix, ordered candidates, and stage.

        Returns:
            Identity-preserving ``CandidateScoringResult`` with one raw logprob
            per requested candidate label.

        Raises:
            typevet.domain.errors.ScoringError: When the call fails before a
                valid result exists. Concrete subclasses depend on the adapter.
            typevet.domain.errors.ScoringValidationError: Missing, duplicate, or
                unexpected candidate labels, or non-finite logprobs (fail-closed;
                remaining candidates are never renormalized to fill gaps).
            typevet.domain.errors.ScoringUnsupportedCapabilityError: When the
                backend or adapter cannot honor ``request.stage``.
        """
        ...


ScoringPort = CandidateScoringPort
