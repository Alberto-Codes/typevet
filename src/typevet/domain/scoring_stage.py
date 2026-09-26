"""Score stage enum for candidate logprob extraction.

Examples:
    ```python
    from typevet.domain.scoring_stage import ScoreStage

    assert ScoreStage.PRE_SAMPLING.value == "pre_sampling"
    ```

See Also:
    - [typevet.domain.candidate_scoring_request][]: Scoring request type
"""

from __future__ import annotations

from enum import StrEnum


class ScoreStage(StrEnum):
    """When logits are read relative to sampler state.

    Attributes:
        PRE_SAMPLING (ScoreStage): Softmax over full pre-sample logits (#116).
        POST_SAMPLING (ScoreStage): After sampler candidate list (not M1 stock).

    Examples:
        ```python
        stage = ScoreStage.PRE_SAMPLING
        assert stage == "pre_sampling"
        ```
    """

    PRE_SAMPLING = "pre_sampling"
    POST_SAMPLING = "post_sampling"
