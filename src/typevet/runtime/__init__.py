"""Thin orchestration facades over domain and ports (#148).

Examples:
    ```python
    from typevet.runtime import decide_categorical, judge_with_scoring
    ```

See Also:
    - [typevet.decide_categorical][]: Root compatibility shim
    - [typevet.judge][]: Root compatibility shim

Attributes:
    decide_categorical (callable): M1 categorical decision via scoring port.
    judge_with_scoring (callable): One-shot scoring-backed judgment helper.
    ScoringJudgmentAdapter (type): Sync ``JudgmentPort`` over ``CandidateScoringPort``.
    compose_scoring_prefix (callable): Degraded ChatML scoring prefix composition.
"""

from typevet.runtime.categorical import decide_categorical
from typevet.runtime.judgment import ScoringJudgmentAdapter, judge_with_scoring
from typevet.runtime.scoring_prefix import compose_scoring_prefix

__all__ = [
    "ScoringJudgmentAdapter",
    "compose_scoring_prefix",
    "decide_categorical",
    "judge_with_scoring",
]
