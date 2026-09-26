"""Public judgment helpers over the scoring-backed adapter.

Examples:
    ```python
    from typevet.judge import judge_with_scoring
    ```

See Also:
    - [typevet.adapters.outbound.judgment_scoring][]: Adapter implementation
"""

from typevet.adapters.outbound.judgment_scoring import (
    ScoringJudgmentAdapter,
    judge_with_scoring,
)

__all__ = ["ScoringJudgmentAdapter", "judge_with_scoring"]
