"""Runtime facade for scoring-backed judgment (#148).

Examples:
    ```python
    from typevet.runtime.judgment import judge_with_scoring
    ```

See Also:
    - [typevet.judge][]: Root compatibility shim
    - [typevet.adapters.outbound.judgment_scoring][]: Adapter implementation
"""

from typevet.adapters.outbound.judgment_scoring import (
    ScoringJudgmentAdapter,
    judge_with_scoring,
)

__all__ = ["ScoringJudgmentAdapter", "judge_with_scoring"]
