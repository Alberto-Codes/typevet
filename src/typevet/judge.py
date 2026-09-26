"""Compatibility shim for scoring-backed judgment (#148).

Examples:
    ```python
    from typevet.judge import judge_with_scoring
    ```

See Also:
    - [typevet.runtime.judgment][]: Canonical facade
"""

from typevet.runtime.judgment import ScoringJudgmentAdapter, judge_with_scoring

__all__ = ["ScoringJudgmentAdapter", "judge_with_scoring"]
