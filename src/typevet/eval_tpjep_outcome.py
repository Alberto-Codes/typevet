"""Compatibility shim for TPJEP answer-to-outcome mapping (#147).

Prefer importing from ``typevet.evaluation.tpjep.outcome`` in new code.

Examples:
    ```python
    from typevet.eval_tpjep_outcome import outcome_from_answer
    ```

See Also:
    - [typevet.evaluation.tpjep.outcome][]: New home for this module
"""

from typevet.evaluation.tpjep.outcome import (
    ChoiceAnswer,
    NoulAnswer,
    ScoreAnswer,
    TpjepScheduledTask,
    outcome_from_answer,
    prob_valid,
)

__all__ = [
    "ChoiceAnswer",
    "NoulAnswer",
    "ScoreAnswer",
    "TpjepScheduledTask",
    "outcome_from_answer",
    "prob_valid",
]
