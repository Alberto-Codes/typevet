"""Compatibility shim for the loader eval runner (#147).

Prefer importing from ``typevet.evaluation.runner.core`` in new code.

Examples:
    ```python
    from typevet.eval_runner import run_eval_tasks
    ```

See Also:
    - [typevet.evaluation.runner.core][]: New home for this module
"""

from typevet.evaluation.runner.core import (
    DEFAULT_METRIC_EXACT_MATCH,
    DEFAULT_METRIC_NOUL_AGREEMENT,
    EvalRunReport,
    EvalTaskSpec,
    GenerationError,
    GenerationPort,
    GenerationRequest,
    run_eval_tasks,
)

__all__ = [
    "DEFAULT_METRIC_EXACT_MATCH",
    "DEFAULT_METRIC_NOUL_AGREEMENT",
    "EvalRunReport",
    "EvalTaskSpec",
    "GenerationError",
    "GenerationPort",
    "GenerationRequest",
    "run_eval_tasks",
]
