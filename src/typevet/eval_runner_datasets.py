"""Compatibility shim for eval runner datasets (#147).

Prefer importing from ``typevet.evaluation.runner.datasets`` in new code.

Examples:
    ```python
    from typevet.eval_runner_datasets import load_eval_tasks
    ```

See Also:
    - [typevet.evaluation.runner.datasets][]: New home for this module
"""

from typevet.evaluation.runner.datasets import (
    B77_NOUL,
    BOOLQ_ANSWER_NOUL_SCHEMA,
    BOOLQ_NOUL,
    DEFAULT_BANKING77_LIMIT,
    DEFAULT_BOOLQ_LIMIT,
    DEFAULT_SEED,
    REPORTS_UNAUTHORIZED_NOUL_SCHEMA,
    SUPPORTED_DATASETS,
    EvalDatasetName,
    EvalTaskSpec,
    load_eval_tasks,
    load_test_split,
    load_validation_split,
    serialize_boolq_state,
)

__all__ = [
    "B77_NOUL",
    "BOOLQ_ANSWER_NOUL_SCHEMA",
    "BOOLQ_NOUL",
    "DEFAULT_BANKING77_LIMIT",
    "DEFAULT_BOOLQ_LIMIT",
    "DEFAULT_SEED",
    "REPORTS_UNAUTHORIZED_NOUL_SCHEMA",
    "SUPPORTED_DATASETS",
    "EvalDatasetName",
    "EvalTaskSpec",
    "load_eval_tasks",
    "load_test_split",
    "load_validation_split",
    "serialize_boolq_state",
]
