"""Compatibility shim for the TPJEP eight-task fixture loader (#147).

Prefer importing from ``typevet.evaluation.tpjep.loader`` in new code.

Examples:
    ```python
    from typevet.eval_tpjep_loader import jevbench_row_to_scheduled_task
    ```

See Also:
    - [typevet.evaluation.tpjep.loader][]: New home for this module
"""

from typevet.evaluation.tpjep.loader import (
    EIGHT_TASK_IDS,
    PRIMARY_QUESTION_NAME,
    TPJEP_DATASET_GIT_COMMIT,
    TPJEP_LOCAL_CONCAT_HASH,
    TPJEP_LOCAL_CONCAT_RECIPE,
    TPJEP_MANIFEST_HASH,
    TPJEP_MANIFEST_RECIPE,
    Choice,
    Noul,
    Question,
    QuestionTypeName,
    Score,
    TpjepScheduledTask,
    jevbench_row_to_scheduled_task,
    load_eight_task_fixture,
    model_inputs_for_task,
)

__all__ = [
    "EIGHT_TASK_IDS",
    "PRIMARY_QUESTION_NAME",
    "TPJEP_DATASET_GIT_COMMIT",
    "TPJEP_LOCAL_CONCAT_HASH",
    "TPJEP_LOCAL_CONCAT_RECIPE",
    "TPJEP_MANIFEST_HASH",
    "TPJEP_MANIFEST_RECIPE",
    "Choice",
    "Noul",
    "Question",
    "QuestionTypeName",
    "Score",
    "TpjepScheduledTask",
    "jevbench_row_to_scheduled_task",
    "load_eight_task_fixture",
    "model_inputs_for_task",
]
