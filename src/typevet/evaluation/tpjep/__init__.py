"""TPJEP evaluation: fixture loading, outcomes, records and the runner (#147).

Examples:
    ```python
    from typevet.evaluation.tpjep import TpjepRunConfig, load_eight_task_fixture
    ```

See Also:
    - [typevet.evaluation.tpjep.loader][]: Eight-task fixture and manifest hashes
    - [typevet.evaluation.tpjep.outcome][]: Answer to outcome mapping
    - [typevet.evaluation.tpjep.records][]: Attempt records and summaries
    - [typevet.evaluation.tpjep.runner][]: Offline and live task runs

Attributes:
    EIGHT_TASK_IDS (tuple): Task ids in the offline eight-task fixture.
    TPJEP_DATASET_GIT_COMMIT (str): Pinned upstream dataset commit.
    TPJEP_LOCAL_CONCAT_HASH (str): Hash of the local concatenated fixture.
    TPJEP_MANIFEST_HASH (str): Hash of the fixture manifest.
    TPJEP_PROTOCOL_V0 (str): Record protocol id for attempt rows.
    TpjepScheduledTask (type): One scheduled question with gold data.
    TpjepAttemptRecord (type): One attempt row for a scheduled task.
    TpjepOutcome (type): Outcome of a single attempt.
    TpjepRunSummary (type): Aggregate counts over attempt records.
    TpjepRunConfig (type): Run configuration for a TPJEP pass.
    TpjepRunMetadata (type): Provenance metadata for one run.
    TpjepRunReceipt (type): Run summary plus metadata and records.
    load_eight_task_fixture (function): Read the offline eight-task fixture.
    model_inputs_for_task (function): Build prompt inputs for a task.
    outcome_from_answer (function): Map a judgment answer to an outcome.
    prob_valid (function): Report whether probabilities are in bounds.
    summarize_tpjep_records (function): Aggregate attempt records.
    run_tpjep_tasks (function): Run scheduled tasks through a judgment port.
    run_tpjep_with_receipt (function): Run tasks and return a receipt.
"""

from typevet.evaluation.tpjep.loader import (
    EIGHT_TASK_IDS,
    TPJEP_DATASET_GIT_COMMIT,
    TPJEP_LOCAL_CONCAT_HASH,
    TPJEP_MANIFEST_HASH,
    TpjepScheduledTask,
    load_eight_task_fixture,
    model_inputs_for_task,
)
from typevet.evaluation.tpjep.outcome import outcome_from_answer, prob_valid
from typevet.evaluation.tpjep.records import (
    TPJEP_PROTOCOL_V0,
    TpjepAttemptRecord,
    TpjepOutcome,
    TpjepRunSummary,
    summarize_tpjep_records,
)
from typevet.evaluation.tpjep.runner import (
    TpjepRunConfig,
    TpjepRunMetadata,
    TpjepRunReceipt,
    run_tpjep_tasks,
    run_tpjep_with_receipt,
)

__all__ = [
    "EIGHT_TASK_IDS",
    "TPJEP_DATASET_GIT_COMMIT",
    "TPJEP_LOCAL_CONCAT_HASH",
    "TPJEP_MANIFEST_HASH",
    "TPJEP_PROTOCOL_V0",
    "TpjepAttemptRecord",
    "TpjepOutcome",
    "TpjepRunConfig",
    "TpjepRunMetadata",
    "TpjepRunReceipt",
    "TpjepRunSummary",
    "TpjepScheduledTask",
    "load_eight_task_fixture",
    "model_inputs_for_task",
    "outcome_from_answer",
    "prob_valid",
    "run_tpjep_tasks",
    "run_tpjep_with_receipt",
    "summarize_tpjep_records",
]
