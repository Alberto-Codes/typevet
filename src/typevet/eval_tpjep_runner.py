"""Compatibility shim for the TPJEP task runner (#147).

Prefer importing from ``typevet.evaluation.tpjep.runner`` in new code.

Examples:
    ```python
    from typevet.eval_tpjep_runner import model_inputs_for_task
    ```

See Also:
    - [typevet.evaluation.tpjep.runner][]: New home for this module
"""

from typevet.evaluation.tpjep.runner import (
    TPJEP_DATASET_GIT_COMMIT,
    TPJEP_LOCAL_CONCAT_HASH,
    TPJEP_LOCAL_CONCAT_RECIPE,
    TPJEP_MANIFEST_HASH,
    TPJEP_MANIFEST_RECIPE,
    TPJEP_PROTOCOL_V0,
    JudgmentError,
    JudgmentPort,
    JudgmentResponse,
    JudgmentValidationError,
    ScoringValidationError,
    TpjepAttemptRecord,
    TpjepOutcome,
    TpjepRunConfig,
    TpjepRunMetadata,
    TpjepRunReceipt,
    TpjepRunSummary,
    TpjepScheduledTask,
    TransportError,
    model_inputs_for_task,
    outcome_from_answer,
    prob_valid,
    run_metadata_from_config,
    run_tpjep_tasks,
    run_tpjep_with_receipt,
    summarize_tpjep_records,
)

__all__ = [
    "TPJEP_DATASET_GIT_COMMIT",
    "TPJEP_LOCAL_CONCAT_HASH",
    "TPJEP_LOCAL_CONCAT_RECIPE",
    "TPJEP_MANIFEST_HASH",
    "TPJEP_MANIFEST_RECIPE",
    "TPJEP_PROTOCOL_V0",
    "JudgmentError",
    "JudgmentPort",
    "JudgmentResponse",
    "JudgmentValidationError",
    "ScoringValidationError",
    "TpjepAttemptRecord",
    "TpjepOutcome",
    "TpjepRunConfig",
    "TpjepRunMetadata",
    "TpjepRunReceipt",
    "TpjepRunSummary",
    "TpjepScheduledTask",
    "TransportError",
    "model_inputs_for_task",
    "outcome_from_answer",
    "prob_valid",
    "run_metadata_from_config",
    "run_tpjep_tasks",
    "run_tpjep_with_receipt",
    "summarize_tpjep_records",
]
