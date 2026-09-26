"""Compatibility shim for the PSAI metadata loader (#147).

Prefer importing from ``typevet.evaluation.datasets.psai`` in new code.

Examples:
    ```python
    from typevet.eval_psai import choice_sub_category
    ```

See Also:
    - [typevet.evaluation.datasets.psai][]: New home for this module
"""

from typevet.evaluation.datasets.psai import (
    APP_TYPE_LABELS,
    BANNED_STATE_KEYS,
    BENCHMARK_LABELS,
    CATEGORY_LABELS,
    DATASET_ID,
    DATASET_LICENSE,
    DEFAULT_SAMPLE_SEED,
    DIFFICULTY_LABELS,
    FAMILY,
    METADATA_DECISIONS_SCHEMA,
    METADATA_MANIFEST,
    OS_LABELS,
    OTHER_SUB_CATEGORY,
    PRIMARY_NOUL_NAME,
    SOURCE,
    SPLIT,
    SUB_CATEGORY_CHOICE_NAME,
    SUB_CATEGORY_LABELS,
    TASK_NAME_COLUMN,
    PsaiExample,
    choice_sub_category,
    dedupe_shuffle_sample,
    export_task,
    export_tasks,
    iter_jsonl_rows,
    iter_train_metadata_parquet,
    load_train_split,
    map_examples,
    map_row,
    metadata_manifest,
    noul_requires_login,
    psai_state,
    questions_payload,
    task_id,
    validate_task_state,
)

__all__ = [
    "APP_TYPE_LABELS",
    "BANNED_STATE_KEYS",
    "BENCHMARK_LABELS",
    "CATEGORY_LABELS",
    "DATASET_ID",
    "DATASET_LICENSE",
    "DEFAULT_SAMPLE_SEED",
    "DIFFICULTY_LABELS",
    "FAMILY",
    "METADATA_DECISIONS_SCHEMA",
    "METADATA_MANIFEST",
    "OS_LABELS",
    "OTHER_SUB_CATEGORY",
    "PRIMARY_NOUL_NAME",
    "SOURCE",
    "SPLIT",
    "SUB_CATEGORY_CHOICE_NAME",
    "SUB_CATEGORY_LABELS",
    "TASK_NAME_COLUMN",
    "PsaiExample",
    "choice_sub_category",
    "dedupe_shuffle_sample",
    "export_task",
    "export_tasks",
    "iter_jsonl_rows",
    "iter_train_metadata_parquet",
    "load_train_split",
    "map_examples",
    "map_row",
    "metadata_manifest",
    "noul_requires_login",
    "psai_state",
    "questions_payload",
    "task_id",
    "validate_task_state",
]
