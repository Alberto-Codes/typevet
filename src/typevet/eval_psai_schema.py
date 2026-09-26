"""Compatibility shim for the PSAI metadata schema (#147).

Prefer importing from ``typevet.evaluation.datasets.psai_schema`` in new code.

Examples:
    ```python
    from typevet.eval_psai_schema import APP_TYPE_LABELS
    ```

See Also:
    - [typevet.evaluation.datasets.psai_schema][]: New home for this module
"""

from typevet.evaluation.datasets.psai_schema import (
    APP_TYPE_LABELS,
    BANNED_STATE_KEYS,
    BENCHMARK_LABELS,
    CATEGORY_LABELS,
    CONTRACT_ROW_MAX,
    CONTRACT_ROW_MIN,
    DATASET_ID,
    DATASET_LICENSE,
    DEFAULT_SAMPLE_SEED,
    DIFFICULTY_LABELS,
    FAMILY,
    METADATA_DECISIONS_SCHEMA,
    METADATA_MANIFEST,
    METADATA_SCHEMA_VERSION,
    OS_LABELS,
    OTHER_SUB_CATEGORY,
    PRIMARY_NOUL_NAME,
    SOURCE,
    SPLIT,
    SUB_CATEGORY_CHOICE_NAME,
    SUB_CATEGORY_LABELS,
    TASK_NAME_COLUMN,
)

__all__ = [
    "APP_TYPE_LABELS",
    "BANNED_STATE_KEYS",
    "BENCHMARK_LABELS",
    "CATEGORY_LABELS",
    "CONTRACT_ROW_MAX",
    "CONTRACT_ROW_MIN",
    "DATASET_ID",
    "DATASET_LICENSE",
    "DEFAULT_SAMPLE_SEED",
    "DIFFICULTY_LABELS",
    "FAMILY",
    "METADATA_DECISIONS_SCHEMA",
    "METADATA_MANIFEST",
    "METADATA_SCHEMA_VERSION",
    "OS_LABELS",
    "OTHER_SUB_CATEGORY",
    "PRIMARY_NOUL_NAME",
    "SOURCE",
    "SPLIT",
    "SUB_CATEGORY_CHOICE_NAME",
    "SUB_CATEGORY_LABELS",
    "TASK_NAME_COLUMN",
]
