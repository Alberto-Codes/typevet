"""Compatibility shim for the Civil Comments loader (#147).

Prefer importing from ``typevet.evaluation.datasets.civil_comments`` in new code.

Examples:
    ```python
    from typevet.eval_civil_comments import balanced_sample
    ```

See Also:
    - [typevet.evaluation.datasets.civil_comments][]: New home for this module
"""

from typevet.evaluation.datasets.civil_comments import (
    DATASET_CONFIG,
    DATASET_ID,
    DATASETS_SERVER_MAX_PAGE,
    DATASETS_SERVER_ROWS_URL,
    DEFAULT_TIER_SEED,
    IS_TOXIC_NOUL_SCHEMA,
    IS_TOXIC_NOUL_SCHEMA_VERSION,
    PRIMARY_NOUL_NAME,
    SOURCE,
    SPLIT,
    TEXT_COLUMN,
    TIER_A_LIMIT,
    TIER_B_LIMIT,
    TIER_MANIFEST,
    TOXICITY_COLUMN,
    TOXICITY_THRESHOLD,
    CivilCommentsExample,
    TierName,
    balanced_sample,
    download_test_csv,
    iter_test_rows,
    load_test_split,
    load_tier,
    load_tier_a,
    load_tier_b,
    map_examples,
    map_row,
    proxy_label_for_toxicity,
    tier_limit,
    tier_manifest,
)

__all__ = [
    "DATASETS_SERVER_MAX_PAGE",
    "DATASETS_SERVER_ROWS_URL",
    "DATASET_CONFIG",
    "DATASET_ID",
    "DEFAULT_TIER_SEED",
    "IS_TOXIC_NOUL_SCHEMA",
    "IS_TOXIC_NOUL_SCHEMA_VERSION",
    "PRIMARY_NOUL_NAME",
    "SOURCE",
    "SPLIT",
    "TEXT_COLUMN",
    "TIER_A_LIMIT",
    "TIER_B_LIMIT",
    "TIER_MANIFEST",
    "TOXICITY_COLUMN",
    "TOXICITY_THRESHOLD",
    "CivilCommentsExample",
    "TierName",
    "balanced_sample",
    "download_test_csv",
    "iter_test_rows",
    "load_test_split",
    "load_tier",
    "load_tier_a",
    "load_tier_b",
    "map_examples",
    "map_row",
    "proxy_label_for_toxicity",
    "tier_limit",
    "tier_manifest",
]
