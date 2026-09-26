"""Compatibility shim for the PubMedQA loader (#147).

Prefer importing from ``typevet.evaluation.datasets.pubmedqa`` in new code.

Examples:
    ```python
    from typevet.eval_pubmedqa import assert_state_keys_allowed
    ```

See Also:
    - [typevet.evaluation.datasets.pubmedqa][]: New home for this module
"""

from typevet.evaluation.datasets.pubmedqa import (
    ANSWER_CHOICE_SCHEMA,
    BANNED_STATE_KEYS,
    CHOICE_LABELS,
    DATASET_CONFIG,
    DATASET_ID,
    DATASETS_SERVER_INFO_URL,
    DATASETS_SERVER_MAX_PAGE,
    DATASETS_SERVER_ROWS_URL,
    EXCLUDED_CONFIGS,
    PRIMARY_CHOICE_NAME,
    SOURCE,
    SPLIT,
    SUBSET,
    PubMedQAExample,
    assert_state_keys_allowed,
    balanced_sample,
    build_state,
    download_labeled_jsonl,
    iter_labeled_rows,
    load_labeled_split,
    map_examples,
    map_row,
    normalize_choice_label,
)

__all__ = [
    "ANSWER_CHOICE_SCHEMA",
    "BANNED_STATE_KEYS",
    "CHOICE_LABELS",
    "DATASETS_SERVER_INFO_URL",
    "DATASETS_SERVER_MAX_PAGE",
    "DATASETS_SERVER_ROWS_URL",
    "DATASET_CONFIG",
    "DATASET_ID",
    "EXCLUDED_CONFIGS",
    "PRIMARY_CHOICE_NAME",
    "SOURCE",
    "SPLIT",
    "SUBSET",
    "PubMedQAExample",
    "assert_state_keys_allowed",
    "balanced_sample",
    "build_state",
    "download_labeled_jsonl",
    "iter_labeled_rows",
    "load_labeled_split",
    "map_examples",
    "map_row",
    "normalize_choice_label",
]
