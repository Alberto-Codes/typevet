"""Compatibility shim for the CLINC loader (#147).

Prefer importing from ``typevet.evaluation.datasets.clinc`` in new code.

Examples:
    ```python
    from typevet.eval_clinc import assert_state_keys_allowed
    ```

See Also:
    - [typevet.evaluation.datasets.clinc][]: New home for this module
"""

from typevet.evaluation.datasets.clinc import (
    BANNED_STATE_KEYS,
    CONFIG,
    DATASET_ID,
    DEFAULT_DOMAIN,
    IN_SCOPE_NOUL_SCHEMA,
    NOUL_LABELS,
    OOS_INTENT,
    PRIMARY_CHOICE_NAME,
    PRIMARY_NOUL_NAME,
    SOURCE,
    SPLIT,
    ClincExample,
    assert_state_keys_allowed,
    balanced_sample,
    build_state,
    choice_labels_for_domain,
    choice_schema_for_domain,
    domain_intent_map,
    domain_keys,
    download_plus_jsonl,
    intent_slug_from_record,
    iter_plus_rows,
    load_plus_split,
    map_examples,
    map_row,
    plus_intent_names,
)

__all__ = [
    "BANNED_STATE_KEYS",
    "CONFIG",
    "DATASET_ID",
    "DEFAULT_DOMAIN",
    "IN_SCOPE_NOUL_SCHEMA",
    "NOUL_LABELS",
    "OOS_INTENT",
    "PRIMARY_CHOICE_NAME",
    "PRIMARY_NOUL_NAME",
    "SOURCE",
    "SPLIT",
    "ClincExample",
    "assert_state_keys_allowed",
    "balanced_sample",
    "build_state",
    "choice_labels_for_domain",
    "choice_schema_for_domain",
    "domain_intent_map",
    "domain_keys",
    "download_plus_jsonl",
    "intent_slug_from_record",
    "iter_plus_rows",
    "load_plus_split",
    "map_examples",
    "map_row",
    "plus_intent_names",
]
