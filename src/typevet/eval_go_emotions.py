"""Compatibility shim for the GoEmotions loader (#147).

Prefer importing from ``typevet.evaluation.datasets.go_emotions`` in new code.

Examples:
    ```python
    from typevet.eval_go_emotions import assert_state_keys_allowed
    ```

See Also:
    - [typevet.evaluation.datasets.go_emotions][]: New home for this module
"""

from typevet.evaluation.datasets.go_emotions import (
    BANNED_STATE_KEYS,
    CHOICE_LABELS,
    CONFIG,
    EMOTION_CHOICE_SCHEMA,
    NEUTRAL_LABEL,
    PRIMARY_CHOICE_NAME,
    PRUNED_LABELS,
    SOURCE,
    SPLIT,
    UPSTREAM_LABELS,
    GoEmotionsExample,
    assert_state_keys_allowed,
    balanced_sample,
    build_state,
    download_train_jsonl,
    iter_train_rows,
    labels_after_neutral_rule,
    load_train_split,
    map_examples,
    map_row,
    parse_upstream_labels,
    strict_gold_label,
    try_map_row,
)

__all__ = [
    "BANNED_STATE_KEYS",
    "CHOICE_LABELS",
    "CONFIG",
    "EMOTION_CHOICE_SCHEMA",
    "NEUTRAL_LABEL",
    "PRIMARY_CHOICE_NAME",
    "PRUNED_LABELS",
    "SOURCE",
    "SPLIT",
    "UPSTREAM_LABELS",
    "GoEmotionsExample",
    "assert_state_keys_allowed",
    "balanced_sample",
    "build_state",
    "download_train_jsonl",
    "iter_train_rows",
    "labels_after_neutral_rule",
    "load_train_split",
    "map_examples",
    "map_row",
    "parse_upstream_labels",
    "strict_gold_label",
    "try_map_row",
]
