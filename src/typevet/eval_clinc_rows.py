"""Compatibility shim for CLINC row mapping (#147).

Prefer importing from ``typevet.evaluation.datasets.clinc_rows`` in new code.

Examples:
    ```python
    from typevet.eval_clinc_rows import assert_state_keys_allowed
    ```

See Also:
    - [typevet.evaluation.datasets.clinc_rows][]: New home for this module
"""

from typevet.evaluation.datasets.clinc_rows import (
    BANNED_STATE_KEYS,
    CONFIG,
    OOS_INTENT,
    SOURCE,
    SPLIT,
    ClincExample,
    assert_state_keys_allowed,
    balanced_sample,
    build_state,
    choice_labels_for_domain,
    intent_slug_from_record,
    iter_plus_rows,
    map_examples,
    map_row,
    plus_intent_names,
)

__all__ = [
    "BANNED_STATE_KEYS",
    "CONFIG",
    "OOS_INTENT",
    "SOURCE",
    "SPLIT",
    "ClincExample",
    "assert_state_keys_allowed",
    "balanced_sample",
    "build_state",
    "choice_labels_for_domain",
    "intent_slug_from_record",
    "iter_plus_rows",
    "map_examples",
    "map_row",
    "plus_intent_names",
]
