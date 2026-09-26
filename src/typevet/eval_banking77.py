"""Compatibility shim for the Banking77 loader (#147).

Prefer importing from ``typevet.evaluation.datasets.banking77`` in new code.

Examples:
    ```python
    from typevet.eval_banking77 import balanced_sample
    ```

See Also:
    - [typevet.evaluation.datasets.banking77][]: New home for this module
"""

from typevet.evaluation.datasets.banking77 import (
    FRAUD_INTENTS,
    INTENT_COLUMN,
    PRIMARY_NOUL_NAME,
    REPORTS_UNAUTHORIZED_NOUL_SCHEMA,
    SOURCE,
    SPLIT,
    TEST_CSV_URL,
    TEXT_COLUMN,
    Banking77Example,
    balanced_sample,
    download_test_csv,
    iter_test_rows,
    load_test_split,
    map_examples,
    map_row,
    proxy_label_for_intent,
)

__all__ = [
    "FRAUD_INTENTS",
    "INTENT_COLUMN",
    "PRIMARY_NOUL_NAME",
    "REPORTS_UNAUTHORIZED_NOUL_SCHEMA",
    "SOURCE",
    "SPLIT",
    "TEST_CSV_URL",
    "TEXT_COLUMN",
    "Banking77Example",
    "balanced_sample",
    "download_test_csv",
    "iter_test_rows",
    "load_test_split",
    "map_examples",
    "map_row",
    "proxy_label_for_intent",
]
