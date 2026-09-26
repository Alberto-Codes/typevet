"""Compatibility shim for the DIFrauD loader (#147).

Prefer importing from ``typevet.evaluation.datasets.difraud`` in new code.

Examples:
    ```python
    from typevet.eval_difraud import download_test_jsonl
    ```

See Also:
    - [typevet.evaluation.datasets.difraud][]: New home for this module
"""

from typevet.evaluation.datasets.difraud import (
    DEFAULT_DOMAIN,
    IS_SCAM_NOUL_SCHEMA,
    IS_SCAM_NOUL_SCHEMA_VERSION,
    LABEL_COLUMN,
    PRIMARY_NOUL_NAME,
    SOURCE,
    SPLIT,
    SUPPORTED_DOMAINS,
    TEST_JSONL_URL,
    TEXT_COLUMN,
    DIFrauDExample,
    download_test_jsonl,
    iter_test_rows,
    load_test_split,
    map_examples,
    map_row,
)

__all__ = [
    "DEFAULT_DOMAIN",
    "IS_SCAM_NOUL_SCHEMA",
    "IS_SCAM_NOUL_SCHEMA_VERSION",
    "LABEL_COLUMN",
    "PRIMARY_NOUL_NAME",
    "SOURCE",
    "SPLIT",
    "SUPPORTED_DOMAINS",
    "TEST_JSONL_URL",
    "TEXT_COLUMN",
    "DIFrauDExample",
    "download_test_jsonl",
    "iter_test_rows",
    "load_test_split",
    "map_examples",
    "map_row",
]
