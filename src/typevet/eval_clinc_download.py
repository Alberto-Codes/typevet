"""Compatibility shim for the CLINC download helper (#147).

Prefer importing from ``typevet.evaluation.datasets.clinc_download`` in new code.

Examples:
    ```python
    from typevet.eval_clinc_download import download_plus_jsonl
    ```

See Also:
    - [typevet.evaluation.datasets.clinc_download][]: New home for this module
"""

from typevet.evaluation.datasets.clinc_download import (
    CONFIG,
    DATASET_ID,
    DATASETS_SERVER_INFO_URL,
    DATASETS_SERVER_MAX_PAGE,
    DATASETS_SERVER_ROWS_URL,
    SPLIT,
    download_plus_jsonl,
    plus_intent_names,
)

__all__ = [
    "CONFIG",
    "DATASETS_SERVER_INFO_URL",
    "DATASETS_SERVER_MAX_PAGE",
    "DATASETS_SERVER_ROWS_URL",
    "DATASET_ID",
    "SPLIT",
    "download_plus_jsonl",
    "plus_intent_names",
]
