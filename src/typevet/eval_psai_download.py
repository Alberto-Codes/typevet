"""Compatibility shim for the PSAI download helper (#147).

Prefer importing from ``typevet.evaluation.datasets.psai_download`` in new code.

Examples:
    ```python
    from typevet.eval_psai_download import download_metadata_jsonl_from_parquet
    ```

See Also:
    - [typevet.evaluation.datasets.psai_download][]: New home for this module
"""

from typevet.evaluation.datasets.psai_download import (
    DATASET_CONFIG,
    DATASET_ID,
    DATASETS_SERVER_PARQUET_URL,
    METADATA_COLUMNS,
    SPLIT,
    download_metadata_jsonl_from_parquet,
    iter_parquet_metadata_rows,
    iter_train_metadata_parquet,
    list_train_parquet_urls,
    strip_heavy_fields,
)

__all__ = [
    "DATASETS_SERVER_PARQUET_URL",
    "DATASET_CONFIG",
    "DATASET_ID",
    "METADATA_COLUMNS",
    "SPLIT",
    "download_metadata_jsonl_from_parquet",
    "iter_parquet_metadata_rows",
    "iter_train_metadata_parquet",
    "list_train_parquet_urls",
    "strip_heavy_fields",
]
