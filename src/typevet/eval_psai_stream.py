"""Compatibility shim for PSAI row streaming (#147).

Prefer importing from ``typevet.evaluation.datasets.psai_stream`` in new code.

Examples:
    ```python
    from typevet.eval_psai_stream import dedupe_rows
    ```

See Also:
    - [typevet.evaluation.datasets.psai_stream][]: New home for this module
"""

from typevet.evaluation.datasets.psai_stream import (
    METADATA_COLUMNS,
    STRIP_COLUMNS,
    UNIQUE_ID_COLUMN,
    dedupe_rows,
    dedupe_shuffle_sample,
    iter_jsonl_rows,
    strip_heavy_fields,
)

__all__ = [
    "METADATA_COLUMNS",
    "STRIP_COLUMNS",
    "UNIQUE_ID_COLUMN",
    "dedupe_rows",
    "dedupe_shuffle_sample",
    "iter_jsonl_rows",
    "strip_heavy_fields",
]
