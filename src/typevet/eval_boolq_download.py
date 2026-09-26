"""Compatibility shim for the BoolQ download helper (#147).

Prefer importing from ``typevet.evaluation.datasets.boolq_download`` in new code.

Examples:
    ```python
    from typevet.eval_boolq_download import download_validation_jsonl
    ```

See Also:
    - [typevet.evaluation.datasets.boolq_download][]: New home for this module
"""

from typevet.evaluation.datasets.boolq_download import (
    download_validation_jsonl,
)

__all__ = [
    "download_validation_jsonl",
]
