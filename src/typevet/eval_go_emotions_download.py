"""Compatibility shim for the GoEmotions download helper (#147).

Prefer importing from ``typevet.evaluation.datasets.go_emotions_download`` in new code.

Examples:
    ```python
    from typevet.eval_go_emotions_download import download_train_jsonl
    ```

See Also:
    - [typevet.evaluation.datasets.go_emotions_download][]: New home for this module
"""

from typevet.evaluation.datasets.go_emotions_download import (
    UPSTREAM_LABELS,
    download_train_jsonl,
)

__all__ = [
    "UPSTREAM_LABELS",
    "download_train_jsonl",
]
