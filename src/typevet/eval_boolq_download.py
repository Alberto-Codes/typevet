"""Hugging Face datasets-server download for BoolQ validation JSONL.

Bulk passage text streams at eval time; see ``typevet.eval_boolq`` for mapping,
tiers, and JevBench export.

Examples:
    Download via an injected client in tests (production callers use
    :func:`typevet.eval_boolq.load_validation_split`):

    ```python
    import httpx
    from typevet.eval_boolq_download import download_validation_jsonl

    with httpx.Client() as client:
        jsonl = download_validation_jsonl(client=client)
    ```

See Also:
    - [typevet.eval_boolq][]: public loader API and task export
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Final

import httpx

_DATASET_ID: Final[str] = "google/boolq"
_DATASET_CONFIG: Final[str] = "default"
_SPLIT: Final[str] = "validation"
_PASSAGE_COLUMN: Final[str] = "passage"
_QUESTION_COLUMN: Final[str] = "question"
_ANSWER_COLUMN: Final[str] = "answer"
_IDX_COLUMN: Final[str] = "idx"
_DATASETS_SERVER_ROWS_URL: Final[str] = "https://datasets-server.huggingface.co/rows"
_DATASETS_SERVER_MAX_PAGE: Final[int] = 100


def _rows_to_jsonl(rows: Iterable[tuple[str, str, bool, int]]) -> str:
    lines: list[str] = []
    for passage, question, answer, idx in rows:
        payload = {
            _PASSAGE_COLUMN: passage,
            _QUESTION_COLUMN: question,
            _ANSWER_COLUMN: answer,
            _IDX_COLUMN: idx,
        }
        lines.append(json.dumps(payload, ensure_ascii=False))
    return "\n".join(lines) + ("\n" if lines else "")


def _fetch_validation_split_size(client: httpx.Client) -> int:
    response = client.get(
        "https://datasets-server.huggingface.co/info",
        params={"dataset": _DATASET_ID},
    )
    response.raise_for_status()
    payload = response.json()
    return int(
        payload["dataset_info"][_DATASET_CONFIG]["splits"][_SPLIT]["num_examples"]
    )


def download_validation_jsonl(
    *,
    client: httpx.Client | None = None,
    page_size: int = 100,
) -> str:
    """Download the public BoolQ validation split as JSONL text.

    Streams passage and question fields via the Hugging Face datasets server
    (no ``datasets`` / parquet dependency). Bulk passage text is **not**
    vendored in git (#76); use this for offline eval caches or periodic runs.

    Args:
        client: Optional shared HTTP client for tests.
        page_size: Rows per datasets-server page (max 100).

    Returns:
        JSONL with ``passage``, ``question``, ``answer``, and ``idx`` fields.

    Raises:
        ValueError: When ``page_size`` is outside ``1.._DATASETS_SERVER_MAX_PAGE``.
        httpx.HTTPError: When the datasets-server request fails.
    """
    if page_size < 1 or page_size > _DATASETS_SERVER_MAX_PAGE:
        msg = (
            "page_size must be between 1 and "
            f"{_DATASETS_SERVER_MAX_PAGE} for the datasets server"
        )
        raise ValueError(msg)

    def _download_with(active: httpx.Client) -> str:
        total = _fetch_validation_split_size(active)
        collected: list[tuple[str, str, bool, int]] = []
        offset = 0
        while offset < total:
            length = min(page_size, total - offset)
            response = active.get(
                _DATASETS_SERVER_ROWS_URL,
                params={
                    "dataset": _DATASET_ID,
                    "config": _DATASET_CONFIG,
                    "split": _SPLIT,
                    "offset": offset,
                    "length": length,
                },
            )
            response.raise_for_status()
            payload = response.json()
            for page_index, item in enumerate(payload["rows"]):
                row = item["row"]
                row_idx = (
                    int(row[_IDX_COLUMN]) if _IDX_COLUMN in row else offset + page_index
                )
                collected.append(
                    (
                        str(row[_PASSAGE_COLUMN]),
                        str(row[_QUESTION_COLUMN]),
                        bool(row[_ANSWER_COLUMN]),
                        row_idx,
                    )
                )
            offset += length
        return _rows_to_jsonl(collected)

    if client is None:
        with httpx.Client(timeout=120.0) as owned:
            return _download_with(owned)
    return _download_with(client)
