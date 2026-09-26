"""Hugging Face datasets-server download for go_emotions simplified JSONL.

Multi-label Hub rows stream at eval time; see ``typevet.evaluation.datasets.go_emotions`` for
strict exactly-one conversion, prune, and Choice mapping.

Examples:
    Download via an injected client in tests:

    ```python
    import httpx
    from typevet.evaluation.datasets.go_emotions_download import download_train_jsonl

    with httpx.Client() as client:
        jsonl = download_train_jsonl(client=client)
    ```

See Also:
    - [typevet.evaluation.datasets.go_emotions][]: public loader API
    - docs/reference/go-emotions-conversion-and-prune.md: v1 policy
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Final

import httpx

UPSTREAM_LABELS: Final[tuple[str, ...]] = (
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
    "neutral",
)

_DATASET_ID: Final[str] = "google-research-datasets/go_emotions"
_DATASET_CONFIG: Final[str] = "simplified"
_SPLIT: Final[str] = "train"
_DATASETS_SERVER_ROWS_URL: Final[str] = "https://datasets-server.huggingface.co/rows"
_DATASETS_SERVER_INFO_URL: Final[str] = "https://datasets-server.huggingface.co/info"
_DATASETS_SERVER_MAX_PAGE: Final[int] = 100


def _label_indices_to_names(indices: Sequence[int]) -> list[str]:
    return [UPSTREAM_LABELS[index] for index in indices]


def _rows_to_jsonl(rows: Iterable[Mapping[str, object]]) -> str:
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)


def _fetch_train_split_size(client: httpx.Client) -> int:
    response = client.get(
        _DATASETS_SERVER_INFO_URL,
        params={"dataset": _DATASET_ID},
    )
    response.raise_for_status()
    payload = response.json()
    return int(
        payload["dataset_info"][_DATASET_CONFIG]["splits"][_SPLIT]["num_examples"]
    )


def download_train_jsonl(
    *,
    client: httpx.Client | None = None,
    page_size: int = 100,
) -> str:
    """Download the public ``simplified`` train split as JSONL text.

    Streams ``text`` and ``labels`` via the Hugging Face datasets server (no
    ``datasets`` dependency). Label indices are expanded to Hub emotion names.

    Args:
        client: Optional shared HTTP client for tests.
        page_size: Rows per datasets-server page (max 100).

    Returns:
        JSONL with ``id``, ``text``, and ``labels`` (list of names).

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
        total = _fetch_train_split_size(active)
        collected: list[dict[str, object]] = []
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
            for item in payload["rows"]:
                row = item["row"]
                indices = row["labels"]
                if not isinstance(indices, list):
                    msg = "go_emotions labels must be a list of indices"
                    raise TypeError(msg)
                collected.append(
                    {
                        "id": row["id"],
                        "text": row["text"],
                        "labels": _label_indices_to_names(indices),
                    }
                )
            offset += length
        return _rows_to_jsonl(collected)

    if client is None:
        with httpx.Client(timeout=120.0) as owned:
            return _download_with(owned)
    return _download_with(client)
