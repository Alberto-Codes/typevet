"""HF datasets-server download for CLINC ``plus`` train split.

Examples:
    Stream one page with a shared client (tests use ``httpx.MockTransport``):

    ```python
    import httpx

    from typevet.evaluation.datasets.clinc_download import download_plus_jsonl

    with httpx.Client() as client:
        jsonl = download_plus_jsonl(client=client, page_size=100)
    ```

See Also:
    - [typevet.evaluation.datasets.clinc][]: domain-sharded loader facade
    - [typevet.evaluation.datasets.clinc_shard][]: ``plus_intent_names`` for index decoding
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from typing import Any, Final

import httpx

from typevet.evaluation.datasets.clinc_shard import plus_intent_names

DATASET_ID: Final[str] = "clinc/clinc_oos"
CONFIG: Final[str] = "plus"
SPLIT: Final[str] = "train"
DATASETS_SERVER_ROWS_URL: Final[str] = "https://datasets-server.huggingface.co/rows"
DATASETS_SERVER_INFO_URL: Final[str] = "https://datasets-server.huggingface.co/info"
DATASETS_SERVER_MAX_PAGE: Final[int] = 100


def _rows_to_jsonl(rows: Iterable[Mapping[str, Any]]) -> str:
    return "".join(json.dumps(dict(row), ensure_ascii=False) + "\n" for row in rows)


def _fetch_plus_split_size(client: httpx.Client) -> int:
    response = client.get(DATASETS_SERVER_INFO_URL, params={"dataset": DATASET_ID})
    response.raise_for_status()
    payload = response.json()
    return int(payload["dataset_info"][CONFIG]["splits"][SPLIT]["num_examples"])


def download_plus_jsonl(
    *,
    client: httpx.Client | None = None,
    page_size: int = 100,
) -> str:
    """Download the public ``plus`` train split as JSONL (slug intents).

    Args:
        client: Optional HTTP client; a temporary client is opened when omitted.
        page_size: Rows per datasets-server page (1-100).

    Returns:
        JSONL text with ``text`` and slug ``intent`` fields per line.

    Raises:
        ValueError: ``page_size`` is outside the allowed range.
    """
    if page_size < 1 or page_size > DATASETS_SERVER_MAX_PAGE:
        msg = (
            "page_size must be between 1 and "
            f"{DATASETS_SERVER_MAX_PAGE} for the datasets server"
        )
        raise ValueError(msg)
    names = plus_intent_names()

    def _download_with(active: httpx.Client) -> str:
        total = _fetch_plus_split_size(active)
        collected: list[dict[str, Any]] = []
        offset = 0
        while offset < total:
            length = min(page_size, total - offset)
            response = active.get(
                DATASETS_SERVER_ROWS_URL,
                params={
                    "dataset": DATASET_ID,
                    "config": CONFIG,
                    "split": SPLIT,
                    "offset": offset,
                    "length": length,
                },
            )
            response.raise_for_status()
            payload = response.json()
            for item in payload["rows"]:
                row = item["row"]
                collected.append(
                    {"text": row["text"], "intent": names[int(row["intent"])]}
                )
            offset += length
        return _rows_to_jsonl(collected)

    if client is None:
        with httpx.Client(timeout=120.0) as owned:
            return _download_with(owned)
    return _download_with(client)
