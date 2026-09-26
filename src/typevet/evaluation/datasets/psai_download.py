"""Hugging Face parquet metadata streaming for PSAI (no screenshot decode).

Lists train parquet shards via the public datasets-server API, then reads
only ``METADATA_COLUMNS`` from each file (column projection — screenshots
and ``events`` are never loaded).

Examples:
    List shard URLs with an injected client:

    ```python
    import httpx

    from typevet.evaluation.datasets.psai_download import list_train_parquet_urls

    with httpx.Client() as client:
        urls = list_train_parquet_urls(client=client)
    assert urls[0].endswith(".parquet")
    ```

See Also:
    - [typevet.evaluation.datasets.psai_stream][]: dedupe, shuffle, JSONL helpers
    - [typevet.evaluation.datasets.psai][]: Decision mapping and task export
"""

from __future__ import annotations

import importlib
import io
import json
from collections.abc import Iterator
from typing import Any, Final

import httpx

from typevet.evaluation.datasets.psai_stream import METADATA_COLUMNS, strip_heavy_fields

try:
    _PYARROW_PARQUET = importlib.import_module("pyarrow.parquet")
except ImportError:
    _PYARROW_PARQUET = None

DATASET_ID: Final[str] = "anaisleila/computer-use-data-psai"
DATASET_CONFIG: Final[str] = "default"
SPLIT: Final[str] = "train"
DATASETS_SERVER_PARQUET_URL: Final[str] = (
    "https://datasets-server.huggingface.co/parquet"
)


def _require_pyarrow() -> Any:
    if _PYARROW_PARQUET is None:
        msg = (
            "PSAI parquet streaming requires the pyarrow package; "
            "install it in the eval environment before calling "
            "iter_train_metadata_parquet"
        )
        raise ImportError(msg)
    return _PYARROW_PARQUET


def list_train_parquet_urls(*, client: httpx.Client) -> list[str]:
    """Return HTTPS URLs for train parquet shards (datasets-server).

    Args:
        client: HTTP client (inject for tests).

    Returns:
        Ordered list of parquet file URLs.

    Raises:
        httpx.HTTPError: When the datasets-server request fails.
        KeyError: When the response shape is unexpected.
    """
    response = client.get(
        DATASETS_SERVER_PARQUET_URL,
        params={
            "dataset": DATASET_ID,
            "config": DATASET_CONFIG,
            "split": SPLIT,
        },
    )
    response.raise_for_status()
    payload = response.json()
    files = payload["parquet_files"]
    return [str(entry["url"]) for entry in files]


def iter_parquet_metadata_rows(
    parquet_bytes: bytes,
    *,
    columns: tuple[str, ...] = METADATA_COLUMNS,
) -> Iterator[dict[str, Any]]:
    """Yield metadata dicts from one parquet blob (column projection only).

    Args:
        parquet_bytes: Raw ``.parquet`` file bytes.
        columns: Column names to read (defaults to metadata-only set).

    Yields:
        Row dicts with heavy columns already absent from the parquet read.

    Raises:
        ImportError: When ``pyarrow`` is not installed.
    """
    pq = _require_pyarrow()
    table = pq.read_table(io.BytesIO(parquet_bytes), columns=list(columns))
    for row in table.to_pylist():
        yield strip_heavy_fields(row)


def iter_train_metadata_parquet(
    *,
    client: httpx.Client | None = None,
    urls: list[str] | None = None,
) -> Iterator[dict[str, Any]]:
    """Stream metadata rows from all train parquet shards.

    Args:
        client: HTTP client; a temporary client is opened when omitted.
        urls: Optional shard URL list (defaults to datasets-server listing).

    Yields:
        Metadata dicts in shard order (callers dedupe and shuffle).

    Raises:
        ImportError: When ``pyarrow`` is not installed.
        httpx.HTTPError: When a shard download fails.
    """

    def _stream(active: httpx.Client) -> Iterator[dict[str, Any]]:
        shard_urls = (
            urls if urls is not None else list_train_parquet_urls(client=active)
        )
        for url in shard_urls:
            response = active.get(url, follow_redirects=True, timeout=600.0)
            response.raise_for_status()
            yield from iter_parquet_metadata_rows(response.content)

    if client is None:
        with httpx.Client() as owned:
            yield from _stream(owned)
        return
    yield from _stream(client)


def download_metadata_jsonl_from_parquet(
    *,
    client: httpx.Client | None = None,
    urls: list[str] | None = None,
) -> str:
    """Download metadata columns from parquet shards as JSONL text.

    Args:
        client: Optional HTTP client.
        urls: Optional shard URLs (for tests).

    Returns:
        JSONL with one metadata object per line.

    Raises:
        ImportError: When ``pyarrow`` is not installed.
    """
    lines = [
        json.dumps(row, ensure_ascii=False)
        for row in iter_train_metadata_parquet(client=client, urls=urls)
    ]
    return "".join(f"{line}\n" for line in lines)
