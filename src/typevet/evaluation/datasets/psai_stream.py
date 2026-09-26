"""PSAI metadata row streaming helpers (dedupe, shuffle, JSONL, strip).

Strips heavy Hub columns (``screenshots``, ``events``, video paths) without
decoding image payloads. Used by ``typevet.evaluation.datasets.psai`` and parquet download.

Examples:
    Parse vendored smoke JSONL (CI):

    ```python
    from pathlib import Path

    from typevet.evaluation.datasets.psai_stream import (
        dedupe_shuffle_sample,
        iter_jsonl_rows,
    )

    text = Path("tests/fixtures/psai/metadata_smoke.jsonl").read_text()
    rows = dedupe_shuffle_sample(iter_jsonl_rows(text), limit=4, seed=0)
    assert len(rows) == 4
    ```

See Also:
    - [typevet.evaluation.datasets.psai][]: field map and JevBench export
    - [typevet.evaluation.datasets.psai_download][]: HF parquet metadata streaming
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping
from typing import Any, Final

UNIQUE_ID_COLUMN: Final[str] = "unique_data_id"
STRIP_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "screenshots",
        "events",
        "video_file",
        "dom_snaps_file",
        "metadata",
        "reasoning_steps",
        "tags",
        "taskId",
        "videoSize",
        "completedAt",
        "application_website",
    }
)
METADATA_COLUMNS: Final[tuple[str, ...]] = (
    UNIQUE_ID_COLUMN,
    "task_name",
    "category",
    "subCategory",
    "benchmark",
    "appType",
    "difficulty",
    "os",
    "requires_login",
)


def _seeded_order(items: list[Any], seed: int) -> list[Any]:
    """Return a deterministic permutation of ``items`` from ``seed``.

    Args:
        items: Values to reorder.
        seed: Stable seed for reproducible eval samples.

    Returns:
        New list in seeded order (does not use ``random``).
    """
    decorated = sorted(
        (
            hashlib.sha256(f"{seed}:{index}".encode()).digest(),
            index,
            item,
        )
        for index, item in enumerate(items)
    )
    return [item for _, _, item in decorated]


def strip_heavy_fields(row: Mapping[str, Any]) -> dict[str, Any]:
    """Copy a Hub row, dropping heavy or out-of-scope columns.

    Args:
        row: Raw PSAI metadata mapping (parquet or datasets-server).

    Returns:
        Shallow copy without ``STRIP_COLUMNS`` keys (no screenshot decode).
    """
    return {key: value for key, value in row.items() if key not in STRIP_COLUMNS}


def iter_jsonl_rows(jsonl_text: str) -> Iterator[dict[str, Any]]:
    """Yield metadata dicts from PSAI JSONL text.

    Args:
        jsonl_text: UTF-8 JSONL with ``unique_data_id`` and decision fields.

    Yields:
        Parsed row objects in file order (blank lines skipped).

    Raises:
        ValueError: When a line is invalid JSON or missing ``unique_data_id``.
    """
    for line_number, line in enumerate(jsonl_text.splitlines(), 1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            row = json.loads(stripped)
        except json.JSONDecodeError as exc:
            msg = f"PSAI JSONL line {line_number} is not valid JSON"
            raise ValueError(msg) from exc
        if UNIQUE_ID_COLUMN not in row:
            msg = f"PSAI JSONL line {line_number} needs {UNIQUE_ID_COLUMN!r} field"
            raise ValueError(msg)
        yield strip_heavy_fields(row)


def dedupe_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Keep the first row for each ``unique_data_id``.

    Args:
        rows: Metadata mappings in encounter order.

    Returns:
        Deduped list with heavy fields stripped.
    """
    seen: set[str] = set()
    kept: list[dict[str, Any]] = []
    for row in rows:
        cleaned = strip_heavy_fields(row)
        uid = cleaned.get(UNIQUE_ID_COLUMN)
        if not uid or not isinstance(uid, str):
            continue
        if uid in seen:
            continue
        seen.add(uid)
        kept.append(cleaned)
    return kept


def dedupe_shuffle_sample(
    rows: Iterable[Mapping[str, Any]],
    *,
    limit: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    """Dedupe on ``unique_data_id``, shuffle, then cap at ``limit``.

    Args:
        rows: Metadata stream or iterable.
        limit: Maximum rows after shuffle, or ``None`` for all deduped rows.
        seed: Seed for deterministic hash-based ordering.

    Returns:
        Sampled metadata dicts in seeded order.
    """
    deduped = dedupe_rows(rows)
    ordered = _seeded_order(deduped, seed)
    if limit is not None:
        return ordered[:limit]
    return ordered
