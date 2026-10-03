"""Quick, Draw! first-N recognized doodles with a local cache (#412).

The Quick, Draw! dataset holds simplified strokes for 345 categories, one
NDJSON file per category on Google's public bucket. Each file is near 80 MB,
so the loader streams it and keeps only the first ``n`` lines whose
``recognized`` field is true. It then closes the stream and writes those
lines to ``<cache>/<word>.first<n>.ndjson``. A cache hit sends no request.

A prefix has no SHA-256 pin: the receipt records the URL, ``n`` and the
key ids instead. The dataset is Google Quick, Draw! under CC BY 4.0. No
drawing goes into the repository.

Attributes:
    STROKES_URL (str): Simplified-strokes NDJSON URL; ``{word}`` is the
        URL-quoted category name.
    CACHE_ENV_VAR (str): Environment variable that sets the cache directory.
    DATASET_LICENSE (str): Licence of the drawings.
    DATASET_CREDIT (str): Credit line for the drawings.

Examples:
    ```python
    from typevet_evals.datasets.quickdraw import fetch_doodles

    doodles = fetch_doodles("cat", 5)
    assert doodles[0].word == "cat"
    ```

See Also:
    - [typevet_evals.doodle_duel][]: one ``Choice`` per doodle
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Final
from urllib.parse import quote

import httpx

STROKES_URL: Final[str] = (
    "https://storage.googleapis.com/quickdraw_dataset/full/simplified/{word}.ndjson"
)
CACHE_ENV_VAR: Final[str] = "TYPEVET_QUICKDRAW_CACHE"
DATASET_LICENSE: Final[str] = "CC BY 4.0"
DATASET_CREDIT: Final[str] = "Google Quick, Draw!"

_STROKE_AXES: Final[int] = 2
_DOWNLOAD_TIMEOUT: Final[float] = 600.0

Point = tuple[int, int]
Stroke = tuple[Point, ...]


@dataclass(frozen=True, slots=True)
class Doodle:
    """One recognized drawing in simplified strokes.

    Attributes:
        key_id (str): Unique drawing id from the dataset.
        word (str): Category name, the true label.
        countrycode (str): Two-letter country code of the player.
        strokes (tuple[tuple[tuple[int, int], ...], ...]): Strokes, each a
            sequence of ``(x, y)`` points in the 0 to 255 range.

    Examples:
        ```python
        Doodle("1", "cat", "US", (((0, 0), (10, 10)),))
        ```
    """

    key_id: str
    word: str
    countrycode: str
    strokes: tuple[Stroke, ...]


def parse_line(line: str) -> Doodle:
    """Parse one NDJSON line of simplified strokes.

    Args:
        line: One JSON object with ``word``, ``key_id``, ``countrycode`` and
            ``drawing`` (each stroke is ``[[x...], [y...]]``).

    Returns:
        The doodle.

    Raises:
        ValueError: When a stroke does not hold two equal-length axes.
    """
    record = json.loads(line)
    strokes: list[Stroke] = []
    for stroke in record["drawing"]:
        if len(stroke) < _STROKE_AXES or len(stroke[0]) != len(stroke[1]):
            msg = f"stroke in {record['key_id']} needs equal x and y lists"
            raise ValueError(msg)
        strokes.append(tuple(zip(stroke[0], stroke[1], strict=True)))
    return Doodle(
        key_id=str(record["key_id"]),
        word=str(record["word"]),
        countrycode=str(record.get("countrycode", "")),
        strokes=tuple(strokes),
    )


def first_recognized(lines: Iterable[str], n: int) -> list[str]:
    """Keep the first ``n`` lines whose ``recognized`` field is true.

    The function stops reading at the ``n``-th kept line, so a stream is
    read no further than needed.

    Args:
        lines: NDJSON lines in file order; blank lines are skipped.
        n: Lines to keep.

    Returns:
        Up to ``n`` raw lines, in file order.
    """
    kept: list[str] = []
    if n < 1:
        return kept
    for line in lines:
        if line.strip() and json.loads(line).get("recognized") is True:
            kept.append(line.strip())
            if len(kept) == n:
                break
    return kept


def resolve_cache_dir(cache_dir: Path | None = None) -> Path:
    """Return the Quick, Draw! cache directory.

    Args:
        cache_dir: Explicit directory; wins over the environment.

    Returns:
        ``cache_dir``, else ``$TYPEVET_QUICKDRAW_CACHE``, else
        ``~/.cache/typevet/quickdraw``.
    """
    if cache_dir is not None:
        return cache_dir
    configured = os.environ.get(CACHE_ENV_VAR)
    if configured:
        return Path(configured)
    return Path.home() / ".cache" / "typevet" / "quickdraw"


def cache_path(word: str, n: int, cache_dir: Path | None = None) -> Path:
    """Return the cache file for the first ``n`` doodles of ``word``.

    Args:
        word: Category name.
        n: Doodles in the file.
        cache_dir: Explicit directory, as in ``resolve_cache_dir``.

    Returns:
        ``<cache>/<word>.first<n>.ndjson``.
    """
    return resolve_cache_dir(cache_dir) / f"{word}.first{n}.ndjson"


def _stream_first(word: str, n: int, client: httpx.Client) -> list[str]:
    url = STROKES_URL.format(word=quote(word))
    with client.stream(
        "GET", url, follow_redirects=True, timeout=_DOWNLOAD_TIMEOUT
    ) as response:
        response.raise_for_status()
        return first_recognized(response.iter_lines(), n)


def fetch_doodles(
    word: str,
    n: int,
    *,
    cache_dir: Path | None = None,
    client: httpx.Client | None = None,
) -> tuple[Doodle, ...]:
    """Return the first ``n`` recognized doodles of ``word``.

    A cache hit reads the file and sends no request. A miss streams the
    category file, keeps the first ``n`` recognized lines, closes the stream
    and writes the cache file.

    Args:
        word: Category name, as in the dataset ``categories.txt``.
        n: Doodles to return; at least 1.
        cache_dir: Explicit cache directory, as in ``resolve_cache_dir``.
        client: HTTP client for a miss; a new client is opened and closed
            when ``None``.

    Returns:
        ``n`` doodles in file order.

    Raises:
        ValueError: When ``n`` is below 1, or the file or the cache holds
            fewer than ``n`` recognized doodles.
    """
    if n < 1:
        msg = f"n must be at least 1, got {n}"
        raise ValueError(msg)
    path = cache_path(word, n, cache_dir)
    if path.exists():
        lines = [x for x in path.read_text(encoding="utf-8").splitlines() if x]
    elif client is None:
        with httpx.Client() as own_client:
            lines = _stream_first(word, n, own_client)
    else:
        lines = _stream_first(word, n, client)
    if len(lines) != n:
        msg = f"{word}: expected {n} recognized doodles, found {len(lines)}"
        raise ValueError(msg)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(f"{path.name}.part")
        partial.write_text("\n".join(lines) + "\n", encoding="utf-8")
        partial.replace(path)
    return tuple(parse_line(line) for line in lines)
