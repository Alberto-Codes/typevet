"""Civil Comments test-split loader with τ=0.5 toxicity Noul oracle.

google/civil_comments (CC0) annotates public comment text with continuous
``toxicity`` scores. typevet maps scores at threshold τ=0.5 to ``toxic`` or
``not_toxic``. v1 uses ``toxicity`` only; other attribute columns are ignored.
Eval tiers A (200) and B (2000) are balanced halves on the test split; see
``evals/fixtures/civil_comments_tier_manifest_v1.yaml``.

Examples:
    Load a balanced tier-A slice from vendored CSV (no Hub in CI):

    ```python
    from pathlib import Path

    from typevet.eval_civil_comments import load_tier_a

    csv_text = Path("tests/fixtures/civil_comments/test_subset.csv").read_text()
    rows = load_tier_a(csv_text=csv_text, seed=0)
    assert {r.label for r in rows} <= {"toxic", "not_toxic"}
    ```

See Also:
    - [typevet.eval_banking77][]: Banking77 fraud-proxy loader (separate corpus)
    - [typevet.eval_difraud][]: DIFrauD scam/legit loader (separate corpus)
    - [typevet.eval_banking77][]: Banking77 fraud-proxy loader (separate corpus)
    - [typevet.eval_boolq][]: BoolQ passage yes/no Noul loader (separate corpus)
    - [typevet.eval_pubmedqa][]: PubMedQA labeled Choice loader (separate corpus)
    - [typevet.domain.decision_compile][]: compile Noul schemas for fixtures
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any, Final, Literal

import httpx

TierName = Literal["A", "B"]


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


SOURCE: Final[str] = "civil_comments"
SPLIT: Final[str] = "test"
TEXT_COLUMN: Final[str] = "text"
TOXICITY_COLUMN: Final[str] = "toxicity"
TOXICITY_THRESHOLD: Final[float] = 0.5
TIER_A_LIMIT: Final[int] = 200
TIER_B_LIMIT: Final[int] = 2000
DEFAULT_TIER_SEED: Final[int] = 0
PRIMARY_NOUL_NAME: Final[str] = "is_toxic"
IS_TOXIC_NOUL_SCHEMA_VERSION: Final[str] = "1"

DATASETS_SERVER_ROWS_URL: Final[str] = "https://datasets-server.huggingface.co/rows"
DATASETS_SERVER_MAX_PAGE: Final[int] = 100
DATASET_ID: Final[str] = "google/civil_comments"
DATASET_CONFIG: Final[str] = "default"

IS_TOXIC_NOUL_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        PRIMARY_NOUL_NAME: {
            "type": "boolean",
            "instructions": "Is this comment toxic or harmful to others?",
            "return_probabilities": True,
        }
    },
    "required": [PRIMARY_NOUL_NAME],
    "additionalProperties": False,
}

TIER_MANIFEST: Final[dict[str, Any]] = {
    "manifest_version": 1,
    "loader_issue": 72,
    "research_issue": 65,
    "dataset_id": DATASET_ID,
    "split": SPLIT,
    "toxicity_threshold": TOXICITY_THRESHOLD,
    "default_seed": DEFAULT_TIER_SEED,
    "tiers": {
        "A": {
            "total_limit": TIER_A_LIMIT,
            "balanced": True,
            "per_class": TIER_A_LIMIT // 2,
        },
        "B": {
            "total_limit": TIER_B_LIMIT,
            "balanced": True,
            "per_class": TIER_B_LIMIT // 2,
        },
    },
}


@dataclass(frozen=True, slots=True)
class CivilCommentsExample:
    """One Civil Comments test row with τ=0.5 binary label.

    Attributes:
        text (str): Public comment body (sensitive; fixtures use mild samples).
        toxicity (float): Hub continuous toxicity score in ``[0, 1]``.
        label (str): ``toxic`` or ``not_toxic`` from ``TOXICITY_THRESHOLD``.
        split (str): Dataset split name (always ``test`` for this loader).
        source (str): Corpus id for eval manifests.

    Examples:
        Build one row:

        ```python
        from typevet.eval_civil_comments import CivilCommentsExample

        CivilCommentsExample(text="hello", toxicity=0.1, label="not_toxic")
        ```
    """

    text: str
    toxicity: float
    label: str
    split: str = SPLIT
    source: str = SOURCE


def proxy_label_for_toxicity(
    toxicity_score: float,
    *,
    threshold: float = TOXICITY_THRESHOLD,
) -> str:
    """Collapse one toxicity score to a binary proxy label at τ.

    Args:
        toxicity_score: Continuous ``toxicity`` field from the dataset card.
        threshold: Decision threshold τ (default ``0.5`` per #65).

    Returns:
        ``toxic`` when ``toxicity_score >= threshold``, else ``not_toxic``.
    """
    return "toxic" if toxicity_score >= threshold else "not_toxic"


def map_row(
    text: str,
    toxicity_score: float,
    *,
    threshold: float = TOXICITY_THRESHOLD,
) -> CivilCommentsExample:
    """Map one row to a :class:`CivilCommentsExample`.

    Args:
        text: Comment text.
        toxicity_score: Continuous toxicity annotation.
        threshold: τ for binary collapse.

    Returns:
        Example with proxy label applied.
    """
    return CivilCommentsExample(
        text=text,
        toxicity=toxicity_score,
        label=proxy_label_for_toxicity(toxicity_score, threshold=threshold),
    )


def iter_test_rows(csv_text: str) -> Iterator[tuple[str, float]]:
    """Yield ``(text, toxicity)`` pairs from Civil Comments CSV text.

    Args:
        csv_text: UTF-8 CSV with at least ``text`` and ``toxicity`` columns.
            Extra attribute columns (``insult``, ``threat``, …) are ignored.

    Yields:
        Parsed row pairs in file order.

    Raises:
        ValueError: When required columns are missing.
    """
    reader = csv.DictReader(io.StringIO(csv_text))
    if reader.fieldnames is None:
        msg = "Civil Comments CSV is empty"
        raise ValueError(msg)
    fields = set(reader.fieldnames)
    if TEXT_COLUMN not in fields or TOXICITY_COLUMN not in fields:
        msg = (
            f"Civil Comments CSV needs {TEXT_COLUMN!r} and {TOXICITY_COLUMN!r} columns"
        )
        raise ValueError(msg)
    for row in reader:
        yield str(row[TEXT_COLUMN]), float(row[TOXICITY_COLUMN])


def map_examples(rows: Iterable[tuple[str, float]]) -> list[CivilCommentsExample]:
    """Map many CSV rows to examples.

    Args:
        rows: ``(text, toxicity)`` pairs.

    Returns:
        Mapped examples in input order.
    """
    return [map_row(text, score) for text, score in rows]


def balanced_sample(
    examples: list[CivilCommentsExample],
    limit: int | None,
    seed: int,
) -> list[CivilCommentsExample]:
    """Return equal toxic and not_toxic counts (tier A/B semantics).

    Args:
        examples: Full mapped test split.
        limit: Total cap, or ``None`` for every toxic row plus a matching
            ``not_toxic`` count.
        seed: Seed for deterministic hash-based ordering (not ``random``).

    Returns:
        Balanced sample in seeded order.
    """
    toxic = _seeded_order(
        [e for e in examples if e.label == "toxic"],
        seed,
    )
    other = _seeded_order(
        [e for e in examples if e.label != "toxic"],
        seed + 1,
    )
    if limit is not None:
        toxic = toxic[: limit // 2]
    sample = toxic + other[: len(toxic)]
    return _seeded_order(sample, seed + 2)


def tier_limit(tier: TierName) -> int:
    """Return the total row cap for eval tier ``A`` or ``B``.

    Args:
        tier: ``A`` (smoke) or ``B`` (default complementary).

    Returns:
        ``TIER_A_LIMIT`` or ``TIER_B_LIMIT``.
    """
    if tier == "A":
        return TIER_A_LIMIT
    return TIER_B_LIMIT


def tier_manifest() -> dict[str, Any]:
    """Return the seeded tier manifest constants for Civil Comments eval.

    Returns:
        Deep-copied manifest metadata (threshold, limits, default seed).
    """
    return json.loads(json.dumps(TIER_MANIFEST))


def _rows_to_csv(rows: Iterable[tuple[str, float]]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=[TEXT_COLUMN, TOXICITY_COLUMN])
    writer.writeheader()
    for text, toxicity in rows:
        writer.writerow({TEXT_COLUMN: text, TOXICITY_COLUMN: toxicity})
    return buffer.getvalue()


def _fetch_test_split_size(client: httpx.Client) -> int:
    response = client.get(
        "https://datasets-server.huggingface.co/info",
        params={"dataset": DATASET_ID},
    )
    response.raise_for_status()
    payload = response.json()
    return int(payload["dataset_info"][DATASET_CONFIG]["splits"][SPLIT]["num_examples"])


def download_test_csv(
    *,
    client: httpx.Client | None = None,
    page_size: int = 100,
) -> str:
    """Download the public Civil Comments test split as minimal CSV text.

    Fetches ``text`` and ``toxicity`` only via the Hugging Face datasets server
    (no ``datasets`` / parquet dependency). Other toxicity attributes are v1
    non-goals.

    Args:
        client: Optional shared HTTP client for tests.
        page_size: Rows per datasets-server page (max 100).

    Returns:
        CSV with ``text`` and ``toxicity`` columns.

    Raises:
        ValueError: When ``page_size`` is outside ``1..DATASETS_SERVER_MAX_PAGE``.
        httpx.HTTPError: When the datasets-server request fails.
    """
    if page_size < 1 or page_size > DATASETS_SERVER_MAX_PAGE:
        msg = (
            "page_size must be between 1 and "
            f"{DATASETS_SERVER_MAX_PAGE} for the datasets server"
        )
        raise ValueError(msg)

    def _download_with(active: httpx.Client) -> str:
        total = _fetch_test_split_size(active)
        collected: list[tuple[str, float]] = []
        offset = 0
        while offset < total:
            length = min(page_size, total - offset)
            response = active.get(
                DATASETS_SERVER_ROWS_URL,
                params={
                    "dataset": DATASET_ID,
                    "config": DATASET_CONFIG,
                    "split": SPLIT,
                    "offset": offset,
                    "length": length,
                },
            )
            response.raise_for_status()
            payload = response.json()
            for item in payload["rows"]:
                row = item["row"]
                collected.append((str(row[TEXT_COLUMN]), float(row[TOXICITY_COLUMN])))
            offset += length
        return _rows_to_csv(collected)

    if client is None:
        with httpx.Client(timeout=120.0) as owned:
            return _download_with(owned)
    return _download_with(client)


def load_test_split(
    *,
    limit: int | None = None,
    seed: int = DEFAULT_TIER_SEED,
    balanced: bool = False,
    csv_text: str | None = None,
    client: httpx.Client | None = None,
) -> list[CivilCommentsExample]:
    """Load the Civil Comments test split with optional balanced sampling.

    Args:
        limit: Row cap. With ``balanced=True``, total rows (half toxic, half
            not_toxic). Without balance, first ``limit`` rows in CSV order.
        seed: Seed for deterministic ordering when ``balanced=True``.
        balanced: When ``True``, apply tier-style class balance before return.
        csv_text: Pre-fetched CSV (CI fixtures). When ``None``, downloads via
            the public datasets-server API.
        client: Optional HTTP client when downloading.

    Returns:
        Mapped test examples.
    """
    text = csv_text if csv_text is not None else download_test_csv(client=client)
    examples = map_examples(iter_test_rows(text))
    if balanced:
        return balanced_sample(examples, limit, seed)
    if limit is not None:
        return examples[:limit]
    return examples


def load_tier(
    tier: TierName,
    *,
    seed: int = DEFAULT_TIER_SEED,
    csv_text: str | None = None,
    client: httpx.Client | None = None,
) -> list[CivilCommentsExample]:
    """Load a balanced tier-A or tier-B sample from the test split.

    Args:
        tier: ``A`` (200 rows) or ``B`` (2000 rows) per #65.
        seed: Seed for deterministic hash-based ordering.
        csv_text: Pre-fetched CSV (CI fixtures).
        client: Optional HTTP client when downloading.

    Returns:
        Balanced mapped test examples capped at the tier limit.
    """
    return load_test_split(
        limit=tier_limit(tier),
        seed=seed,
        balanced=True,
        csv_text=csv_text,
        client=client,
    )


def load_tier_a(
    *,
    seed: int = DEFAULT_TIER_SEED,
    csv_text: str | None = None,
    client: httpx.Client | None = None,
) -> list[CivilCommentsExample]:
    """Load balanced tier A (200 rows, 100 per class).

    Args:
        seed: Seed for deterministic ordering.
        csv_text: Pre-fetched CSV (CI fixtures).
        client: Optional HTTP client when downloading.

    Returns:
        Tier-A balanced sample.
    """
    return load_tier("A", seed=seed, csv_text=csv_text, client=client)


def load_tier_b(
    *,
    seed: int = DEFAULT_TIER_SEED,
    csv_text: str | None = None,
    client: httpx.Client | None = None,
) -> list[CivilCommentsExample]:
    """Load balanced tier B (2000 rows, 1000 per class).

    Args:
        seed: Seed for deterministic ordering.
        csv_text: Pre-fetched CSV (CI fixtures).
        client: Optional HTTP client when downloading.

    Returns:
        Tier-B balanced sample.
    """
    return load_tier("B", seed=seed, csv_text=csv_text, client=client)
