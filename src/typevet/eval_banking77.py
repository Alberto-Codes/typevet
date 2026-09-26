"""Banking77 test-split loader with finvet six-intent fraud proxy collapse.

PolyAI Banking77 (CC BY 4.0) assigns one of 77 intent names per query. typevet
maps finvet's six ``FRAUD_INTENTS`` to proxy ``fraud``; every other intent is
``not_fraud``. Primary v1 Noul is ``reports_unauthorized`` (see fixture schema).

Examples:
    Load a balanced subset from vendored CSV text (no Hub in CI):

    ```python
    from pathlib import Path

    from typevet.eval_banking77 import load_test_split

    csv_text = Path("tests/fixtures/banking77/test_subset.csv").read_text()
    rows = load_test_split(csv_text=csv_text, balanced=True, limit=4, seed=0)
    assert {r.proxy_label for r in rows} == {"fraud", "not_fraud"}
    ```

See Also:
    - [typevet.eval_difraud][]: DIFrauD scam/legit loader (separate corpus)
    - [typevet.eval_civil_comments][]: Civil Comments toxicity loader (separate corpus)
    - [typevet.eval_pubmedqa][]: PubMedQA labeled Choice loader (separate corpus)
    - [typevet.domain.decision_compile][]: compile Noul schemas for fixtures
"""

from __future__ import annotations

import csv
import hashlib
import io
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any, Final

import httpx


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


TEST_CSV_URL: Final[str] = (
    "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/"
    "master/banking_data/test.csv"
)
TEXT_COLUMN: Final[str] = "text"
INTENT_COLUMN: Final[str] = "category"
SOURCE: Final[str] = "banking77"
SPLIT: Final[str] = "test"
PRIMARY_NOUL_NAME: Final[str] = "reports_unauthorized"

# Same six intents as finvet ``FRAUD_INTENTS`` (finvet ``data/banking77.py``).
FRAUD_INTENTS: Final[tuple[str, ...]] = (
    "card_payment_not_recognised",
    "cash_withdrawal_not_recognised",
    "direct_debit_payment_not_recognised",
    "transaction_charged_twice",
    "compromised_card",
    "extra_charge_on_statement",
)

REPORTS_UNAUTHORIZED_NOUL_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        PRIMARY_NOUL_NAME: {
            "type": "boolean",
            "description": (
                "Does the customer report a transaction they did not authorize?"
            ),
        }
    },
    "required": [PRIMARY_NOUL_NAME],
    "additionalProperties": False,
}


@dataclass(frozen=True, slots=True)
class Banking77Example:
    """One Banking77 test row with finvet proxy label.

    Attributes:
        text (str): Customer banking query.
        intent (str): Banking77 ``category`` intent name.
        proxy_label (str): ``fraud`` or ``not_fraud`` from six-intent collapse.
        split (str): Dataset split name (always ``test`` for this loader).
        source (str): Corpus id for eval manifests.

    Examples:
        Build one row:

        ```python
        from typevet.eval_banking77 import Banking77Example

        Banking77Example(text="hi", intent="Refund_not_showing_up", proxy_label="fraud")
        ```
    """

    text: str
    intent: str
    proxy_label: str
    split: str = SPLIT
    source: str = SOURCE


def proxy_label_for_intent(intent_name: str) -> str:
    """Collapse one Banking77 intent to finvet's binary proxy label.

    Args:
        intent_name: Banking77 ``category`` value.

    Returns:
        ``fraud`` when ``intent_name`` is in ``FRAUD_INTENTS``, else ``not_fraud``.
    """
    return "fraud" if intent_name in FRAUD_INTENTS else "not_fraud"


def map_row(text: str, intent_name: str) -> Banking77Example:
    """Map one CSV row to a :class:`Banking77Example`.

    Args:
        text: Customer query text.
        intent_name: Banking77 intent name.

    Returns:
        Example with proxy label applied.
    """
    return Banking77Example(
        text=text,
        intent=intent_name,
        proxy_label=proxy_label_for_intent(intent_name),
    )


def iter_test_rows(csv_text: str) -> Iterator[tuple[str, str]]:
    """Yield ``(text, intent)`` pairs from Banking77 test CSV text.

    Args:
        csv_text: UTF-8 CSV with ``text`` and ``category`` columns.

    Yields:
        Parsed row pairs in file order.

    Raises:
        ValueError: When required columns are missing.
    """
    reader = csv.DictReader(io.StringIO(csv_text))
    if reader.fieldnames is None:
        msg = "Banking77 CSV is empty"
        raise ValueError(msg)
    fields = set(reader.fieldnames)
    if TEXT_COLUMN not in fields or INTENT_COLUMN not in fields:
        msg = f"Banking77 CSV needs {TEXT_COLUMN!r} and {INTENT_COLUMN!r} columns"
        raise ValueError(msg)
    for row in reader:
        yield str(row[TEXT_COLUMN]), str(row[INTENT_COLUMN])


def map_examples(rows: Iterable[tuple[str, str]]) -> list[Banking77Example]:
    """Map many CSV rows to examples.

    Args:
        rows: ``(text, intent)`` pairs.

    Returns:
        Mapped examples in input order.
    """
    return [map_row(text, intent) for text, intent in rows]


def balanced_sample(
    examples: list[Banking77Example],
    limit: int | None,
    seed: int,
) -> list[Banking77Example]:
    """Return equal fraud and not_fraud counts (finvet ``balance`` semantics).

    Args:
        examples: Full mapped test split.
        limit: Total cap, or ``None`` for every fraud row plus a matching
            ``not_fraud`` count.
        seed: Seed for deterministic hash-based ordering (not ``random``).

    Returns:
        Balanced sample in seeded order.
    """
    fraud = _seeded_order(
        [e for e in examples if e.proxy_label == "fraud"],
        seed,
    )
    other = _seeded_order(
        [e for e in examples if e.proxy_label != "fraud"],
        seed + 1,
    )
    if limit is not None:
        fraud = fraud[: limit // 2]
    sample = fraud + other[: len(fraud)]
    return _seeded_order(sample, seed + 2)


def download_test_csv(
    url: str = TEST_CSV_URL,
    client: httpx.Client | None = None,
) -> str:
    """Download the public Banking77 test CSV.

    Args:
        url: CSV location (defaults to PolyAI task-specific-datasets).
        client: Optional shared HTTP client for tests.

    Returns:
        Raw CSV text.
    """
    if client is None:
        with httpx.Client(timeout=60.0) as owned:
            response = owned.get(url)
            response.raise_for_status()
            return response.text
    response = client.get(url)
    response.raise_for_status()
    return response.text


def load_test_split(
    *,
    limit: int | None = None,
    seed: int = 0,
    balanced: bool = False,
    csv_text: str | None = None,
    client: httpx.Client | None = None,
) -> list[Banking77Example]:
    """Load the Banking77 test split with optional balanced sampling.

    Args:
        limit: Row cap. With ``balanced=True``, total rows (half fraud, half
            not_fraud). Without balance, first ``limit`` rows in CSV order.
        seed: Seed for deterministic ordering when ``balanced=True``.
        balanced: When ``True``, apply finvet-style class balance before return.
        csv_text: Pre-fetched CSV (CI fixtures). When ``None``, downloads
            ``TEST_CSV_URL``.
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
