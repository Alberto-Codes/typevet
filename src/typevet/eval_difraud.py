"""DIFrauD test-split loader and versioned ``is_scam`` Noul schema fixture.

DIFrauD (MIT, `difraud/difraud` on Hugging Face) labels short texts as deceptive
or not. typevet maps label ``1`` to ``scam`` and ``0`` to ``legit``. v1 default
domain is ``sms``; ``phishing`` and ``job_scams`` are optional configs on the
same card. This module does **not** ask Banking77 fraud questions
(``reports_unauthorized``, ``fraud_type``).

Examples:
    Load a vendored JSONL subset (no Hub in CI):

    ```python
    from pathlib import Path

    from typevet.eval_difraud import DEFAULT_DOMAIN, load_test_split

    jsonl = Path("tests/fixtures/difraud/sms_test_subset.jsonl").read_text()
    rows = load_test_split(jsonl_text=jsonl, domain=DEFAULT_DOMAIN, limit=4)
    assert {r.label for r in rows} <= {"scam", "legit"}
    ```

See Also:
    - [typevet.eval_banking77][]: Banking77 fraud-proxy loader (separate corpus)
    - [typevet.eval_civil_comments][]: Civil Comments toxicity loader (separate corpus)
    - [typevet.domain.decision_compile][]: compile Noul schemas for fixtures
"""

from __future__ import annotations

import hashlib
import json
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


SOURCE: Final[str] = "difraud"
SPLIT: Final[str] = "test"
TEXT_COLUMN: Final[str] = "text"
LABEL_COLUMN: Final[str] = "label"
DEFAULT_DOMAIN: Final[str] = "sms"
SUPPORTED_DOMAINS: Final[tuple[str, ...]] = ("sms", "phishing", "job_scams")
PRIMARY_NOUL_NAME: Final[str] = "is_scam"
IS_SCAM_NOUL_SCHEMA_VERSION: Final[str] = "1"

TEST_JSONL_URL: Final[str] = (
    "https://huggingface.co/datasets/difraud/difraud/resolve/main/{domain}/test.jsonl"
)

IS_SCAM_NOUL_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        PRIMARY_NOUL_NAME: {
            "type": "boolean",
            "instructions": (
                "Is this message a scam, phishing or social-engineering attempt?"
            ),
            "return_probabilities": True,
        }
    },
    "required": [PRIMARY_NOUL_NAME],
    "additionalProperties": False,
}


@dataclass(frozen=True, slots=True)
class DIFrauDExample:
    """One DIFrauD test row with scam or legit label.

    Attributes:
        text (str): Message body (the suspect text itself, not a customer report).
        label (str): ``scam`` or ``legit`` from the card's binary ``label`` column.
        domain (str): Card config name (``sms``, ``phishing``, or ``job_scams``).
        split (str): Dataset split name (always ``test`` for this loader).
        source (str): Corpus id for eval manifests.

    Examples:
        Build one row:

        ```python
        from typevet.eval_difraud import DIFrauDExample

        DIFrauDExample(text="Win cash now", label="scam", domain="sms")
        ```
    """

    text: str
    label: str
    domain: str = DEFAULT_DOMAIN
    split: str = SPLIT
    source: str = SOURCE


def _validate_domain(domain: str) -> None:
    if domain not in SUPPORTED_DOMAINS:
        supported = ", ".join(SUPPORTED_DOMAINS)
        msg = f"unsupported DIFrauD domain {domain!r}; choose one of: {supported}"
        raise ValueError(msg)


def map_row(
    text: str,
    is_deceptive: int | bool,
    *,
    domain: str = DEFAULT_DOMAIN,
) -> DIFrauDExample:
    """Map one DIFrauD row to a :class:`DIFrauDExample`.

    Args:
        text: Message text.
        is_deceptive: Hub ``label`` value; truthy means deceptive.
        domain: Card config name for metadata.

    Returns:
        Example labelled ``scam`` or ``legit``.
    """
    _validate_domain(domain)
    label = "scam" if is_deceptive else "legit"
    return DIFrauDExample(text=text, label=label, domain=domain)


def iter_test_rows(jsonl_text: str) -> Iterator[tuple[str, int]]:
    """Yield ``(text, label)`` pairs from DIFrauD test JSONL text.

    Args:
        jsonl_text: UTF-8 JSONL with ``text`` and ``label`` fields per line.

    Yields:
        Parsed row pairs in file order.

    Raises:
        ValueError: When a line is not valid JSON or columns are missing.
    """
    for line_number, line in enumerate(jsonl_text.splitlines(), 1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            row = json.loads(stripped)
        except json.JSONDecodeError as exc:
            msg = f"DIFrauD JSONL line {line_number} is not valid JSON"
            raise ValueError(msg) from exc
        if TEXT_COLUMN not in row or LABEL_COLUMN not in row:
            msg = (
                f"DIFrauD JSONL line {line_number} needs "
                f"{TEXT_COLUMN!r} and {LABEL_COLUMN!r} fields"
            )
            raise ValueError(msg)
        yield str(row[TEXT_COLUMN]), int(row[LABEL_COLUMN])


def map_examples(
    rows: Iterable[tuple[str, int]],
    *,
    domain: str = DEFAULT_DOMAIN,
) -> list[DIFrauDExample]:
    """Map many JSONL rows to examples.

    Args:
        rows: ``(text, label)`` pairs.
        domain: Card config name stored on each example.

    Returns:
        Mapped examples in input order.
    """
    _validate_domain(domain)
    return [map_row(text, flag, domain=domain) for text, flag in rows]


def download_test_jsonl(
    domain: str = DEFAULT_DOMAIN,
    *,
    client: httpx.Client | None = None,
) -> str:
    """Download one DIFrauD domain's public test JSONL.

    Args:
        domain: Card config name (``sms`` default).
        client: Optional shared HTTP client for tests.

    Returns:
        Raw JSONL text.
    """
    _validate_domain(domain)
    url = TEST_JSONL_URL.format(domain=domain)
    if client is None:
        with httpx.Client(timeout=60.0, follow_redirects=True) as owned:
            response = owned.get(url)
            response.raise_for_status()
            return response.text
    response = client.get(url)
    response.raise_for_status()
    return response.text


def load_test_split(
    *,
    domain: str = DEFAULT_DOMAIN,
    limit: int | None = None,
    seed: int = 0,
    jsonl_text: str | None = None,
    client: httpx.Client | None = None,
) -> list[DIFrauDExample]:
    """Load one DIFrauD domain test split with optional seeded reorder and cap.

    Unlike Banking77, rows are **not** class-balanced; the returned mix reflects
    the domain's natural base rate (see docs). Ordering uses hash-based seeding
    (not ``random``) so CI stays suppression-free.

    Args:
        domain: ``sms`` (v1 default), ``phishing``, or ``job_scams``.
        limit: Maximum rows after seeded reorder.
        seed: Seed for deterministic hash-based ordering.
        jsonl_text: Pre-fetched JSONL (CI fixtures). When ``None``, downloads
            the public Hub file for ``domain``.
        client: Optional HTTP client when downloading.

    Returns:
        Seed-ordered mapped test examples.
    """
    _validate_domain(domain)
    text = (
        jsonl_text
        if jsonl_text is not None
        else download_test_jsonl(domain, client=client)
    )
    examples = map_examples(iter_test_rows(text), domain=domain)
    examples = _seeded_order(examples, seed)
    if limit is not None:
        return examples[:limit]
    return examples
