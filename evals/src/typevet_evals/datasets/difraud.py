"""DIFrauD loaders, SMS splits and versioned ``is_scam`` Noul schema fixture.

DIFrauD (MIT, `difraud/difraud` on Hugging Face) labels short texts as deceptive
or not. typevet maps label ``1`` to ``scam`` and ``0`` to ``legit``. v1 default
domain is ``sms``; ``phishing`` and ``job_scams`` are optional configs on the
same card. This module does **not** ask Banking77 fraud questions
(``reports_unauthorized``, ``fraud_type``).

Examples:
    Load a vendored JSONL subset (no Hub in CI):

    ```python
    from pathlib import Path

    from typevet_evals.datasets.difraud import DEFAULT_DOMAIN, load_test_split

    jsonl = Path("tests/fixtures/difraud/sms_test_subset.jsonl").read_text()
    rows = load_test_split(jsonl_text=jsonl, domain=DEFAULT_DOMAIN, limit=4)
    assert {r.label for r in rows} <= {"scam", "legit"}
    ```

    Build SMS train, validation and held-out splits at the pinned revision
    (network; the held-out set excludes the 500 #236 test rows):

    ```python
    from typevet_evals.datasets.difraud import load_splits

    splits = load_splits(seed=0)
    ```

See Also:
    - [typevet_evals.datasets.banking77][]: Banking77 fraud-proxy loader (separate corpus)
    - [typevet_evals.datasets.civil_comments][]: Civil Comments toxicity loader
    - [typevet_evals.datasets.boolq][]: BoolQ passage yes/no Noul loader (separate corpus)
    - [typevet_evals.datasets.pubmedqa][]: PubMedQA labeled Choice loader (separate corpus)
    - [typevet.domain.decision_compile][]: compile Noul schemas for fixtures
    - [typevet_evals.throughput.public_workload][]: the #236 set that
      ``prior_measured_ids`` reproduces
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, replace
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

PINNED_REVISION: Final[str] = "aaaf94b336c563a14806bb4f3f58727bed9ed8d4"
SPLIT_JSONL_URL: Final[str] = (
    "https://huggingface.co/datasets/difraud/difraud/resolve/{revision}/sms/{split}.jsonl"
)
PRIOR_MEASURED_LIMIT: Final[int] = 500
PRIOR_MEASURED_SEED: Final[int] = 0

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
        from typevet_evals.datasets.difraud import DIFrauDExample

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


def record_id(text: str) -> str:
    """Return the DIFrauD record id of one message text.

    Upstream rows carry only ``text`` and ``label``, so the id is a hash of the
    exact UTF-8 text. Two rows with the same text share one id.

    Args:
        text: Message text, unchanged.

    Returns:
        ``sha256:`` and the first 16 hex characters of the text digest.
    """
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class DIFrauDRecord:
    """One DIFrauD SMS row with its record id.

    Attributes:
        record_id (str): Value of :func:`record_id` for the row text.
        example (DIFrauDExample): Mapped row; ``split`` names its source file.

    Examples:
        ```python
        from typevet_evals.datasets.difraud import DIFrauDRecord, map_row, record_id

        row = DIFrauDRecord(record_id("Win cash now"), map_row("Win cash now", 1))
        ```
    """

    record_id: str
    example: DIFrauDExample


@dataclass(frozen=True, slots=True)
class DIFrauDSplits:
    """DIFrauD SMS train, validation and held-out rows with no shared record id.

    Attributes:
        train (tuple[DIFrauDRecord, ...]): Rows from ``train.jsonl``.
        validation (tuple[DIFrauDRecord, ...]): Rows from ``validation.jsonl``
            whose id is not in ``train``.
        held_out (tuple[DIFrauDRecord, ...]): Rows from ``test.jsonl`` whose id
            is not in ``train``, ``validation`` or ``prior_measured_ids``.
        prior_measured_ids (frozenset[str]): Ids of the 500 #236 test rows.

    Examples:
        ```python
        from typevet_evals.datasets.difraud import load_splits

        splits = load_splits(seed=0)
        held_out_ids = {row.record_id for row in splits.held_out}
        assert not held_out_ids & splits.prior_measured_ids
        ```
    """

    train: tuple[DIFrauDRecord, ...]
    validation: tuple[DIFrauDRecord, ...]
    held_out: tuple[DIFrauDRecord, ...]
    prior_measured_ids: frozenset[str]


def download_split_jsonl(
    split: str,
    *,
    revision: str = PINNED_REVISION,
    client: httpx.Client | None = None,
) -> str:
    """Download one DIFrauD SMS split file at a fixed Hub revision.

    Args:
        split: ``train``, ``validation`` or ``test``.
        revision: Hub commit of ``difraud/difraud``.
        client: Optional shared HTTP client for tests.

    Returns:
        Raw JSONL text.
    """
    url = SPLIT_JSONL_URL.format(revision=revision, split=split)
    if client is None:
        with httpx.Client(timeout=60.0, follow_redirects=True) as owned:
            response = owned.get(url)
    else:
        response = client.get(url)
    response.raise_for_status()
    return response.text


def prior_measured_ids(test_jsonl: str) -> frozenset[str]:
    """Return the ids of the SMS test rows that #236 measured.

    Calls :func:`load_test_split` with the #236 limit and seed, which is the
    call ``typevet_evals.throughput.public_workload`` makes.

    Args:
        test_jsonl: SMS ``test.jsonl`` text.

    Returns:
        Ids of the ``PRIOR_MEASURED_LIMIT`` selected rows.
    """
    rows = load_test_split(
        jsonl_text=test_jsonl, limit=PRIOR_MEASURED_LIMIT, seed=PRIOR_MEASURED_SEED
    )
    return frozenset(record_id(row.text) for row in rows)


def _split_records(
    jsonl_text: str, split: str, taken: set[str], seed: int
) -> tuple[DIFrauDRecord, ...]:
    records: list[DIFrauDRecord] = []
    for example in map_examples(iter_test_rows(jsonl_text)):
        rid = record_id(example.text)
        if rid in taken:
            continue
        taken.add(rid)
        records.append(DIFrauDRecord(rid, replace(example, split=split)))
    return tuple(_seeded_order(records, seed))


def build_splits(
    *, train_jsonl: str, validation_jsonl: str, test_jsonl: str, seed: int = 0
) -> DIFrauDSplits:
    """Build SMS splits in which no record id appears twice.

    A record keeps the first place it appears in the order train, validation,
    held-out. The held-out set also drops every #236 row. ``seed`` orders each
    split; the #236 selection always uses ``PRIOR_MEASURED_SEED``.

    Args:
        train_jsonl: SMS ``train.jsonl`` text.
        validation_jsonl: SMS ``validation.jsonl`` text.
        test_jsonl: SMS ``test.jsonl`` text.
        seed: Seed for the hash-based order of each split.

    Returns:
        The three disjoint splits and the excluded #236 ids.
    """
    prior = prior_measured_ids(test_jsonl)
    taken: set[str] = set()
    train = _split_records(train_jsonl, "train", taken, seed)
    validation = _split_records(validation_jsonl, "validation", taken, seed)
    taken |= prior
    held_out = _split_records(test_jsonl, SPLIT, taken, seed)
    return DIFrauDSplits(train, validation, held_out, prior)


def load_splits(
    *,
    seed: int = 0,
    revision: str = PINNED_REVISION,
    client: httpx.Client | None = None,
) -> DIFrauDSplits:
    """Download the three SMS files and build the disjoint splits.

    Args:
        seed: Seed for the hash-based order of each split.
        revision: Hub commit of ``difraud/difraud``.
        client: Optional shared HTTP client for tests.

    Returns:
        Splits from :func:`build_splits`.
    """
    texts = {
        name: download_split_jsonl(name, revision=revision, client=client)
        for name in ("train", "validation", "test")
    }
    return build_splits(
        train_jsonl=texts["train"],
        validation_jsonl=texts["validation"],
        test_jsonl=texts["test"],
        seed=seed,
    )
