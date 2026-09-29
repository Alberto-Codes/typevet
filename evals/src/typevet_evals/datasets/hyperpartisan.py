"""Hyperpartisan byarticle train loader, HTML cleanup, and holdout export.

SemEval hyperpartisan news detection (CC BY 4.0) provides **645 byarticle**
training articles with article-level ``hyperpartisan`` labels. Official
by-article test labels are unpublished; typevet uses a **seeded stratified
holdout** from the train pool for eval. **bypublisher** rows (distant publisher
labels) are out of scope and must not appear in fixtures or loaders.

v1 primary Noul is ``hyperpartisan`` (boolean + ``return_probabilities``).
Task ``state`` is ``{title, body}`` after HTML cleanup and body length cap.
In-repo fixtures are tiny smoke JSONL only; bulk XML/JSONL stays offline.

Examples:
    Load holdout rows from vendored smoke JSONL (no corpus download in CI):

    ```python
    from pathlib import Path

    from typevet_evals.datasets.hyperpartisan import (
        export_tasks,
        load_holdout_split,
    )

    jsonl = Path("tests/fixtures/hyperpartisan/byarticle_smoke.jsonl").read_text()
    rows = load_holdout_split(jsonl_text=jsonl, seed=0)
    tasks = export_tasks(rows)
    assert tasks[0]["expected"]["hyperpartisan"] in {True, False}
    ```

See Also:
    - [typevet_evals.datasets.boolq][]: BoolQ passage yes/no Noul loader (separate corpus)
    - [typevet_evals.datasets.civil_comments][]: Civil Comments toxicity loader
    - [typevet_evals.datasets.difraud][]: DIFrauD scam/legit loader (separate corpus)
    - [typevet.domain.decision_compile][]: compile Noul schemas for fixtures
"""

from __future__ import annotations

import hashlib
import html
import json
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal

SplitName = Literal["train", "holdout"]
PartitionName = Literal["train", "holdout", "all"]

_HTML_TAG_RE: Final[re.Pattern[str]] = re.compile(r"<[^>]+>")

SOURCE: Final[str] = "hyperpartisan"
FAMILY: Final[str] = "hyperpartisan"
SPLIT_BYARTICLE: Final[str] = "byarticle-train"
EXCLUDED_SPLIT: Final[str] = "bypublisher"
ARTICLE_ID_COLUMN: Final[str] = "article_id"
TITLE_COLUMN: Final[str] = "title"
BODY_COLUMN: Final[str] = "body"
LABEL_COLUMN: Final[str] = "hyperpartisan"
CORPUS_BYARTICLE_TRAIN_ROWS: Final[int] = 645
DEFAULT_HOLDOUT_SEED: Final[int] = 0
HOLDOUT_FRACTION: Final[float] = 0.2
BODY_MAX_CHARS: Final[int] = 8000
CONTRACT_ROW_MIN: Final[int] = 12
CONTRACT_ROW_MAX: Final[int] = 24
PRIMARY_NOUL_NAME: Final[str] = "hyperpartisan"
HYPERPARTISAN_NOUL_SCHEMA_VERSION: Final[str] = "1"
DATASET_ID: Final[str] = "SemEval-2019/Task4-hyperpartisan-news-detection"
DATASET_LICENSE: Final[str] = "CC BY 4.0"

BANNED_STATE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "hyperpartisan",
        "expected",
        "label",
        "ground_truth",
        "publisher",
        "publisher_name",
    }
)

HYPERPARTISAN_NOUL_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        PRIMARY_NOUL_NAME: {
            "type": "boolean",
            "instructions": "Is this news article hyperpartisan?",
            "return_probabilities": True,
        }
    },
    "required": [PRIMARY_NOUL_NAME],
    "additionalProperties": False,
}

HOLDOUT_MANIFEST: Final[dict[str, Any]] = {
    "manifest_version": 1,
    "loader_issue": 81,
    "research_issue": 68,
    "dataset_id": DATASET_ID,
    "split": SPLIT_BYARTICLE,
    "license": DATASET_LICENSE,
    "corpus_byarticle_train_rows": CORPUS_BYARTICLE_TRAIN_ROWS,
    "excluded_splits": [EXCLUDED_SPLIT],
    "default_holdout_seed": DEFAULT_HOLDOUT_SEED,
    "holdout_fraction": HOLDOUT_FRACTION,
    "body_max_chars": BODY_MAX_CHARS,
    "non_goals": [
        "bypublisher distant publisher labels",
        "SemEval official by-article test claim",
        "full 645-row corpus in git",
    ],
}


def _seeded_order(items: list[Any], seed: int) -> list[Any]:
    decorated = sorted(
        (
            hashlib.sha256(f"{seed}:{index}".encode()).digest(),
            index,
            item,
        )
        for index, item in enumerate(items)
    )
    return [item for _, _, item in decorated]


def clean_html(raw: str) -> str:
    """Strip HTML tags and collapse whitespace to plain text.

    Args:
        raw: HTML or plain article fragment.

    Returns:
        Unescaped plain text with tags removed and whitespace normalized.
    """
    unescaped = html.unescape(raw)
    plain = _HTML_TAG_RE.sub(" ", unescaped)
    return " ".join(plain.split())


def truncate_body(text: str, *, max_chars: int = BODY_MAX_CHARS) -> str:
    """Cap cleaned body length for eval state.

    Args:
        text: Cleaned body text.
        max_chars: Maximum character count before ellipsis truncation.

    Returns:
        ``text`` when within the cap, else a suffix-truncated string ending
        with ``...``.
    """
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


@dataclass(frozen=True, slots=True)
class HyperpartisanExample:
    """One byarticle row with cleaned title/body and partition split.

    Attributes:
        article_id (str): Stable SemEval article identifier.
        title (str): Article title (stripped).
        body (str): HTML-cleaned, length-capped body text.
        hyperpartisan (bool): Article-level partisan label.
        partition (SplitName): ``train`` or ``holdout`` after seeded split.
        split (str): Corpus split name (always byarticle train pool).
        source (str): Corpus id for eval manifests.

    Examples:
        Build one row:

        ```python
        from typevet_evals.datasets.hyperpartisan import HyperpartisanExample

        HyperpartisanExample(
            article_id="hp-smoke-001",
            title="Headline",
            body="Plain body text.",
            hyperpartisan=False,
            partition="train",
        )
        ```
    """

    article_id: str
    title: str
    body: str
    hyperpartisan: bool
    partition: SplitName
    split: str = SPLIT_BYARTICLE
    source: str = SOURCE


def hyperpartisan_state(title: str, body: str) -> dict[str, str]:
    """Build object ``state`` with ``title`` and ``body`` only.

    Args:
        title: Article title for task state.
        body: Cleaned body for task state.

    Returns:
        Mapping with ``title`` and ``body`` keys only.
    """
    return {TITLE_COLUMN: title, BODY_COLUMN: body}


def validate_task_state(state: Mapping[str, Any]) -> None:
    """Reject leakage keys in exported task ``state``.

    Args:
        state: Task state mapping.

    Raises:
        ValueError: When a banned key is present or keys are not exactly
            ``title`` and ``body``.
    """
    banned = BANNED_STATE_KEYS.intersection(state.keys())
    if banned:
        msg = f"Hyperpartisan state must not include leakage keys: {sorted(banned)!r}"
        raise ValueError(msg)
    expected = {TITLE_COLUMN, BODY_COLUMN}
    if set(state.keys()) != expected:
        msg = (
            f"Hyperpartisan state keys must be exactly {sorted(expected)!r}; "
            f"got {sorted(state.keys())!r}"
        )
        raise ValueError(msg)


def map_row(
    article_id: str,
    title: str,
    body_html: str,
    hyperpartisan: bool,
    *,
    partition: SplitName,
) -> HyperpartisanExample:
    """Map one byarticle row after HTML cleanup.

    Args:
        article_id: SemEval article id.
        title: Raw title text.
        body_html: Raw HTML body.
        hyperpartisan: Article-level label.
        partition: Train or holdout partition assignment.

    Returns:
        Example with cleaned title and body.
    """
    body = truncate_body(clean_html(body_html))
    return HyperpartisanExample(
        article_id=article_id,
        title=title.strip(),
        body=body,
        hyperpartisan=hyperpartisan,
        partition=partition,
    )


def _reject_bypublisher_row(row: Mapping[str, Any], line_number: int) -> None:
    split_value = row.get("split") or row.get("source_split")
    if split_value == EXCLUDED_SPLIT:
        msg = (
            f"Hyperpartisan JSONL line {line_number} uses excluded "
            f"{EXCLUDED_SPLIT!r} split; byarticle only"
        )
        raise ValueError(msg)


def iter_byarticle_rows(jsonl_text: str) -> Iterator[tuple[str, str, str, bool]]:
    """Yield ``(article_id, title, body, hyperpartisan)`` from byarticle JSONL.

    Args:
        jsonl_text: UTF-8 JSONL with article id, title, body, and label fields.

    Yields:
        Parsed row tuples in file order (blank lines skipped).

    Raises:
        ValueError: When a line is invalid JSON, uses excluded splits, or
            omits required fields.
    """
    for line_number, line in enumerate(jsonl_text.splitlines(), 1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            row = json.loads(stripped)
        except json.JSONDecodeError as exc:
            msg = f"Hyperpartisan JSONL line {line_number} is not valid JSON"
            raise ValueError(msg) from exc
        _reject_bypublisher_row(row, line_number)
        for key in (ARTICLE_ID_COLUMN, TITLE_COLUMN, BODY_COLUMN, LABEL_COLUMN):
            if key not in row:
                msg = (
                    f"Hyperpartisan JSONL line {line_number} needs "
                    f"{ARTICLE_ID_COLUMN!r}, {TITLE_COLUMN!r}, {BODY_COLUMN!r}, "
                    f"and {LABEL_COLUMN!r}"
                )
                raise ValueError(msg)
        yield (
            str(row[ARTICLE_ID_COLUMN]),
            str(row[TITLE_COLUMN]),
            str(row[BODY_COLUMN]),
            bool(row[LABEL_COLUMN]),
        )


def map_examples(
    rows: Iterable[tuple[str, str, str, bool]],
    *,
    partition: SplitName,
) -> list[HyperpartisanExample]:
    """Map many JSONL rows to examples for one partition.

    Args:
        rows: ``(article_id, title, body, hyperpartisan)`` tuples.
        partition: Partition tag for every mapped example.

    Returns:
        Mapped examples in input order.
    """
    return [
        map_row(aid, title, body, flag, partition=partition)
        for aid, title, body, flag in rows
    ]


def stratified_holdout_ids(
    rows: Sequence[tuple[str, str, str, bool]],
    *,
    holdout_fraction: float = HOLDOUT_FRACTION,
    seed: int = DEFAULT_HOLDOUT_SEED,
) -> frozenset[str]:
    """Return article ids assigned to the stratified holdout partition.

    Args:
        rows: Parsed byarticle tuples with labels for stratification.
        holdout_fraction: Fraction of each label class to hold out.
        seed: Seed for deterministic hash-based ordering.

    Returns:
        Article ids selected for the holdout partition.

    Raises:
        ValueError: When ``holdout_fraction`` is not strictly between 0 and 1.
    """
    if not 0.0 < holdout_fraction < 1.0:
        msg = f"holdout_fraction must be between 0 and 1; got {holdout_fraction!r}"
        raise ValueError(msg)
    by_label: dict[bool, list[str]] = {True: [], False: []}
    for article_id, _, _, flag in rows:
        by_label[flag].append(article_id)
    holdout: set[str] = set()
    for label_flag, ids in by_label.items():
        ordered = _seeded_order(ids, seed + int(label_flag))
        count = max(1, int(len(ordered) * holdout_fraction))
        holdout.update(ordered[:count])
    return frozenset(holdout)


def partition_rows(
    jsonl_text: str,
    *,
    holdout_fraction: float = HOLDOUT_FRACTION,
    seed: int = DEFAULT_HOLDOUT_SEED,
) -> tuple[list[HyperpartisanExample], list[HyperpartisanExample]]:
    """Split byarticle rows into train and stratified holdout pools.

    Args:
        jsonl_text: Byarticle training JSONL text.
        holdout_fraction: Fraction of each label class to hold out.
        seed: Seed for deterministic holdout selection.

    Returns:
        ``(train_examples, holdout_examples)`` with disjoint article ids.
    """
    parsed = list(iter_byarticle_rows(jsonl_text))
    holdout_ids = stratified_holdout_ids(
        parsed,
        holdout_fraction=holdout_fraction,
        seed=seed,
    )
    train: list[HyperpartisanExample] = []
    holdout: list[HyperpartisanExample] = []
    for article_id, title, body, flag in parsed:
        partition: SplitName = "holdout" if article_id in holdout_ids else "train"
        example = map_row(
            article_id,
            title,
            body,
            flag,
            partition=partition,
        )
        if partition == "holdout":
            holdout.append(example)
        else:
            train.append(example)
    return train, holdout


def holdout_manifest() -> dict[str, Any]:
    """Return holdout manifest metadata for eval inventories.

    Returns:
        Deep-copied holdout manifest constants (seed, fraction, license).
    """
    return json.loads(json.dumps(HOLDOUT_MANIFEST))


def task_id(example: HyperpartisanExample) -> str:
    """Stable id ``hyperpartisan-{partition}-{article_id}``.

    Args:
        example: Mapped byarticle row.

    Returns:
        JevBench-style task id string.
    """
    return f"{FAMILY}-{example.partition}-{example.article_id}"


def questions_payload() -> list[dict[str, Any]]:
    """JevBench-shaped Noul question entry for ``hyperpartisan``.

    Returns:
        Single-element list describing the primary boolean Noul.
    """
    return [
        {
            "name": PRIMARY_NOUL_NAME,
            "syntax": "Noul",
            "instructions": HYPERPARTISAN_NOUL_SCHEMA["properties"][PRIMARY_NOUL_NAME][
                "instructions"
            ],
            "return_probabilities": True,
        }
    ]


def export_task(example: HyperpartisanExample) -> dict[str, Any]:
    """Export one example as a JevBench-shaped complementary task.

    Args:
        example: Mapped byarticle row.

    Returns:
        Task dict with ``id``, ``family``, ``split``, ``provenance``,
        ``expected``, ``state``, and ``questions``.

    Raises:
        ValueError: When ``state`` would include banned keys.
    """
    state = hyperpartisan_state(example.title, example.body)
    validate_task_state(state)
    return {
        "id": task_id(example),
        "family": FAMILY,
        "split": example.partition,
        "provenance": {
            "dataset_id": DATASET_ID,
            "license": DATASET_LICENSE,
            "split": example.split,
            "partition": example.partition,
            "article_id": example.article_id,
            "source": example.source,
        },
        "expected": {PRIMARY_NOUL_NAME: example.hyperpartisan},
        "state": state,
        "questions": questions_payload(),
    }


def export_tasks(examples: Sequence[HyperpartisanExample]) -> list[dict[str, Any]]:
    """Export many examples as JevBench-shaped tasks.

    Args:
        examples: Mapped byarticle rows.

    Returns:
        Task dicts in input order.
    """
    return [export_task(example) for example in examples]


def load_byarticle_pool(
    *,
    jsonl_text: str,
    partition: PartitionName = "all",
    holdout_fraction: float = HOLDOUT_FRACTION,
    seed: int = DEFAULT_HOLDOUT_SEED,
) -> list[HyperpartisanExample]:
    """Load byarticle rows, optionally filtered to train or holdout.

    Args:
        jsonl_text: Byarticle training JSONL text.
        partition: ``train``, ``holdout``, or ``all`` (concatenated pools).
        holdout_fraction: Fraction of each label class to hold out.
        seed: Seed for deterministic holdout selection.

    Returns:
        Examples for the requested partition after stratified split.
    """
    train, holdout = partition_rows(
        jsonl_text,
        holdout_fraction=holdout_fraction,
        seed=seed,
    )
    if partition == "train":
        return train
    if partition == "holdout":
        return holdout
    return train + holdout


def load_holdout_split(
    *,
    jsonl_text: str,
    holdout_fraction: float = HOLDOUT_FRACTION,
    seed: int = DEFAULT_HOLDOUT_SEED,
) -> list[HyperpartisanExample]:
    """Load the stratified holdout partition (primary eval split).

    Args:
        jsonl_text: Byarticle training JSONL text.
        holdout_fraction: Fraction of each label class to hold out.
        seed: Seed for deterministic holdout selection.

    Returns:
        Holdout examples only.
    """
    return load_byarticle_pool(
        jsonl_text=jsonl_text,
        partition="holdout",
        holdout_fraction=holdout_fraction,
        seed=seed,
    )


def load_train_split(
    *,
    jsonl_text: str,
    holdout_fraction: float = HOLDOUT_FRACTION,
    seed: int = DEFAULT_HOLDOUT_SEED,
) -> list[HyperpartisanExample]:
    """Load the remaining byarticle train pool after holdout removal.

    Args:
        jsonl_text: Byarticle training JSONL text.
        holdout_fraction: Fraction of each label class to hold out.
        seed: Seed for deterministic holdout selection.

    Returns:
        Train examples after holdout removal.
    """
    return load_byarticle_pool(
        jsonl_text=jsonl_text,
        partition="train",
        holdout_fraction=holdout_fraction,
        seed=seed,
    )
