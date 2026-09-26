"""go_emotions ``simplified`` loader with strict exactly-one Choice (24-enum).

google-research-datasets/go_emotions (Apache 2.0) is multi-label upstream.
typevet applies the v1 policy ([#67](https://github.com/Alberto-Codes/typevet/issues/67)):
neutral co-label stripping, **strict exactly-one** gold (drop ambiguous rows),
then drops the four pruned labels. Eval ``state`` is ``{"text": ...}`` only.

Examples:
    Load a contract-sized subset from vendored JSONL (no Hub in CI):

    ```python
    from pathlib import Path

    from typevet.evaluation.datasets.go_emotions import load_train_split

    jsonl = Path("tests/fixtures/go_emotions/simplified_train_subset.jsonl").read_text()
    rows = load_train_split(jsonl_text=jsonl, limit=12)
    assert rows[0].choice_label in CHOICE_LABELS
    ```

See Also:
    - docs/reference/go-emotions-conversion-and-prune.md: conversion and prune
    - [typevet.evaluation.datasets.pubmedqa][]: PubMedQA Choice loader (separate corpus)
    - [typevet.evaluation.datasets.banking77][]: Banking77 fraud-proxy loader (separate corpus)
    - [typevet.domain.decision_compile][]: compile Choice schemas for fixtures
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import httpx

from typevet.evaluation.datasets.go_emotions_download import (
    UPSTREAM_LABELS,
    download_train_jsonl,
)

SOURCE: Final[str] = "go_emotions"
CONFIG: Final[str] = "simplified"
SPLIT: Final[str] = "train"
NEUTRAL_LABEL: Final[str] = "neutral"
PRIMARY_CHOICE_NAME: Final[str] = "emotion"
PRUNED_LABELS: Final[frozenset[str]] = frozenset(
    {"grief", "nervousness", "pride", "relief"}
)
CHOICE_LABELS: Final[tuple[str, ...]] = tuple(
    name for name in UPSTREAM_LABELS if name not in PRUNED_LABELS
)
BANNED_STATE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "labels",
        "label",
        "expected",
        "ground_truth",
        "answer_key",
        "emotion",
    }
)

EMOTION_CHOICE_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        PRIMARY_CHOICE_NAME: {
            "type": "string",
            "enum": list(CHOICE_LABELS),
            "instructions": (
                "Given the user message, which single emotion label best applies?"
            ),
        }
    },
    "required": [PRIMARY_CHOICE_NAME],
    "additionalProperties": False,
}


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


@dataclass(frozen=True, slots=True)
class GoEmotionsExample:
    """One converted row with eval ``state`` and gold Choice label.

    Attributes:
        row_id (str): Hub ``id`` for the comment.
        state (dict[str, Any]): ``{"text": ...}`` only.
        choice_label (str): Gold emotion in the 24-enum after strict conversion.
        split (str): HF split name (``train`` for v1).
        source (str): Corpus id for eval manifests.
        config (str): HF config (``simplified``).

    Examples:
        Build one row:

        ```python
        from typevet.evaluation.datasets.go_emotions import (
            GoEmotionsExample,
            build_state,
        )

        state = build_state("hello")
        GoEmotionsExample(row_id="1", state=state, choice_label="joy")
        ```
    """

    row_id: str
    state: dict[str, Any]
    choice_label: str
    split: str = SPLIT
    source: str = SOURCE
    config: str = CONFIG


def assert_state_keys_allowed(state: Mapping[str, Any]) -> None:
    """Reject leakage keys in eval ``state``.

    Args:
        state: Candidate state object.

    Raises:
        ValueError: When a banned key is present at the top level.
    """
    banned = BANNED_STATE_KEYS.intersection(state.keys())
    if banned:
        names = ", ".join(sorted(banned))
        msg = f"go_emotions state must not include banned keys: {names}"
        raise ValueError(msg)


def build_state(text: str) -> dict[str, Any]:
    """Build v1 ``state`` from Hub ``text``.

    Args:
        text: Comment text from the Hub row.

    Returns:
        ``{"text": text}`` with leakage keys validated.
    """
    state: dict[str, Any] = {"text": text}
    assert_state_keys_allowed(state)
    return state


def labels_after_neutral_rule(raw_labels: Sequence[str]) -> list[str]:
    """Apply #67 neutral co-label rule to upstream label names.

    Args:
        raw_labels: Upstream emotion names (deduplicated in order).

    Returns:
        Labels after stripping ``neutral`` when co-labeled with other emotions.
    """
    unique = list(dict.fromkeys(str(label) for label in raw_labels))
    if NEUTRAL_LABEL in unique and len(unique) > 1:
        return [label for label in unique if label != NEUTRAL_LABEL]
    return unique


def strict_gold_label(raw_labels: Sequence[str]) -> str | None:
    """Return v1 gold emotion name, or ``None`` when the row must drop.

    Args:
        raw_labels: Upstream label names before neutral rule and pruning.

    Returns:
        Single emotion in ``CHOICE_LABELS``, or ``None`` when ambiguous or pruned.
    """
    working = labels_after_neutral_rule(raw_labels)
    if len(working) != 1:
        return None
    gold = working[0]
    if gold in PRUNED_LABELS or gold not in CHOICE_LABELS:
        return None
    return gold


def parse_upstream_labels(raw: object) -> list[str]:
    """Parse HF ``labels`` as name list or ClassLabel indices.

    Args:
        raw: Hub ``labels`` field (strings or integer indices).

    Returns:
        Upstream emotion name list.

    Raises:
        TypeError: When ``raw`` is not a list.
    """
    if not isinstance(raw, list):
        msg = "go_emotions labels must be a list"
        raise TypeError(msg)
    if not raw:
        return []
    first = raw[0]
    if isinstance(first, int):
        return [UPSTREAM_LABELS[index] for index in raw]
    return [str(label) for label in raw]


def try_map_row(record: Mapping[str, Any]) -> GoEmotionsExample | None:
    """Map one Hub row when strict conversion keeps it; else ``None``.

    Args:
        record: Object with ``id``, ``text``, and ``labels``.

    Returns:
        Example when strict conversion succeeds, else ``None``.

    Raises:
        ValueError: When required fields are missing.
    """
    missing = [key for key in ("id", "text", "labels") if key not in record]
    if missing:
        names = ", ".join(missing)
        msg = f"go_emotions row missing required fields: {names}"
        raise ValueError(msg)
    gold = strict_gold_label(parse_upstream_labels(record["labels"]))
    if gold is None:
        return None
    state = build_state(str(record["text"]))
    return GoEmotionsExample(row_id=str(record["id"]), state=state, choice_label=gold)


def map_row(record: Mapping[str, Any]) -> GoEmotionsExample:
    """Map one Hub row that must survive strict conversion.

    Args:
        record: Object with ``id``, ``text``, and ``labels``.

    Returns:
        Example with normalized state and choice label.

    Raises:
        ValueError: When required fields are missing or the row is dropped.
    """
    example = try_map_row(record)
    if example is None:
        msg = "go_emotions row was dropped by strict exactly-one conversion"
        raise ValueError(msg)
    return example


def iter_train_rows(jsonl_text: str) -> Iterator[Mapping[str, Any]]:
    """Yield parsed JSON objects from simplified train JSONL text.

    Args:
        jsonl_text: UTF-8 JSONL (one HF row per line).

    Yields:
        Parsed row objects in file order.

    Raises:
        TypeError: When a line parses to a non-object JSON value.
        ValueError: When a line is not valid JSON.
    """
    for line_number, line in enumerate(jsonl_text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            msg = f"go_emotions JSONL line {line_number} is not valid JSON"
            raise ValueError(msg) from exc
        if not isinstance(payload, Mapping):
            msg = f"go_emotions JSONL line {line_number} must be a JSON object"
            raise TypeError(msg)
        yield payload


def map_examples(records: Iterable[Mapping[str, Any]]) -> list[GoEmotionsExample]:
    """Map kept rows in input order (drops strict/prune failures).

    Args:
        records: Parsed simplified train objects.

    Returns:
        Kept examples in input order.
    """
    kept: list[GoEmotionsExample] = []
    for record in records:
        example = try_map_row(record)
        if example is not None:
            kept.append(example)
    return kept


def balanced_sample(
    examples: list[GoEmotionsExample],
    limit: int | None,
    seed: int,
) -> list[GoEmotionsExample]:
    """Return equal counts per Choice label when ``limit`` allows.

    Args:
        examples: Full mapped train split.
        limit: Total cap, or ``None`` for the smallest class count times 24.
        seed: Seed for deterministic hash-based ordering (not ``random``).

    Returns:
        Balanced sample in seeded order.
    """
    buckets: dict[str, list[GoEmotionsExample]] = {label: [] for label in CHOICE_LABELS}
    for example in examples:
        buckets[example.choice_label].append(example)
    per_label_lists = [
        _seeded_order(buckets[label], seed + index)
        for index, label in enumerate(CHOICE_LABELS)
    ]
    min_count = min(len(bucket) for bucket in per_label_lists)
    if limit is not None:
        per_class = max(0, limit // len(CHOICE_LABELS))
        min_count = min(min_count, per_class)
    sample = [row for bucket in per_label_lists for row in bucket[:min_count]]
    return _seeded_order(sample, seed + len(CHOICE_LABELS))


def load_train_split(
    *,
    limit: int | None = None,
    seed: int = 0,
    balanced: bool = False,
    jsonl_text: str | None = None,
    client: httpx.Client | None = None,
) -> list[GoEmotionsExample]:
    """Load ``simplified`` train with strict conversion and optional sampling.

    Args:
        limit: Row cap. With ``balanced=True``, total rows (roughly equal per
            Choice label). Without balance, first ``limit`` rows in file order.
        seed: Seed for deterministic ordering when ``balanced=True``.
        balanced: When ``True``, balance across the 24 emotion labels.
        jsonl_text: Pre-fetched JSONL (CI fixtures). When ``None``, downloads via
            :func:`typevet.evaluation.datasets.go_emotions_download.download_train_jsonl`.
        client: Optional HTTP client when downloading.

    Returns:
        Mapped simplified train examples.
    """
    text = jsonl_text if jsonl_text is not None else download_train_jsonl(client=client)
    examples = map_examples(iter_train_rows(text))
    if balanced:
        return balanced_sample(examples, limit, seed)
    if limit is not None:
        return examples[:limit]
    return examples
