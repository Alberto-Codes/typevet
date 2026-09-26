"""Map CLINC ``plus`` JSONL rows to domain-sharded [ClincExample][] records.

Examples:
    Map vendored micro JSONL for the banking shard (no Hub in CI):

    ```python
    from pathlib import Path

    from typevet.eval_clinc_rows import iter_plus_rows, map_examples

    jsonl = Path("tests/fixtures/clinc/plus_banking_micro.jsonl").read_text()
    rows = map_examples(iter_plus_rows(jsonl), domain="banking")
    assert len(rows) == 15
    ```

See Also:
    - [typevet.eval_clinc_shard][]: domain catalog and schema fixtures
    - [typevet.eval_clinc][]: ``load_plus_split`` entry point
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from typevet.eval_clinc_shard import (
    BANNED_STATE_KEYS,
    CONFIG,
    OOS_INTENT,
    SOURCE,
    SPLIT,
    choice_labels_for_domain,
    plus_intent_names,
)


@dataclass(frozen=True, slots=True)
class ClincExample:
    """One CLINC ``plus`` row after domain filtering and optional OOS Noul gold.

    Attributes:
        row_id (str): Stable row identifier within a loaded split slice.
        state (dict[str, Any]): Eval state (``text`` only; no label leakage).
        domain (str): Selected domain shard key.
        domain_intents (tuple[str, ...]): Fifteen in-domain intent slugs for ``domain``.
        choice_label (str | None): In-domain intent slug gold, or ``None`` for OOS rows.
        noul_label (str | None): ``yes`` / ``no`` when ``include_oos``; else ``None``.
        split (str): Hub split name (``train`` for ``plus``).
        source (str): Corpus id ``clinc/clinc_oos``.
        config (str): Hub config name (``plus``).

    Examples:
        Build one in-domain row:

        ```python
        from typevet.eval_clinc_rows import ClincExample, build_state
        from typevet.eval_clinc_shard import DEFAULT_DOMAIN, choice_labels_for_domain

        labels = choice_labels_for_domain(DEFAULT_DOMAIN)
        ClincExample(
            row_id="0",
            state=build_state("check balance"),
            domain=DEFAULT_DOMAIN,
            domain_intents=labels,
            choice_label=labels[0],
            noul_label=None,
        )
        ```
    """

    row_id: str
    state: dict[str, Any]
    domain: str
    domain_intents: tuple[str, ...]
    choice_label: str | None
    noul_label: str | None
    split: str = SPLIT
    source: str = SOURCE
    config: str = CONFIG


def assert_state_keys_allowed(state: Mapping[str, Any]) -> None:
    """Reject eval state keys that would leak labels or shard metadata.

    Args:
        state: Candidate eval state mapping.

    Raises:
        ValueError: When a banned key is present.
    """
    banned = BANNED_STATE_KEYS.intersection(state.keys())
    if banned:
        names = ", ".join(sorted(banned))
        raise ValueError(f"CLINC state must not include banned keys: {names}")


def build_state(text: str) -> dict[str, Any]:
    """Build the v1 CLINC eval state containing only utterance text.

    Args:
        text: User utterance string.

    Returns:
        ``{"text": text}`` after banned-key validation.
    """
    state: dict[str, Any] = {"text": text}
    assert_state_keys_allowed(state)
    return state


def intent_slug_from_record(
    record: Mapping[str, Any],
    intent_names: Sequence[str],
) -> str:
    """Resolve a JSONL ``intent`` field to a slug string.

    Args:
        record: One CLINC JSON object with ``intent`` slug or Hub index.
        intent_names: Global ``plus`` slug list for index lookup.

    Returns:
        Normalized intent slug.

    Raises:
        ValueError: Intent index out of range.
        TypeError: ``intent`` is neither ``str`` nor ``int``.
    """
    raw = record["intent"]
    if isinstance(raw, str):
        return raw.strip()
    if isinstance(raw, int):
        try:
            return intent_names[raw]
        except IndexError as exc:
            raise ValueError(f"CLINC intent index {raw} out of range") from exc
    raise TypeError(f"CLINC intent must be slug or index, got {type(raw).__name__}")


def map_row(
    record: Mapping[str, Any],
    *,
    domain: str,
    include_oos: bool = False,
    intent_names: Sequence[str] | None = None,
    row_id: str | None = None,
) -> ClincExample | None:
    """Map one JSONL record to a [ClincExample][] or drop cross-domain rows.

    Args:
        record: JSON object with ``text`` and ``intent``.
        domain: Domain shard to keep.
        include_oos: When true, retain global ``oos`` rows with Noul gold.
        intent_names: Optional slug list; defaults to ``plus_intent_names()``.
        row_id: Optional stable id; otherwise derived from utterance hash.

    Returns:
        Mapped example, or ``None`` when the row is out of shard scope.

    Raises:
        ValueError: Missing required fields.
    """
    if "text" not in record or "intent" not in record:
        raise ValueError("CLINC row missing required fields: text, intent")
    names = intent_names if intent_names is not None else plus_intent_names()
    slug = intent_slug_from_record(record, names)
    allowed = choice_labels_for_domain(domain)
    allowed_set = frozenset(allowed)
    text = str(record["text"])
    rid = (
        row_id if row_id is not None else hashlib.sha256(text.encode()).hexdigest()[:16]
    )
    if slug == OOS_INTENT:
        if not include_oos:
            return None
        return ClincExample(
            row_id=rid,
            state=build_state(text),
            domain=domain,
            domain_intents=allowed,
            choice_label=None,
            noul_label="no",
        )
    if slug not in allowed_set:
        return None
    return ClincExample(
        row_id=rid,
        state=build_state(text),
        domain=domain,
        domain_intents=allowed,
        choice_label=slug,
        noul_label="yes" if include_oos else None,
    )


def iter_plus_rows(jsonl_text: str) -> Iterator[Mapping[str, Any]]:
    """Parse CLINC ``plus`` JSONL text into mapping records.

    Args:
        jsonl_text: Full split body with one JSON object per line.

    Yields:
        Parsed row objects.

    Raises:
        ValueError: Invalid JSON on a non-empty line.
        TypeError: Parsed value is not a JSON object.
    """
    for line_number, line in enumerate(jsonl_text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"CLINC JSONL line {line_number} is not valid JSON"
            ) from exc
        if not isinstance(payload, Mapping):
            raise TypeError(f"CLINC JSONL line {line_number} must be a JSON object")
        yield payload


def map_examples(
    records: Iterable[Mapping[str, Any]],
    *,
    domain: str,
    include_oos: bool = False,
    intent_names: Sequence[str] | None = None,
) -> list[ClincExample]:
    """Map an iterable of JSONL records, preserving enumeration order in ``row_id``.

    Args:
        records: Parsed CLINC rows.
        domain: Domain shard filter.
        include_oos: Pass-through to ``map_row``.
        intent_names: Optional global slug list.

    Returns:
        In-shard examples only (cross-domain rows dropped).
    """
    mapped: list[ClincExample] = []
    for index, record in enumerate(records):
        example = map_row(
            record,
            domain=domain,
            include_oos=include_oos,
            intent_names=intent_names,
            row_id=str(index),
        )
        if example is not None:
            mapped.append(example)
    return mapped


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


def balanced_sample(
    examples: list[ClincExample],
    domain: str,
    limit: int | None,
    seed: int,
) -> list[ClincExample]:
    """Return a per-intent balanced slice with deterministic ordering.

    Args:
        examples: Pre-filtered in-domain (and optional OOS) examples.
        domain: Domain whose 15 labels define buckets.
        limit: Optional cap applied evenly across intent buckets.
        seed: Stable shuffle seed.

    Returns:
        Balanced subset; OOS rows are appended when present.
    """
    labels = choice_labels_for_domain(domain)
    buckets: dict[str, list[ClincExample]] = {label: [] for label in labels}
    for example in examples:
        if example.choice_label is not None:
            buckets[example.choice_label].append(example)
    per_label_lists = [
        _seeded_order(buckets[label], seed + index)
        for index, label in enumerate(labels)
    ]
    min_count = min(len(bucket) for bucket in per_label_lists)
    if limit is not None:
        min_count = min(min_count, max(0, limit // len(labels)))
    sample = [row for bucket in per_label_lists for row in bucket[:min_count]]
    oos = [ex for ex in examples if ex.noul_label == "no"]
    if oos:
        sample.extend(_seeded_order(oos, seed + len(labels)))
    return _seeded_order(sample, seed + len(labels) + 1)
