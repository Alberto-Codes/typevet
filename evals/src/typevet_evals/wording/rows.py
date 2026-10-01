"""Seed-agnostic rows of a wording run: a state and a gold label (#369).

A ``WordingRow`` holds the state the judge reads (a text or a mapping), the
gold label, the record id and the ``split`` it came from. The runner and the
held-out scorer read these rows, so a ``Noul`` and a ``Choice`` seed share
one path. The gold label never reaches the port.

``difraud_rows`` maps DIFrauD records to gold ``"1"`` (scam) or ``"0"``.
``pubmedqa_rows`` maps PubMedQA examples to the gold label name, and
``pubmedqa_splits`` cuts a balanced PubMedQA pool into train, validation and
held-out rows. ``gepa_rows`` gives gepa-adk's ``input`` and ``expected``
rows and the state of each ``input`` key.

This module reads no split from the network: the caller gives the records.

Attributes:
    POSITIVE_LABEL (str): The DIFrauD label whose gold is ``"1"``.
    AnyRow (TypeAliasType): A DIFrauD record or a ``WordingRow``.

Examples:
    ```python
    rows = difraud_rows(splits.validation)
    gepa, states = gepa_rows(rows)
    ```

See Also:
    - [typevet_evals.wording.runner][]: the evolution runner
    - [typevet_evals.wording.held_out][]: the held-out scorer
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from typevet_evals.datasets.difraud import DIFrauDRecord
from typevet_evals.datasets.pubmedqa import (
    CHOICE_LABELS,
    PubMedQAExample,
    balanced_sample,
)

POSITIVE_LABEL: Final[str] = "scam"


@dataclass(frozen=True, slots=True)
class WordingRow:
    """One row of a wording run: the state, the gold label and its origin.

    Attributes:
        state (str | Mapping[str, Any]): What the judge reads: a message
            text or a state mapping, such as a PubMedQA question and contexts.
        gold (str): The gold label: ``"1"`` or ``"0"`` for a ``Noul`` seed,
            a label name for a ``Choice`` seed.
        record_id (str): The record id, keyword only.
        split (str): The split the row came from, keyword only.

    Examples:
        ```python
        WordingRow({"question": "Q?"}, "maybe", record_id="pubmedqa:1", split="train")
        ```
    """

    state: str | Mapping[str, Any]
    gold: str
    record_id: str = field(default="", kw_only=True)
    split: str = field(default="", kw_only=True)


type AnyRow = DIFrauDRecord | WordingRow


def as_row(record: AnyRow) -> WordingRow:
    """Return the ``WordingRow`` of a DIFrauD record or a row.

    Args:
        record: A DIFrauD record or a ``WordingRow``.

    Returns:
        The row; a DIFrauD record has gold ``"1"`` for scam, else ``"0"``.
    """
    if isinstance(record, WordingRow):
        return record
    gold = "1" if record.example.label == POSITIVE_LABEL else "0"
    return WordingRow(
        record.example.text,
        gold,
        record_id=record.record_id,
        split=record.example.split,
    )


def difraud_rows(records: Iterable[DIFrauDRecord]) -> tuple[WordingRow, ...]:
    """Return the rows of DIFrauD records.

    Args:
        records: DIFrauD records.

    Returns:
        One row per record, with gold ``"1"`` for scam and ``"0"`` for legit.
    """
    return tuple(as_row(record) for record in records)


def pubmedqa_rows(
    examples: Iterable[PubMedQAExample], split: str
) -> tuple[WordingRow, ...]:
    """Return the rows of PubMedQA examples for one split of a wording run.

    Args:
        examples: PubMedQA examples.
        split: The split name the rows carry, such as ``train``.

    Returns:
        One row per example: its state, its label name and the id
        ``pubmedqa:<pubid>``.
    """
    return tuple(
        WordingRow(
            e.state, e.choice_label, record_id=f"pubmedqa:{e.pubid}", split=split
        )
        for e in examples
    )


@dataclass(frozen=True, slots=True)
class WordingSplits:
    """The train, validation and held-out rows of one wording run.

    Attributes:
        train (tuple[WordingRow, ...]): Rows the evolution reflects on.
        validation (tuple[WordingRow, ...]): Rows the evolution selects on.
        held_out (tuple[WordingRow, ...]): Rows of split ``test``.

    Examples:
        ```python
        splits = pubmedqa_splits(examples, train=6, validation=3, held_out=6)
        ```
    """

    train: tuple[WordingRow, ...]
    validation: tuple[WordingRow, ...]
    held_out: tuple[WordingRow, ...]


def _per_label(size: int) -> list[int]:
    """Return the rows of each label for a split of ``size`` rows.

    Args:
        size: Rows in the split.

    Returns:
        One count per label in ``CHOICE_LABELS`` order; the first labels
        take the remainder.
    """
    share, left = divmod(size, len(CHOICE_LABELS))
    return [share + (1 if i < left else 0) for i in range(len(CHOICE_LABELS))]


def pubmedqa_splits(
    examples: Sequence[PubMedQAExample],
    *,
    train: int,
    validation: int,
    held_out: int,
    seed: int = 0,
) -> WordingSplits:
    """Cut a balanced PubMedQA pool into disjoint, balanced splits.

    The pool is ``balanced_sample(examples, None, seed)``. Each label's rows
    go to the held-out split first, then validation, then train, in pool
    order, so the cut depends on the seed and the pubids only.

    Args:
        examples: The mapped ``pqa_labeled`` examples.
        train: Train rows.
        validation: Validation rows.
        held_out: Held-out rows; they carry split ``test``.
        seed: The ordering seed of the balanced pool.

    Returns:
        The three splits.

    Raises:
        ValueError: If the balanced pool has too few rows per label.
    """
    pool = balanced_sample(list(examples), None, seed)
    by_label = {
        label: [e for e in pool if e.choice_label == label] for label in CHOICE_LABELS
    }
    sizes = (("test", held_out), ("validation", validation), ("train", train))
    need = sum(_per_label(size)[0] for _, size in sizes)
    have = min(len(rows) for rows in by_label.values())
    if need > have:
        msg = f"the balanced PubMedQA pool has {have} rows per label; the splits need {need}"
        raise ValueError(msg)
    taken = dict.fromkeys(CHOICE_LABELS, 0)
    cut: dict[str, list[PubMedQAExample]] = {}
    for split, size in sizes:
        cut[split] = []
        for label, count in zip(CHOICE_LABELS, _per_label(size), strict=True):
            cut[split] += by_label[label][taken[label] : taken[label] + count]
            taken[label] += count
    return WordingSplits(
        train=pubmedqa_rows(cut["train"], "train"),
        validation=pubmedqa_rows(cut["validation"], "validation"),
        held_out=pubmedqa_rows(cut["test"], "test"),
    )


def state_key(state: str | Mapping[str, Any]) -> str:
    """Return the gepa-adk ``input`` text of a state.

    Args:
        state: A message text or a state mapping.

    Returns:
        The text itself, or the mapping as sorted JSON.
    """
    if isinstance(state, str):
        return state
    return json.dumps(state, sort_keys=True, ensure_ascii=False)


def gepa_rows(
    rows: Iterable[WordingRow],
) -> tuple[list[dict[str, Any]], dict[str, Mapping[str, Any]]]:
    """Return gepa-adk rows and the state mapping behind each ``input`` key.

    Args:
        rows: The rows.

    Returns:
        One ``{"input": key, "expected": gold}`` row per row, and the state
        of each key whose state is a mapping.
    """
    gepa: list[dict[str, Any]] = []
    states: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        key = state_key(row.state)
        if not isinstance(row.state, str):
            states[key] = row.state
        gepa.append({"input": key, "expected": row.gold})
    return gepa, states
