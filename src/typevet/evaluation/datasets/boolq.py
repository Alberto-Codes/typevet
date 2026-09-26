"""BoolQ validation-split loader and JevBench-shaped Noul task export.

google/boolq (CC BY-SA 3.0) pairs a passage with a yes/no question. typevet
maps the Hub ``answer`` boolean to Noul labels ``no`` and ``yes`` (not Python
``True``/``False`` in ``expected``). v1 uses the **validation** split only.
In-repo fixtures are tiny smoke bundles (≤24 rows, #76); bulk passage text
streams from the public Hugging Face datasets server at eval time.

Examples:
    Load the vendored smoke JSONL (no Hub in CI):

    ```python
    from pathlib import Path

    from typevet.evaluation.datasets.boolq import export_tasks, load_tier_a

    jsonl = Path("tests/fixtures/boolq/validation_smoke.jsonl").read_text()
    rows = load_tier_a(jsonl_text=jsonl, seed=0)
    tasks = export_tasks(rows)
    assert tasks[0]["expected"]["answer"] in {"no", "yes"}
    ```

See Also:
    - [typevet.evaluation.datasets.banking77][]: Banking77 fraud-proxy loader (separate corpus)
    - [typevet.evaluation.datasets.difraud][]: DIFrauD scam/legit loader (separate corpus)
    - [typevet.evaluation.datasets.civil_comments][]: Civil Comments toxicity loader
    - [typevet.evaluation.datasets.pubmedqa][]: PubMedQA labeled Choice loader (separate corpus)
    - [typevet.evaluation.datasets.boolq_download][]: HF datasets-server validation JSONL stream
    - [typevet.domain.decision_compile][]: compile Noul schemas for fixtures
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal

import httpx

from typevet.evaluation.datasets.boolq_download import download_validation_jsonl

TierName = Literal["A", "B"]

PASSAGE_MARKER: Final[str] = "--- passage ---"
QUESTION_MARKER: Final[str] = "--- question ---"


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


SOURCE: Final[str] = "boolq"
FAMILY: Final[str] = "boolq"
SPLIT: Final[str] = "validation"
PASSAGE_COLUMN: Final[str] = "passage"
QUESTION_COLUMN: Final[str] = "question"
ANSWER_COLUMN: Final[str] = "answer"
IDX_COLUMN: Final[str] = "idx"
DEFAULT_TIER_SEED: Final[int] = 0
TIER_A_LIMIT: Final[int] = 24
TIER_B_LIMIT: Final[int] = 256
CONTRACT_ROW_MIN: Final[int] = 12
CONTRACT_ROW_MAX: Final[int] = 24
PRIMARY_NOUL_NAME: Final[str] = "answer"
BOOLQ_ANSWER_NOUL_SCHEMA_VERSION: Final[str] = "1"
DATASET_ID: Final[str] = "google/boolq"
DATASET_CONFIG: Final[str] = "default"
DATASET_LICENSE: Final[str] = "CC BY-SA 3.0"

BANNED_STATE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "answer",
        "expected",
        "label",
        "ground_truth",
        "answer_key",
    }
)

NOUL_LABELS: Final[tuple[str, str]] = ("no", "yes")

BOOLQ_ANSWER_NOUL_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        PRIMARY_NOUL_NAME: {
            "type": "string",
            "enum": list(NOUL_LABELS),
            "instructions": (
                "Based on the passage, is the answer to the question yes?"
            ),
            "return_probabilities": True,
        }
    },
    "required": [PRIMARY_NOUL_NAME],
    "additionalProperties": False,
}

TIER_MANIFEST: Final[dict[str, Any]] = {
    "manifest_version": 1,
    "loader_issue": 77,
    "research_issue": 64,
    "design_issue": 75,
    "judgment_issue": 76,
    "dataset_id": DATASET_ID,
    "split": SPLIT,
    "license": DATASET_LICENSE,
    "default_seed": DEFAULT_TIER_SEED,
    "contract_row_bounds": {"min": CONTRACT_ROW_MIN, "max": CONTRACT_ROW_MAX},
    "tiers": {
        "A": {
            "total_limit": TIER_A_LIMIT,
            "balanced": True,
            "per_class": TIER_A_LIMIT // 2,
            "role": "smoke_contract",
        },
        "B": {
            "total_limit": TIER_B_LIMIT,
            "balanced": True,
            "per_class": TIER_B_LIMIT // 2,
            "role": "regression",
        },
    },
}


@dataclass(frozen=True, slots=True)
class BoolQExample:
    """One BoolQ validation row with yes/no Noul label.

    Attributes:
        passage (str): Supporting passage text (CC BY-SA when from Hub).
        question (str): Yes/no question about the passage.
        label (str): ``yes`` or ``no`` from the Hub ``answer`` boolean.
        idx (int): Stable row index from the card (used in task ids).
        split (str): Dataset split name (always ``validation`` for this loader).
        source (str): Corpus id for eval manifests.

    Examples:
        Build one row:

        ```python
        from typevet.evaluation.datasets.boolq import BoolQExample

        BoolQExample(
            passage="The sky is blue.",
            question="Is the sky green?",
            label="no",
            idx=0,
        )
        ```
    """

    passage: str
    question: str
    label: str
    idx: int
    split: str = SPLIT
    source: str = SOURCE


def noul_label_for_answer(answer: bool) -> str:
    """Map the Hub boolean ``answer`` field to a Noul label.

    Args:
        answer: Hugging Face BoolQ ``answer`` value.

    Returns:
        ``yes`` when ``answer`` is true, else ``no``.
    """
    return "yes" if answer else "no"


def boolq_state(passage: str, question: str) -> dict[str, str]:
    """Build the object ``state`` for a BoolQ task (#75).

    Args:
        passage: Passage under evaluation.
        question: Question about the passage.

    Returns:
        Mapping with ``passage`` and ``question`` keys only.
    """
    return {PASSAGE_COLUMN: passage, QUESTION_COLUMN: question}


def serialize_boolq_state(passage: str, question: str) -> str:
    """Canonical string serializer for BoolQ ``state`` (#75).

    Args:
        passage: Passage under evaluation.
        question: Question about the passage.

    Returns:
        Multi-line string with fixed section markers.
    """
    return f"{PASSAGE_MARKER}\n{passage}\n{QUESTION_MARKER}\n{question}"


def validate_task_state(state: Mapping[str, Any]) -> None:
    """Reject leakage keys in exported task ``state``.

    Args:
        state: Task state mapping.

    Raises:
        ValueError: When a banned key is present.
    """
    banned = BANNED_STATE_KEYS.intersection(state.keys())
    if banned:
        msg = f"BoolQ state must not include leakage keys: {sorted(banned)!r}"
        raise ValueError(msg)
    expected = {PASSAGE_COLUMN, QUESTION_COLUMN}
    if set(state.keys()) != expected:
        msg = (
            f"BoolQ state keys must be exactly {sorted(expected)!r}; "
            f"got {sorted(state.keys())!r}"
        )
        raise ValueError(msg)


def map_row(
    passage: str,
    question: str,
    answer: bool,
    *,
    idx: int,
) -> BoolQExample:
    """Map one BoolQ row to a :class:`BoolQExample`.

    Args:
        passage: Passage text.
        question: Question text.
        answer: Hub boolean answer.
        idx: Stable dataset index for ids.

    Returns:
        Example with ``yes``/``no`` label.
    """
    return BoolQExample(
        passage=passage,
        question=question,
        label=noul_label_for_answer(answer),
        idx=idx,
    )


def iter_validation_rows(jsonl_text: str) -> Iterator[tuple[str, str, bool, int]]:
    """Yield ``(passage, question, answer, idx)`` from BoolQ validation JSONL.

    Args:
        jsonl_text: UTF-8 JSONL with ``passage``, ``question``, and ``answer``.
            ``idx`` is read when present; otherwise line order is used.

    Yields:
        Parsed row tuples in file order.

    Raises:
        ValueError: When a line is invalid or required fields are missing.
    """
    for line_number, line in enumerate(jsonl_text.splitlines(), 1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            row = json.loads(stripped)
        except json.JSONDecodeError as exc:
            msg = f"BoolQ JSONL line {line_number} is not valid JSON"
            raise ValueError(msg) from exc
        for key in (PASSAGE_COLUMN, QUESTION_COLUMN, ANSWER_COLUMN):
            if key not in row:
                msg = (
                    f"BoolQ JSONL line {line_number} needs "
                    f"{PASSAGE_COLUMN!r}, {QUESTION_COLUMN!r}, and "
                    f"{ANSWER_COLUMN!r} fields"
                )
                raise ValueError(msg)
        idx = int(row[IDX_COLUMN]) if IDX_COLUMN in row else line_number - 1
        yield (
            str(row[PASSAGE_COLUMN]),
            str(row[QUESTION_COLUMN]),
            bool(row[ANSWER_COLUMN]),
            idx,
        )


def map_examples(
    rows: Iterable[tuple[str, str, bool, int]],
) -> list[BoolQExample]:
    """Map many JSONL rows to examples.

    Args:
        rows: ``(passage, question, answer, idx)`` tuples.

    Returns:
        Mapped examples in input order.
    """
    return [map_row(p, q, a, idx=idx) for p, q, a, idx in rows]


def balanced_sample(
    examples: list[BoolQExample],
    limit: int | None,
    seed: int,
) -> list[BoolQExample]:
    """Return equal ``yes`` and ``no`` counts (tier A/B semantics).

    Args:
        examples: Full mapped validation split.
        limit: Total cap, or ``None`` for every ``yes`` row plus matching ``no``.
        seed: Seed for deterministic hash-based ordering (not ``random``).

    Returns:
        Balanced sample in seeded order.
    """
    yes_rows = _seeded_order([e for e in examples if e.label == "yes"], seed)
    no_rows = _seeded_order([e for e in examples if e.label == "no"], seed + 1)
    if limit is not None:
        yes_rows = yes_rows[: limit // 2]
    sample = yes_rows + no_rows[: len(yes_rows)]
    return _seeded_order(sample, seed + 2)


def tier_limit(tier: TierName) -> int:
    """Return the total row cap for eval tier ``A`` or ``B``.

    Args:
        tier: ``A`` (smoke contract, 24) or ``B`` (regression, 256).

    Returns:
        ``TIER_A_LIMIT`` or ``TIER_B_LIMIT``.
    """
    if tier == "A":
        return TIER_A_LIMIT
    return TIER_B_LIMIT


def tier_manifest() -> dict[str, Any]:
    """Return the seeded tier manifest constants for BoolQ eval.

    Returns:
        Deep-copied manifest metadata (limits, default seed, license).
    """
    return json.loads(json.dumps(TIER_MANIFEST))


def task_id(example: BoolQExample) -> str:
    """Stable JevBench-style id for one example.

    Args:
        example: Mapped validation row.

    Returns:
        Id string ``boolq-validation-{idx}``.
    """
    return f"{FAMILY}-{example.split}-{example.idx}"


def provenance_for(example: BoolQExample) -> dict[str, Any]:
    """Build provenance metadata for one exported task.

    Args:
        example: Mapped validation row.

    Returns:
        Provenance block including dataset id, license, split, and row index.
    """
    return {
        "dataset_id": DATASET_ID,
        "config": DATASET_CONFIG,
        "license": DATASET_LICENSE,
        "split": example.split,
        "idx": example.idx,
        "source": example.source,
    }


def questions_payload() -> list[dict[str, Any]]:
    """Return JevBench-shaped question entries for the primary Noul.

    Returns:
        Single-element list describing the ``answer`` Noul with ``no``/``yes``.
    """
    return [
        {
            "name": PRIMARY_NOUL_NAME,
            "syntax": "Noul",
            "labels": list(NOUL_LABELS),
            "instructions": BOOLQ_ANSWER_NOUL_SCHEMA["properties"][PRIMARY_NOUL_NAME][
                "instructions"
            ],
            "return_probabilities": True,
        }
    ]


def export_task(example: BoolQExample) -> dict[str, Any]:
    """Export one example as a JevBench-shaped complementary task (#75).

    Args:
        example: Mapped validation row.

    Returns:
        Task dict with ``id``, ``family``, ``split``, ``provenance``,
        ``expected``, ``state``, and ``questions``.

    Raises:
        ValueError: When ``state`` would include banned keys.
    """
    state = boolq_state(example.passage, example.question)
    validate_task_state(state)
    return {
        "id": task_id(example),
        "family": FAMILY,
        "split": example.split,
        "provenance": provenance_for(example),
        "expected": {PRIMARY_NOUL_NAME: example.label},
        "state": state,
        "questions": questions_payload(),
    }


def export_tasks(examples: Sequence[BoolQExample]) -> list[dict[str, Any]]:
    """Export many examples as JevBench-shaped tasks.

    Args:
        examples: Mapped validation rows.

    Returns:
        Task dicts in input order.
    """
    return [export_task(example) for example in examples]


def load_validation_split(
    *,
    limit: int | None = None,
    seed: int = DEFAULT_TIER_SEED,
    balanced: bool = False,
    jsonl_text: str | None = None,
    client: httpx.Client | None = None,
) -> list[BoolQExample]:
    """Load the BoolQ validation split with optional balanced sampling.

    Args:
        limit: Row cap. With ``balanced=True``, total rows (half ``yes``, half
            ``no``). Without balance, first ``limit`` rows in file order.
        seed: Seed for deterministic ordering when ``balanced=True``.
        balanced: When ``True``, apply tier-style class balance before return.
        jsonl_text: Pre-fetched JSONL (CI smoke fixtures). When ``None``,
            downloads via the public datasets-server API.
        client: Optional HTTP client when downloading.

    Returns:
        Mapped validation examples.
    """
    text = (
        jsonl_text
        if jsonl_text is not None
        else download_validation_jsonl(client=client)
    )
    examples = map_examples(iter_validation_rows(text))
    if balanced:
        return balanced_sample(examples, limit, seed)
    if limit is not None:
        return examples[:limit]
    return examples


def load_tier(
    tier: TierName,
    *,
    seed: int = DEFAULT_TIER_SEED,
    jsonl_text: str | None = None,
    client: httpx.Client | None = None,
) -> list[BoolQExample]:
    """Load a balanced tier-A or tier-B sample from the validation split.

    Args:
        tier: ``A`` (24 rows) or ``B`` (256 rows) per #64/#75.
        seed: Seed for deterministic hash-based ordering.
        jsonl_text: Pre-fetched JSONL (CI fixtures).
        client: Optional HTTP client when downloading.

    Returns:
        Balanced mapped validation examples capped at the tier limit.
    """
    return load_validation_split(
        limit=tier_limit(tier),
        seed=seed,
        balanced=True,
        jsonl_text=jsonl_text,
        client=client,
    )


def load_tier_a(
    *,
    seed: int = DEFAULT_TIER_SEED,
    jsonl_text: str | None = None,
    client: httpx.Client | None = None,
) -> list[BoolQExample]:
    """Load balanced tier A (24 rows, 12 per class).

    Args:
        seed: Seed for deterministic ordering.
        jsonl_text: Pre-fetched JSONL (CI fixtures).
        client: Optional HTTP client when downloading.

    Returns:
        Tier-A balanced sample.
    """
    return load_tier("A", seed=seed, jsonl_text=jsonl_text, client=client)


def load_tier_b(
    *,
    seed: int = DEFAULT_TIER_SEED,
    jsonl_text: str | None = None,
    client: httpx.Client | None = None,
) -> list[BoolQExample]:
    """Load balanced tier B (256 rows, 128 per class).

    Args:
        seed: Seed for deterministic ordering.
        jsonl_text: Pre-fetched JSONL (CI fixtures).
        client: Optional HTTP client when downloading.

    Returns:
        Tier-B balanced sample.
    """
    return load_tier("B", seed=seed, jsonl_text=jsonl_text, client=client)
