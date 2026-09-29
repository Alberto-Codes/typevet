"""PubMedQA ``pqa_labeled`` loader with three-way Choice yes/no/maybe.

qiaojin/PubMedQA (MIT) gold subset ``pqa_labeled`` assigns ``final_decision``
in ``{yes, no, maybe}``. typevet maps rows to JevBench-shaped ``state`` per
#75: ``{question, contexts:[{label,text}]}`` (HF ``context`` lists zipped in
order). Artificial and unlabeled configs are out of scope.

Examples:
    Load a contract-sized subset from vendored JSONL (no Hub in CI):

    ```python
    from pathlib import Path

    from typevet_evals.datasets.pubmedqa import load_labeled_split

    jsonl = Path("tests/fixtures/pubmedqa/pqa_labeled_subset.jsonl").read_text()
    rows = load_labeled_split(jsonl_text=jsonl, balanced=True, limit=12, seed=0)
    assert {r.choice_label for r in rows} == {"yes", "no", "maybe"}
    ```

See Also:
    - [typevet_evals.datasets.banking77][]: Banking77 fraud-proxy loader (separate corpus)
    - [typevet_evals.datasets.difraud][]: DIFrauD scam/legit loader (separate corpus)
    - [typevet_evals.datasets.boolq][]: BoolQ passage yes/no Noul loader (separate corpus)
    - [typevet.domain.decision_compile][]: compile Choice schemas for fixtures
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Final

import httpx

SOURCE: Final[str] = "pubmedqa"
SUBSET: Final[str] = "pqa_labeled"
SPLIT: Final[str] = "train"
PRIMARY_CHOICE_NAME: Final[str] = "answer"
CHOICE_LABELS: Final[tuple[str, ...]] = ("yes", "no", "maybe")
EXCLUDED_CONFIGS: Final[tuple[str, ...]] = ("pqa_artificial", "pqa_unlabeled")
BANNED_STATE_KEYS: Final[frozenset[str]] = frozenset(
    {
        "answer",
        "expected",
        "label",
        "ground_truth",
        "answer_key",
        "final_decision",
        "long_answer",
    }
)

DATASETS_SERVER_ROWS_URL: Final[str] = "https://datasets-server.huggingface.co/rows"
DATASETS_SERVER_INFO_URL: Final[str] = "https://datasets-server.huggingface.co/info"
DATASETS_SERVER_MAX_PAGE: Final[int] = 100
DATASET_ID: Final[str] = "qiaojin/PubMedQA"
DATASET_CONFIG: Final[str] = SUBSET

ANSWER_CHOICE_SCHEMA: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        PRIMARY_CHOICE_NAME: {
            "type": "string",
            "enum": list(CHOICE_LABELS),
            "instructions": (
                "Given the biomedical question and PubMed abstract contexts, "
                "is the answer yes, no, or maybe?"
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
class PubMedQAExample:
    """One ``pqa_labeled`` row with eval ``state`` and gold Choice label.

    Attributes:
        pubid (int): PubMed identifier from the HF row.
        state (dict[str, Any]): ``{question, contexts:[{label,text}]}`` only.
        choice_label (str): Gold ``final_decision`` in ``yes`` / ``no`` / ``maybe``.
        split (str): HF split name (``train`` for ``pqa_labeled``).
        source (str): Corpus id for eval manifests.
        subset (str): HF config name (always ``pqa_labeled``).

    Examples:
        Build one row:

        ```python
        from typevet_evals.datasets.pubmedqa import PubMedQAExample, build_state

        state = build_state("Q?", {"contexts": ["ctx"], "labels": ["BACKGROUND"]})
        PubMedQAExample(pubid=1, state=state, choice_label="yes")
        ```
    """

    pubid: int
    state: dict[str, Any]
    choice_label: str
    split: str = SPLIT
    source: str = SOURCE
    subset: str = SUBSET


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
        msg = f"PubMedQA state must not include banned keys: {names}"
        raise ValueError(msg)


def normalize_choice_label(raw: str) -> str:
    """Normalize one HF ``final_decision`` to a canonical Choice label.

    Args:
        raw: Hub ``final_decision`` string.

    Returns:
        Lowercase ``yes``, ``no``, or ``maybe``.

    Raises:
        ValueError: When ``raw`` is not a supported label.
    """
    label = raw.strip().lower()
    if label not in CHOICE_LABELS:
        msg = (
            f"PubMedQA final_decision {raw!r} is not one of {', '.join(CHOICE_LABELS)}"
        )
        raise ValueError(msg)
    return label


def build_state(question: str, context: Mapping[str, Any]) -> dict[str, Any]:
    """Build #75 PubMedQA ``state`` from HF ``question`` and ``context``.

    Zips ``context['labels']`` with ``context['contexts']`` in HF list order.

    Args:
        question: Biomedical question text.
        context: HF ``context`` feature (at least ``contexts`` and ``labels``).

    Returns:
        Object state with ``question`` and ``contexts`` entries.

    Raises:
        TypeError: When ``contexts`` or ``labels`` are not lists.
        ValueError: When list lengths differ.
    """
    texts = context.get("contexts")
    labels = context.get("labels")
    if not isinstance(texts, list) or not isinstance(labels, list):
        msg = "PubMedQA context needs list fields 'contexts' and 'labels'"
        raise TypeError(msg)
    if len(texts) != len(labels):
        msg = (
            "PubMedQA context contexts and labels length mismatch: "
            f"{len(texts)} vs {len(labels)}"
        )
        raise ValueError(msg)
    state: dict[str, Any] = {
        "question": question,
        "contexts": [
            {"label": str(label), "text": str(text)}
            for label, text in zip(labels, texts, strict=True)
        ],
    }
    assert_state_keys_allowed(state)
    return state


def map_row(record: Mapping[str, Any]) -> PubMedQAExample:
    """Map one HF ``pqa_labeled`` JSON object to a :class:`PubMedQAExample`.

    Args:
        record: Object with ``pubid``, ``question``, ``context``, ``final_decision``.

    Returns:
        Example with normalized state and choice label.

    Raises:
        ValueError: When required fields are missing or invalid.
    """
    missing = [
        key
        for key in ("pubid", "question", "context", "final_decision")
        if key not in record
    ]
    if missing:
        names = ", ".join(missing)
        msg = f"PubMedQA row missing required fields: {names}"
        raise ValueError(msg)
    context = record["context"]
    if not isinstance(context, Mapping):
        msg = "PubMedQA context must be an object"
        raise TypeError(msg)
    state = build_state(str(record["question"]), context)
    return PubMedQAExample(
        pubid=int(record["pubid"]),
        state=state,
        choice_label=normalize_choice_label(str(record["final_decision"])),
    )


def iter_labeled_rows(jsonl_text: str) -> Iterator[Mapping[str, Any]]:
    """Yield parsed JSON objects from ``pqa_labeled`` JSONL text.

    Args:
        jsonl_text: UTF-8 JSONL (one HF row per line).

    Yields:
        Parsed row objects in file order.

    Raises:
        ValueError: When a line is empty or not valid JSON.
    """
    for line_number, line in enumerate(jsonl_text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError as exc:
            msg = f"PubMedQA JSONL line {line_number} is not valid JSON"
            raise ValueError(msg) from exc
        if not isinstance(payload, Mapping):
            msg = f"PubMedQA JSONL line {line_number} must be a JSON object"
            raise TypeError(msg)
        yield payload


def map_examples(records: Iterable[Mapping[str, Any]]) -> list[PubMedQAExample]:
    """Map many HF rows to examples.

    Args:
        records: Parsed ``pqa_labeled`` objects.

    Returns:
        Mapped examples in input order.
    """
    return [map_row(record) for record in records]


def balanced_sample(
    examples: list[PubMedQAExample],
    limit: int | None,
    seed: int,
) -> list[PubMedQAExample]:
    """Return equal counts per Choice label when ``limit`` allows.

    Args:
        examples: Full mapped ``pqa_labeled`` split.
        limit: Total cap, or ``None`` for the smallest class count times three.
        seed: Seed for deterministic hash-based ordering (not ``random``).

    Returns:
        Balanced sample in seeded order.
    """
    buckets: dict[str, list[PubMedQAExample]] = {label: [] for label in CHOICE_LABELS}
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


def _rows_to_jsonl(rows: Iterable[Mapping[str, Any]]) -> str:
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)


def _fetch_labeled_split_size(client: httpx.Client) -> int:
    response = client.get(
        DATASETS_SERVER_INFO_URL,
        params={"dataset": DATASET_ID},
    )
    response.raise_for_status()
    payload = response.json()
    return int(payload["dataset_info"][DATASET_CONFIG]["splits"][SPLIT]["num_examples"])


def download_labeled_jsonl(
    *,
    client: httpx.Client | None = None,
    page_size: int = 100,
) -> str:
    """Download the public ``pqa_labeled`` train split as JSONL text.

    Fetches rows via the Hugging Face datasets server (no ``datasets`` dependency).
    Only ``pqa_labeled`` is supported; artificial and unlabeled configs are excluded.

    Args:
        client: Optional shared HTTP client for tests.
        page_size: Rows per datasets-server page (max 100).

    Returns:
        JSONL with one HF row per line.

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
        total = _fetch_labeled_split_size(active)
        collected: list[dict[str, Any]] = []
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
                collected.append(
                    {
                        "pubid": row["pubid"],
                        "question": row["question"],
                        "context": {
                            "contexts": row["context"]["contexts"],
                            "labels": row["context"]["labels"],
                        },
                        "final_decision": row["final_decision"],
                    }
                )
            offset += length
        return _rows_to_jsonl(collected)

    if client is None:
        with httpx.Client(timeout=120.0) as owned:
            return _download_with(owned)
    return _download_with(client)


def load_labeled_split(
    *,
    limit: int | None = None,
    seed: int = 0,
    balanced: bool = False,
    jsonl_text: str | None = None,
    client: httpx.Client | None = None,
) -> list[PubMedQAExample]:
    """Load ``pqa_labeled`` with optional balanced sampling.

    Args:
        limit: Row cap. With ``balanced=True``, total rows (roughly equal per
            Choice label). Without balance, first ``limit`` rows in file order.
        seed: Seed for deterministic ordering when ``balanced=True``.
        balanced: When ``True``, balance across ``yes``, ``no``, and ``maybe``.
        jsonl_text: Pre-fetched JSONL (CI fixtures). When ``None``, downloads via
            the public datasets-server API.
        client: Optional HTTP client when downloading.

    Returns:
        Mapped ``pqa_labeled`` examples.
    """
    text = (
        jsonl_text if jsonl_text is not None else download_labeled_jsonl(client=client)
    )
    examples = map_examples(iter_labeled_rows(text))
    if balanced:
        return balanced_sample(examples, limit, seed)
    if limit is not None:
        return examples[:limit]
    return examples
