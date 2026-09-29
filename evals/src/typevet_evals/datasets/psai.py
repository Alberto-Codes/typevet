"""PSAI computer-use metadata loader (System One slice, no vision).

``anaisleila/computer-use-data-psai`` (MIT) ships multimodal traces; typevet v1
loads **metadata only** ([#56](https://github.com/Alberto-Codes/typevet/issues/56)):
dedupe ``unique_data_id``, shuffle before sample, map Hub fields to Choice/Noul
Decisions, and strip screenshots without decode. Eval unit is one **task** row.

Examples:
    Load smoke rows from vendored JSONL (no Hub in CI):

    ```python
    from pathlib import Path

    from typevet_evals.datasets.psai import export_tasks, load_train_split

    jsonl = Path("tests/fixtures/psai/metadata_smoke.jsonl").read_text()
    rows = load_train_split(jsonl_text=jsonl, limit=4, seed=0)
    tasks = export_tasks(rows)
    assert tasks[0]["expected"]["category"] in CATEGORY_LABELS
    ```

See Also:
    - [typevet_evals.datasets.psai_download][]: HF parquet metadata streaming
    - [typevet_evals.datasets.psai_stream][]: dedupe, shuffle, strip helpers
    - docs/reference/eval-psai-metadata-map.md: field map
    - [typevet.domain.decision_compile][]: compile Decision schemas
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import httpx

from typevet_evals.datasets.psai_download import iter_train_metadata_parquet
from typevet_evals.datasets.psai_schema import (
    APP_TYPE_LABELS,
    BANNED_STATE_KEYS,
    BENCHMARK_LABELS,
    CATEGORY_LABELS,
    DATASET_ID,
    DATASET_LICENSE,
    DEFAULT_SAMPLE_SEED,
    DIFFICULTY_LABELS,
    FAMILY,
    METADATA_DECISIONS_SCHEMA,
    METADATA_MANIFEST,
    OS_LABELS,
    OTHER_SUB_CATEGORY,
    PRIMARY_NOUL_NAME,
    SOURCE,
    SPLIT,
    SUB_CATEGORY_CHOICE_NAME,
    SUB_CATEGORY_LABELS,
    TASK_NAME_COLUMN,
)
from typevet_evals.datasets.psai_stream import (
    dedupe_shuffle_sample,
    iter_jsonl_rows,
)


@dataclass(frozen=True, slots=True)
class PsaiExample:
    """One deduped PSAI task with metadata gold Decisions.

    Attributes:
        unique_data_id (str): Hub dedupe key.
        task_name (str): Natural-language task description (eval state text).
        category (str): ``BROWSER_TASK`` or ``COMPUTER_TASK``.
        difficulty (str): ``EASY``, ``MEDIUM``, or ``HARD``.
        benchmark (str): Benchmark bucket label.
        app_type (str): ``SINGLE_APP`` or ``MULTI_APP`` (Hub ``appType``).
        os (str): OS scope label (≤4 enum).
        requires_login (bool): Login requirement Noul (``""`` maps to false).
        sub_category (str): Capped Choice from ``subCategory[0]`` or ``OTHER``.
        split (str): HF split (``train`` for v1).
        source (str): Corpus id for eval manifests.

    Examples:
        Build one row:

        ```python
        from typevet_evals.datasets.psai import PsaiExample

        PsaiExample(
            unique_data_id="abc",
            task_name="Open settings.",
            category="BROWSER_TASK",
            difficulty="EASY",
            benchmark="Web Bench",
            app_type="SINGLE_APP",
            os="CROSS_PLATFORM",
            requires_login=False,
            sub_category="Web Settings",
        )
        ```
    """

    unique_data_id: str
    task_name: str
    category: str
    difficulty: str
    benchmark: str
    app_type: str
    os: str
    requires_login: bool
    sub_category: str
    split: str = SPLIT
    source: str = SOURCE


def metadata_manifest() -> dict[str, Any]:
    """Return metadata-only manifest constants for eval inventories.

    Returns:
        Deep-copied manifest metadata (license, non-goals, seed).
    """
    return json.loads(json.dumps(METADATA_MANIFEST))


def _choice_label(raw: object, labels: tuple[str, ...], field: str) -> str:
    text = str(raw)
    if text not in labels:
        allowed = ", ".join(labels)
        msg = f"PSAI {field} {text!r} is not one of {allowed}"
        raise ValueError(msg)
    return text


def noul_requires_login(raw: object) -> bool:
    """Map Hub ``requires_login`` string to boolean Noul gold.

    Args:
        raw: Hub string (``yes`` / ``no`` / empty).

    Returns:
        ``True`` only when normalized value is ``yes``; empty string is false.

    Raises:
        ValueError: When the string is neither empty, ``yes``, nor ``no``.
    """
    text = str(raw).strip().lower()
    if text == "":
        return False
    if text == "yes":
        return True
    if text == "no":
        return False
    msg = f"PSAI requires_login {raw!r} must be '', 'yes', or 'no'"
    raise ValueError(msg)


def choice_sub_category(raw: object) -> str:
    """Map ``subCategory[0]`` to capped Choice gold with ``OTHER`` fallback.

    Args:
        raw: Hub list of sub-category strings (or scalar for tests).

    Returns:
        Known label from ``SUB_CATEGORY_LABELS`` or ``OTHER``.
    """
    if isinstance(raw, list):
        if not raw:
            return OTHER_SUB_CATEGORY
        first = str(raw[0])
    else:
        first = str(raw)
    if first in SUB_CATEGORY_LABELS:
        return first
    return OTHER_SUB_CATEGORY


def psai_state(task_name: str) -> dict[str, str]:
    """Build object ``state`` with ``task_name`` only.

    Args:
        task_name: Natural-language task description.

    Returns:
        Mapping with a single ``task_name`` key.
    """
    return {TASK_NAME_COLUMN: task_name}


def validate_task_state(state: Mapping[str, Any]) -> None:
    """Reject leakage keys in exported task ``state``.

    Args:
        state: Task state mapping.

    Raises:
        ValueError: When a banned key is present or keys are not exactly
            ``task_name``.
    """
    banned = BANNED_STATE_KEYS.intersection(state.keys())
    if banned:
        msg = f"PSAI state must not include leakage keys: {sorted(banned)!r}"
        raise ValueError(msg)
    expected = {TASK_NAME_COLUMN}
    if set(state.keys()) != expected:
        msg = (
            f"PSAI state keys must be exactly {sorted(expected)!r}; "
            f"got {sorted(state.keys())!r}"
        )
        raise ValueError(msg)


def map_row(record: Mapping[str, Any]) -> PsaiExample:
    """Map one metadata record to a :class:`PsaiExample`.

    Args:
        record: Object with Hub metadata fields (heavy columns already stripped).

    Returns:
        Example with normalized Decision gold.

    Raises:
        ValueError: When required fields are missing or enum values are invalid.
    """
    required = (
        "unique_data_id",
        TASK_NAME_COLUMN,
        "category",
        "difficulty",
        "benchmark",
        "appType",
        "os",
        "requires_login",
        "subCategory",
    )
    missing = [key for key in required if key not in record]
    if missing:
        names = ", ".join(missing)
        msg = f"PSAI row missing required metadata fields: {names}"
        raise ValueError(msg)
    return PsaiExample(
        unique_data_id=str(record["unique_data_id"]),
        task_name=str(record[TASK_NAME_COLUMN]),
        category=_choice_label(record["category"], CATEGORY_LABELS, "category"),
        difficulty=_choice_label(record["difficulty"], DIFFICULTY_LABELS, "difficulty"),
        benchmark=_choice_label(record["benchmark"], BENCHMARK_LABELS, "benchmark"),
        app_type=_choice_label(record["appType"], APP_TYPE_LABELS, "appType"),
        os=_choice_label(record["os"], OS_LABELS, "os"),
        requires_login=noul_requires_login(record["requires_login"]),
        sub_category=choice_sub_category(record["subCategory"]),
    )


def map_examples(records: Iterable[Mapping[str, Any]]) -> list[PsaiExample]:
    """Map many metadata records to examples.

    Args:
        records: Metadata dicts after dedupe/shuffle sampling.

    Returns:
        Mapped examples in input order.
    """
    return [map_row(record) for record in records]


def task_id(example: PsaiExample) -> str:
    """Stable id ``psai-train-{unique_data_id}``.

    Args:
        example: Mapped metadata row.

    Returns:
        JevBench-style task id string.
    """
    return f"{FAMILY}-{example.split}-{example.unique_data_id}"


def questions_payload() -> list[dict[str, Any]]:
    """JevBench-shaped Decision entries for metadata fields.

    Returns:
        Choice and Noul question descriptors in schema order.
    """
    props = METADATA_DECISIONS_SCHEMA["properties"]
    return [
        {
            "name": "category",
            "syntax": "Choice",
            "labels": list(CATEGORY_LABELS),
            "instructions": props["category"]["instructions"],
        },
        {
            "name": "difficulty",
            "syntax": "Choice",
            "labels": list(DIFFICULTY_LABELS),
            "instructions": props["difficulty"]["instructions"],
        },
        {
            "name": "benchmark",
            "syntax": "Choice",
            "labels": list(BENCHMARK_LABELS),
            "instructions": props["benchmark"]["instructions"],
        },
        {
            "name": "appType",
            "syntax": "Choice",
            "labels": list(APP_TYPE_LABELS),
            "instructions": props["appType"]["instructions"],
        },
        {
            "name": "os",
            "syntax": "Choice",
            "labels": list(OS_LABELS),
            "instructions": props["os"]["instructions"],
        },
        {
            "name": PRIMARY_NOUL_NAME,
            "syntax": "Noul",
            "instructions": props[PRIMARY_NOUL_NAME]["instructions"],
        },
        {
            "name": SUB_CATEGORY_CHOICE_NAME,
            "syntax": "Choice",
            "labels": list(SUB_CATEGORY_LABELS),
            "instructions": props[SUB_CATEGORY_CHOICE_NAME]["instructions"],
        },
    ]


def export_task(example: PsaiExample) -> dict[str, Any]:
    """Export one example as a JevBench-shaped complementary task.

    Args:
        example: Mapped metadata row.

    Returns:
        Task dict with ``id``, ``family``, ``split``, ``provenance``,
        ``expected``, ``state``, and ``questions``.

    Raises:
        ValueError: When ``state`` would include banned keys.
    """
    state = psai_state(example.task_name)
    validate_task_state(state)
    return {
        "id": task_id(example),
        "family": FAMILY,
        "split": example.split,
        "provenance": {
            "dataset_id": DATASET_ID,
            "license": DATASET_LICENSE,
            "split": example.split,
            "unique_data_id": example.unique_data_id,
            "source": example.source,
        },
        "expected": {
            "category": example.category,
            "difficulty": example.difficulty,
            "benchmark": example.benchmark,
            "appType": example.app_type,
            "os": example.os,
            PRIMARY_NOUL_NAME: example.requires_login,
            SUB_CATEGORY_CHOICE_NAME: example.sub_category,
        },
        "state": state,
        "questions": questions_payload(),
    }


def export_tasks(examples: Sequence[PsaiExample]) -> list[dict[str, Any]]:
    """Export many examples as JevBench-shaped tasks.

    Args:
        examples: Mapped metadata rows.

    Returns:
        Task dicts in input order.
    """
    return [export_task(example) for example in examples]


def load_train_split(
    *,
    limit: int | None = None,
    seed: int = DEFAULT_SAMPLE_SEED,
    jsonl_text: str | None = None,
    client: httpx.Client | None = None,
    parquet_urls: list[str] | None = None,
) -> list[PsaiExample]:
    """Load PSAI train metadata with dedupe, shuffle, and optional cap.

    Args:
        limit: Row cap after shuffle, or ``None`` for every deduped row.
        seed: Seed for deterministic hash-based ordering.
        jsonl_text: Pre-fetched metadata JSONL (CI fixtures). When ``None``,
            streams parquet shards via ``eval_psai_download`` (needs pyarrow).
        client: Optional HTTP client when streaming parquet.
        parquet_urls: Optional shard URL override for tests.

    Returns:
        Mapped train examples.

    Raises:
        ImportError: When parquet streaming is requested without pyarrow.
    """
    if jsonl_text is not None:
        sampled = dedupe_shuffle_sample(
            iter_jsonl_rows(jsonl_text), limit=limit, seed=seed
        )
    else:
        sampled = dedupe_shuffle_sample(
            iter_train_metadata_parquet(client=client, urls=parquet_urls),
            limit=limit,
            seed=seed,
        )
    return map_examples(sampled)
