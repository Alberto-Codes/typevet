"""Banking77 and BoolQ task specs for the live eval runner (#98).

Examples:
    Load vendored smoke fixtures (no Hub in CI):

    ```python
    from pathlib import Path

    from typevet_evals.runner.datasets import load_eval_tasks

    csv = Path("tests/fixtures/banking77/test_subset.csv").read_text()
    tasks = load_eval_tasks(
        "banking77",
        limit=2,
        banking77_csv_text=csv,
    )
    assert tasks[0].gold is not None
    ```

See Also:
    - [typevet.evaluation.datasets.banking77][]: Banking77 loader
    - [typevet.evaluation.datasets.boolq][]: BoolQ loader
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final, Literal

from typevet.evaluation.datasets.banking77 import (
    PRIMARY_NOUL_NAME as B77_NOUL,
)
from typevet.evaluation.datasets.banking77 import (
    REPORTS_UNAUTHORIZED_NOUL_SCHEMA,
    load_test_split,
)
from typevet.evaluation.datasets.boolq import (
    BOOLQ_ANSWER_NOUL_SCHEMA,
    load_validation_split,
    serialize_boolq_state,
)
from typevet.evaluation.datasets.boolq import (
    PRIMARY_NOUL_NAME as BOOLQ_NOUL,
)

EvalDatasetName = Literal["banking77", "boolq"]

SUPPORTED_DATASETS: Final[tuple[EvalDatasetName, ...]] = ("banking77", "boolq")

DEFAULT_BANKING77_LIMIT: Final[int] = 4
DEFAULT_BOOLQ_LIMIT: Final[int] = 4
DEFAULT_SEED: Final[int] = 0


@dataclass(frozen=True, slots=True)
class EvalTaskSpec:
    """One loader row ready for ``GenerationPort.generate``.

    Attributes:
        task_id (str): Stable id for logs.
        prompt (str): Model instruction including task state.
        schema (Mapping[str, Any]): JSON Schema object for the Noul field.
        noul_field (str): Key in ``schema`` / result ``value`` to score.
        gold (object): Expected value for ``noul_field`` after validation.
        dataset (str): Loader id echo for reporting.

    Examples:
        ```python
        from typevet_evals.runner.datasets import EvalTaskSpec

        EvalTaskSpec(
            task_id="boolq:0",
            prompt="p",
            schema={"type": "object", "additionalProperties": False},
            noul_field="answer",
            gold="yes",
            dataset="boolq",
        )
        ```
    """

    task_id: str
    prompt: str
    schema: Mapping[str, Any]
    noul_field: str
    gold: object
    dataset: str


def _banking77_prompt(text: str) -> str:
    return (
        "You are reviewing a customer banking message.\n"
        f"Message: {text}\n"
        "Return JSON only. Answer whether the customer reports a transaction "
        "they did not authorize."
    )


def _banking77_tasks(
    *,
    limit: int,
    seed: int,
    csv_text: str | None,
) -> list[EvalTaskSpec]:
    rows = load_test_split(
        limit=limit,
        seed=seed,
        balanced=True,
        csv_text=csv_text,
    )
    return [
        EvalTaskSpec(
            task_id=f"banking77:{index}:{row.intent}",
            prompt=_banking77_prompt(row.text),
            schema=REPORTS_UNAUTHORIZED_NOUL_SCHEMA,
            noul_field=B77_NOUL,
            gold=row.proxy_label == "fraud",
            dataset="banking77",
        )
        for index, row in enumerate(rows)
    ]


def _boolq_prompt(passage: str, question: str) -> str:
    state_text = serialize_boolq_state(passage, question)
    instructions = BOOLQ_ANSWER_NOUL_SCHEMA["properties"][BOOLQ_NOUL]["instructions"]
    return f"{state_text}\n\n{instructions}\nReturn JSON only matching the schema."


def _boolq_tasks(
    *,
    limit: int,
    seed: int,
    jsonl_text: str | None,
) -> list[EvalTaskSpec]:
    rows = load_validation_split(
        limit=limit,
        seed=seed,
        balanced=True,
        jsonl_text=jsonl_text,
    )
    return [
        EvalTaskSpec(
            task_id=f"boolq:{row.idx}",
            prompt=_boolq_prompt(row.passage, row.question),
            schema=BOOLQ_ANSWER_NOUL_SCHEMA,
            noul_field=BOOLQ_NOUL,
            gold=row.label,
            dataset="boolq",
        )
        for row in rows
    ]


def load_eval_tasks(
    dataset: EvalDatasetName,
    *,
    limit: int | None = None,
    seed: int = DEFAULT_SEED,
    banking77_csv_text: str | None = None,
    boolq_jsonl_text: str | None = None,
) -> list[EvalTaskSpec]:
    """Build eval tasks for one supported loader.

    Args:
        dataset: ``banking77`` or ``boolq``.
        limit: Row cap; defaults to four per loader.
        seed: Deterministic balance seed passed to loaders.
        banking77_csv_text: Pre-fetched CSV for Banking77 (CI fixtures).
        boolq_jsonl_text: Pre-fetched JSONL for BoolQ (CI fixtures).

    Returns:
        Task specs in loader order.

    Raises:
        ValueError: When ``dataset`` is unknown.
    """
    if dataset == "banking77":
        cap = limit if limit is not None else DEFAULT_BANKING77_LIMIT
        return _banking77_tasks(limit=cap, seed=seed, csv_text=banking77_csv_text)
    if dataset == "boolq":
        cap = limit if limit is not None else DEFAULT_BOOLQ_LIMIT
        return _boolq_tasks(limit=cap, seed=seed, jsonl_text=boolq_jsonl_text)
    msg = f"unsupported eval dataset {dataset!r}; expected banking77 or boolq"
    raise ValueError(msg)
