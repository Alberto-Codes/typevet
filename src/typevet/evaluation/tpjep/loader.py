"""Load pinned JevBench rows and map them to native judgment inputs (#106).

Gold ``expected`` values stay on the scheduled task for scoring only. They are
never passed to ``JudgmentPort.judge``.

Examples:
    ```python
    from pathlib import Path

    from typevet.evaluation.tpjep.loader import load_eight_task_fixture

    text = Path("tests/fixtures/tpjep/eight_task_smoke.jsonl").read_text()
    tasks = load_eight_task_fixture(text)
    assert len(tasks) == 8
    ```

See Also:
    - [typevet.evaluation.tpjep.runner][]: Offline and live runner
    - [typevet.evaluation.tpjep.records][]: Frozen attempt JSONL (#131)
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final, Literal

from typevet.domain.judgment_questions import Choice, Noul, Question, Score

TPJEP_DATASET_GIT_COMMIT: Final[str] = "f8ce71361165846101d02ebc83ad44e47ae44fc3"
TPJEP_MANIFEST_HASH: Final[str] = (
    "dc3995d8ae1e2fc8e81ce38431add300eb8bb39b85aadfd0c7c32079382dde51"
)
TPJEP_MANIFEST_RECIPE: Final[str] = "typellm_manifest_sha256"
TPJEP_LOCAL_CONCAT_HASH: Final[str] = (
    "771ea8ac252af155606eeb09cad920b32b326c5c2b1e9c9af868be00fe8e2fad"
)
TPJEP_LOCAL_CONCAT_RECIPE: Final[str] = "sha256_concat_original_easy_hard_raw_bytes"

EIGHT_TASK_IDS: Final[tuple[str, ...]] = (
    "original-policy-01-0",
    "original-intent-01-0",
    "original-ordinal-01-0",
    "easy-fact-00",
    "easy-intent-00",
    "hard-opus-a-long_policy-11",
    "hard-opus-a-long_policy-01",
    "hard-opus-a-temporal_numeric-12",
)

_SOURCE_TIER: Final[dict[str, str]] = {
    task_id: (
        "original"
        if task_id.startswith("original")
        else "easy"
        if task_id.startswith("easy")
        else "hard"
    )
    for task_id in EIGHT_TASK_IDS
}

PRIMARY_QUESTION_NAME: Final[str] = "answer"
_SCORE_MIN_LEVELS: Final[int] = 2
QuestionTypeName = Literal["Noul", "Choice", "Score"]


@dataclass(frozen=True, slots=True)
class TpjepScheduledTask:
    """One scheduled TPJEP attempt before judgment.

    Attributes:
        task_id (str): Public JevBench task id.
        source_tier (str): ``original``, ``easy``, or ``hard``.
        question_type (QuestionTypeName): Primitive name for records.
        state (str): Text under evaluation.
        question_name (str): Single question key passed to ``judge``.
        question (Question): Native Noul, Choice, or Score.
        expected (object): Gold label for scoring only.
        labels (tuple[str, ...]): Upstream label order for probability keys.

    Examples:
        ```python
        from typevet.evaluation.tpjep.loader import load_eight_task_fixture

        task = load_eight_task_fixture(text)[0]
        assert task.task_id
        ```
    """

    task_id: str
    source_tier: str
    question_type: QuestionTypeName
    state: str
    question_name: str
    question: Question
    expected: object
    labels: tuple[str, ...]


def _syntax_to_type_name(syntax: str) -> QuestionTypeName:
    mapping: dict[str, QuestionTypeName] = {
        "noul": "Noul",
        "choice": "Choice",
        "score": "Score",
    }
    try:
        return mapping[syntax]
    except KeyError as exc:
        msg = f"unsupported JevBench question type {syntax!r}"
        raise ValueError(msg) from exc


def _map_question(row: Mapping[str, Any]) -> Question:
    raw = row["question"]
    if not isinstance(raw, Mapping):
        msg = "question must be a mapping"
        raise TypeError(msg)
    syntax = raw.get("type")
    instructions = raw.get("instructions")
    if not isinstance(instructions, str) or not instructions:
        msg = "question instructions must be a non-empty string"
        raise ValueError(msg)
    criteria = raw.get("criteria")
    if syntax == "noul":
        crit = criteria if isinstance(criteria, Mapping) else None
        return Noul(instructions=instructions, criteria=dict(crit) if crit else None)
    if syntax == "choice":
        if not isinstance(criteria, Mapping):
            msg = "choice criteria must be a mapping"
            raise TypeError(msg)
        labels = row.get("labels")
        if not isinstance(labels, list) or not labels:
            msg = "choice tasks require non-empty labels list"
            raise ValueError(msg)
        ordered = {str(key): criteria[key] for key in labels}
        return Choice(criteria=ordered, instructions=instructions)
    if syntax == "score":
        if not isinstance(criteria, list) or len(criteria) < _SCORE_MIN_LEVELS:
            msg = "score criteria must be a list with at least two levels"
            raise ValueError(msg)
        return Score(criteria=criteria, instructions=instructions)
    msg = f"unsupported JevBench question type {syntax!r}"
    raise ValueError(msg)


def jevbench_row_to_scheduled_task(
    row: Mapping[str, Any],
    *,
    source_tier: str | None = None,
) -> TpjepScheduledTask:
    """Map one public JevBench JSONL row to a scheduled task.

    Args:
        row: Parsed JSON object with ``id``, ``state``, ``expected``, ``question``.
        source_tier: Tier override; inferred from ``id`` when omitted.

    Returns:
        Task ready for ``run_tpjep_tasks`` without gold in model inputs.

    Raises:
        TypeError: When ``state`` or nested shapes are wrong.
        ValueError: When ``id`` or question syntax is unsupported.
    """
    task_id = str(row["id"])
    tier = source_tier or _SOURCE_TIER.get(task_id)
    if tier is None:
        msg = f"unknown task id {task_id!r}"
        raise ValueError(msg)
    state = row.get("state")
    if not isinstance(state, str):
        msg = "state must be a string"
        raise TypeError(msg)
    question = _map_question(row)
    syntax = row["question"]["type"]
    labels_raw = row.get("labels")
    if isinstance(labels_raw, list):
        labels = tuple(str(x) for x in labels_raw)
    elif isinstance(question, Score):
        labels = tuple(str(i) for i in range(len(question.criteria)))
    else:
        labels = ("false", "true")
    return TpjepScheduledTask(
        task_id=task_id,
        source_tier=tier,
        question_type=_syntax_to_type_name(str(syntax)),
        state=state,
        question_name=PRIMARY_QUESTION_NAME,
        question=question,
        expected=row.get("expected"),
        labels=labels,
    )


def model_inputs_for_task(
    task: TpjepScheduledTask,
) -> tuple[str, dict[str, Question]]:
    """Return ``state`` and ``questions`` for ``JudgmentPort.judge`` (no gold).

    Args:
        task: Scheduled row with scoring-only ``expected``.

    Returns:
        State text and a one-entry questions mapping.
    """
    return task.state, {task.question_name: task.question}


def load_eight_task_fixture(jsonl_text: str) -> list[TpjepScheduledTask]:
    """Load the eight pinned smoke tasks in schedule order.

    Args:
        jsonl_text: UTF-8 JSONL body (typically ``eight_task_smoke.jsonl``).

    Returns:
        Eight tasks ordered per ``EIGHT_TASK_IDS``.

    Raises:
        TypeError: When a line is not a JSON object.
        ValueError: When a pinned id is missing from the fixture.
    """
    by_id: dict[str, TpjepScheduledTask] = {}
    for line_number, line in enumerate(jsonl_text.splitlines(), 1):
        stripped = line.strip()
        if not stripped:
            continue
        row = json.loads(stripped)
        if not isinstance(row, dict):
            msg = f"line {line_number} must be a JSON object"
            raise TypeError(msg)
        task = jevbench_row_to_scheduled_task(row)
        by_id[task.task_id] = task
    missing = [task_id for task_id in EIGHT_TASK_IDS if task_id not in by_id]
    if missing:
        msg = f"fixture missing pinned task ids: {missing}"
        raise ValueError(msg)
    return [by_id[task_id] for task_id in EIGHT_TASK_IDS]
