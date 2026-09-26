"""Unit tests for JevBench row loading and question mapping (#106)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from typevet.domain.judgment_questions import Choice, Noul, Score
from typevet.eval_tpjep_loader import (
    EIGHT_TASK_IDS,
    jevbench_row_to_scheduled_task,
    load_eight_task_fixture,
    model_inputs_for_task,
)

_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "tpjep"
    / "eight_task_smoke.jsonl"
)


@pytest.mark.unit
def test_fixture_has_eight_pinned_ids_in_order() -> None:
    tasks = load_eight_task_fixture(_FIXTURE.read_text(encoding="utf-8"))
    assert [t.task_id for t in tasks] == list(EIGHT_TASK_IDS)
    assert len(tasks) == 8


@pytest.mark.unit
def test_gold_never_in_model_inputs() -> None:
    tasks = load_eight_task_fixture(_FIXTURE.read_text(encoding="utf-8"))
    for task in tasks:
        _state, questions = model_inputs_for_task(task)
        assert "expected" not in questions
        blob = json.dumps(questions, default=repr)
        assert '"expected"' not in blob


@pytest.mark.unit
def test_choice_criteria_follow_upstream_label_order() -> None:
    raw = json.loads(_FIXTURE.read_text(encoding="utf-8").splitlines()[1])
    task = jevbench_row_to_scheduled_task(raw, source_tier="original")
    assert isinstance(task.question, Choice)
    assert tuple(task.question.criteria.keys()) == tuple(raw["labels"])


@pytest.mark.unit
def test_maps_syntax_to_native_types() -> None:
    tasks = load_eight_task_fixture(_FIXTURE.read_text(encoding="utf-8"))
    by_id = {t.task_id: t for t in tasks}
    assert isinstance(by_id["easy-fact-00"].question, Noul)
    assert isinstance(by_id["easy-intent-00"].question, Choice)
    assert isinstance(by_id["original-ordinal-01-0"].question, Score)
