"""Offline TPJEP runner contract (#106): leakage, reorder, Score EV, transport."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from tests.fixtures.judgment_contract import ContractJudgmentFake
from typevet.domain.errors import TransportError
from typevet.domain.judgment_answers import ChoiceAnswer, NoulAnswer, ScoreAnswer
from typevet.domain.judgment_questions import Choice, Question
from typevet.domain.judgment_response import JudgmentResponse
from typevet.domain.media import ImageInput
from typevet.evaluation.tpjep.loader import (
    TPJEP_DATASET_GIT_COMMIT,
    TPJEP_MANIFEST_HASH,
    load_eight_task_fixture,
)
from typevet.evaluation.tpjep.records import summarize_tpjep_records
from typevet.evaluation.tpjep.runner import TpjepRunConfig, run_tpjep_tasks

_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "tpjep"
    / "eight_task_smoke.jsonl"
).read_text(encoding="utf-8")


class _TransportJudgmentFake:
    """Raises ``TransportError`` like a live scoring adapter would."""

    def judge(
        self,
        state: str | dict[str, Any] | list[Any],
        questions: Mapping[str, Question | Mapping[str, Any]],
        model: str,
        *,
        media: tuple[ImageInput, ...] | None = None,
    ) -> JudgmentResponse:
        _ = (state, questions, model, media)
        raise TransportError("reset")
        return JudgmentResponse(model=model)


@pytest.mark.contract
def test_offline_runner_no_gold_leakage_on_fixture_task() -> None:
    task = load_eight_task_fixture(_FIXTURE)[0]
    fake = ContractJudgmentFake(answers={"answer": NoulAnswer(noul=0.1)})
    config = TpjepRunConfig(model="offline-fake", template_class="gemma")
    records = run_tpjep_tasks(fake, [task], config=config)
    state, questions, _model = fake.calls[0]
    assert "expected" not in questions
    assert state == task.state
    assert records[0].outcome == "answered"
    assert records[0].expected == task.expected


@pytest.mark.contract
def test_label_reorder_choice_uses_upstream_order() -> None:
    task = next(
        t
        for t in load_eight_task_fixture(_FIXTURE)
        if t.task_id == "original-intent-01-0"
    )
    assert isinstance(task.question, Choice)
    keys = tuple(task.question.criteria.keys())
    fake = ContractJudgmentFake(
        answers={
            "answer": ChoiceAnswer(
                choice=keys[0],
                confidence=1.0,
                probabilities={k: (1.0 if k == keys[0] else 0.0) for k in keys},
            )
        }
    )
    records = run_tpjep_tasks(
        fake,
        [task],
        config=TpjepRunConfig(model="m", template_class="gemma"),
    )
    assert records[0].predicted == keys[0]


@pytest.mark.contract
def test_score_expected_value_can_differ_from_modal() -> None:
    task = next(
        t
        for t in load_eight_task_fixture(_FIXTURE)
        if t.task_id == "original-ordinal-01-0"
    )
    probs = {0: 0.45, 1: 0.45, 2: 0.05, 3: 0.05}
    fake = ContractJudgmentFake(
        answers={
            "answer": ScoreAnswer(
                score=sum(k * v for k, v in probs.items()),
                confidence=max(probs.values()),
                legend={i: str(i) for i in probs},
                probabilities=probs,
            )
        }
    )
    records = run_tpjep_tasks(
        fake,
        [task],
        config=TpjepRunConfig(model="m", template_class="gemma"),
    )
    modal = max(probs, key=probs.__getitem__)
    assert records[0].predicted != modal
    assert records[0].predicted == pytest.approx(sum(k * v for k, v in probs.items()))


@pytest.mark.contract
def test_transport_failure_retained_in_scheduled_totals() -> None:
    task = load_eight_task_fixture(_FIXTURE)[0]
    fake = _TransportJudgmentFake()
    records = run_tpjep_tasks(
        fake,
        [task],
        config=TpjepRunConfig(model="m", template_class="gemma"),
    )
    summary = summarize_tpjep_records(records)
    assert summary.n_scheduled == 1
    assert summary.n_transport_failed == 1
    assert records[0].dataset_git_commit == TPJEP_DATASET_GIT_COMMIT
    assert records[0].dataset_hash == TPJEP_MANIFEST_HASH
