"""Contract tests: a collections record reaches a JudgmentPort ([#236][i236]).

A recording fake port stands in for the scoring adapter. The record and the
seed are synthetic and use finvet field names only.

Examples:
    ```bash
    uv run pytest -q evals/tests/contract/test_collections_workload_contract.py
    ```

See Also:
    - [typevet_evals.throughput.collections_workload][]: workload and parity

[i236]: https://github.com/Alberto-Codes/typevet/issues/236
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from typevet.domain.judgment_answers import ChoiceAnswer, NoulAnswer
from typevet.domain.judgment_questions import Choice, Noul
from typevet.domain.judgment_response import JudgmentResponse
from typevet.domain.media import ImageInput
from typevet.ports.judgment import JudgmentPort
from typevet_evals.throughput.collections_workload import (
    judge_record,
    load_questions,
    load_records,
)


class _RecordingPort:
    def __init__(self) -> None:
        self.calls: list[tuple[Any, Mapping[str, Any], str]] = []

    def judge(
        self,
        state: str | dict[str, Any] | list[Any],
        questions: Mapping[str, Any],
        model: str,
        *,
        media: tuple[ImageInput, ...] | None = None,
    ) -> JudgmentResponse:
        self.calls.append((state, questions, model))
        return JudgmentResponse(
            model=model,
            answers={
                "will_engage": NoulAnswer(noul=0.7),
                "accepted_offer": ChoiceAnswer(
                    choice="NONE",
                    confidence=0.6,
                    probabilities={"NONE": 0.6, "PLAN_A": 0.4},
                ),
            },
        )


@pytest.mark.contract
def test_port_receives_dict_state_and_one_noul_one_choice(tmp_path: Path) -> None:
    record = {
        "id": "r-9",
        "state": {"balance": 900},
        "logged": {
            "action_taken": {"action": "PLAN_A"},
            "outcome": {"engaged": True, "accepted_offer": "PLAN_A"},
            "compliance_issues": [],
        },
        "meta": {"scenario_id": "s-2"},
    }
    split = tmp_path / "val.jsonl"
    split.write_text(json.dumps(record) + "\n", encoding="utf-8")
    seed = tmp_path / "seed.json"
    seed.write_text(
        json.dumps(
            {
                "will_engage": {
                    "type": "noul",
                    "instructions": "Engage?",
                    "criteria": {"true": "Yes.", "false": "No."},
                },
                "accepted_offer": {
                    "type": "choice",
                    "instructions": "Which offer?",
                    "criteria": {"NONE": "None.", "PLAN_A": "Plan A."},
                },
            }
        ),
        encoding="utf-8",
    )
    fake = _RecordingPort()
    port: JudgmentPort = fake
    (row,) = load_records(split)

    prob = judge_record(port, row, load_questions(seed), "m-1")

    assert prob == 0.7
    assert len(fake.calls) == 1
    state, questions, model = fake.calls[0]
    assert model == "m-1"
    assert state == {"balance": 900, "proposed_action": {"action": "PLAN_A"}}
    assert isinstance(state, dict)
    assert set(questions) == {"will_engage", "accepted_offer"}
    assert [type(q) for q in questions.values()].count(Noul) == 1
    assert [type(q) for q in questions.values()].count(Choice) == 1
