"""Exact-label mapping and failure evidence for the JevBench bridge."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, replace
from typing import Any

import pytest
from jevbench.tasks import Task
from judgevet import (
    ChoiceAnswer,
    JevError,
    NoulAnswer,
    ScoreAnswer,
    SystemOneResponse,
    Usage,
)
from judgevet.providers import ProviderError

from typevet_evals.jevbench import SystemOneAdapter

pytestmark = pytest.mark.unit


class RecordingPort:
    """Return one typed answer and retain each request."""

    def __init__(self, answer: Any = None, error: Exception | None = None) -> None:
        """Store the canned response and start an empty call record."""
        self.answer = answer
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def system_one(
        self,
        state: str | dict[str, Any] | list[Any],
        questions: Mapping[str, Any],
        model: str,
    ) -> SystemOneResponse:
        """Record the body before returning or raising."""
        self.calls.append({"state": state, "questions": questions, "model": model})
        if self.error:
            raise self.error
        answers = {} if self.answer is None else {"decision": self.answer}
        return SystemOneResponse(
            "resolved", Usage(None, 7), answers, {"decision": {"calls": 1}}
        )


def task(kind: str = "noul") -> Task:
    """Create a task whose gold and metadata must stay private."""
    criteria: Any = None
    labels = ["no", "yes"]
    expected: Any = "yes"
    if kind == "choice":
        labels = ["Yes", " yes "]
        criteria = dict.fromkeys(labels, "literal criterion")
        expected = "Yes"
    if kind == "score":
        labels = ["0", "1", "2"]
        criteria = ["low", "mid", "high"]
        expected = 2
    return Task(
        "case",
        "synthetic",
        {"text": "unchanged"},
        {
            "type": kind,
            "instructions": "  Keep\nverbatim! ",
            "criteria": criteria,
            "expected": "SECRET",
        },
        labels,
        expected,
        "public",
        provenance={"secret": "SECRET"},
    )


@pytest.mark.parametrize(
    "kind,answer,expected",
    [
        ("noul", NoulAnswer(0.7), {"no": pytest.approx(0.3), "yes": 0.7}),
        (
            "choice",
            ChoiceAnswer(" yes ", 0.8, {"Yes": 0.2, " yes ": 0.8}),
            {"Yes": 0.2, " yes ": 0.8},
        ),
        (
            "score",
            ScoreAnswer(
                1.1, 0.5, {0: "low", 1: "mid", 2: "high"}, {0: 0.4, 1: 0.1, 2: 0.5}
            ),
            {"0": 0.4, "1": 0.1, "2": 0.5},
        ),
    ],
)
def test_mapping_and_gold_exclusion(kind: str, answer: Any, expected: dict) -> None:
    """Typed probabilities preserve labels, usage, receipts and safe requests."""
    port = RecordingPort(answer)
    adapter = SystemOneAdapter(port, model="requested")
    original = task(kind)
    result = adapter.run(original)
    changed = replace(
        original, expected=None, provenance={"different": True}, id="different"
    )
    adapter.run(changed)
    assert result.ok
    assert result.probs == expected
    assert result.model == "resolved"
    assert result.status is None
    assert result.usage == {"input_tokens": None, "output_tokens": 7}
    assert isinstance(result.raw, dict)
    assert result.raw["receipts"] == {"decision": {"calls": 1}}
    assert result.probs_source == "native"
    assert port.calls[0] == port.calls[1] == result.request_body
    question = {"type": kind, "instructions": original.question["instructions"]}
    if original.question["criteria"] is not None:
        question["criteria"] = original.question["criteria"]
    assert port.calls[0] == {
        "state": original.state,
        "model": "requested",
        "questions": {"decision": question},
    }
    assert "SECRET" not in json.dumps(asdict(result), allow_nan=False)
    assert adapter.reserve_estimate(original) is None
    assert adapter.price_input_per_m is adapter.price_output_per_m is None
    assert adapter.cost_basis == "unknown"


@pytest.mark.parametrize(
    "changes",
    [
        {"labels": ["no", "no"]},
        {"labels": ["NO", "yes"]},
        {"labels": [False, "yes"]},
        {"labels": []},
        {"question": {"type": "unsupported", "instructions": "x"}},
        {
            "question": {
                "type": "choice",
                "instructions": "x",
                "criteria": {"other": "x"},
            }
        },
        {"question": {"type": "score", "instructions": "x", "criteria": ["a", "b"]}},
    ],
)
def test_invalid_task_never_calls_provider(changes: dict) -> None:
    """Malformed label domains fail before a provider call."""
    port = RecordingPort(NoulAnswer(0.7))
    result = SystemOneAdapter(port, model="test").run(replace(task(), **changes))
    assert not result.ok
    assert port.calls == []
    json.dumps(asdict(result), allow_nan=False)


@pytest.mark.parametrize(
    "answer", [None, ChoiceAnswer("x", 1.0, {"x": 1.0}), {"noul": 0.7}]
)
def test_wrong_or_missing_answer(answer: Any) -> None:
    """The named answer must have the matching typed variant."""
    result = SystemOneAdapter(RecordingPort(answer), model="test").run(task())
    assert not result.ok
    json.dumps(asdict(result), allow_nan=False)


@pytest.mark.parametrize(
    "probabilities",
    [
        {"Yes": 1.0},
        {"Yes": 0.3, " yes ": 0.3, "extra": 0.4},
        {"Yes": float("nan"), " yes ": 0.8},
        {"Yes": True, " yes ": 0.0},
    ],
)
def test_mutated_answer_fails_closed(probabilities: dict) -> None:
    """Mutable nested distributions cannot bypass consumer validation."""
    answer = ChoiceAnswer("Yes", 0.8, {"Yes": 0.8, " yes ": 0.2})
    answer.probabilities.clear()
    answer.probabilities.update(probabilities)
    result = SystemOneAdapter(RecordingPort(answer), model="test").run(task("choice"))
    assert not result.ok
    json.dumps(asdict(result), allow_nan=False)


@pytest.mark.parametrize("key", [True, "0"])
def test_score_keys_are_integer_indices(key: Any) -> None:
    """Boolean and string indices fail even after response mutation."""
    answer = ScoreAnswer(
        1.1, 0.5, {0: "low", 1: "mid", 2: "high"}, {0: 0.4, 1: 0.1, 2: 0.5}
    )
    answer.probabilities.clear()
    answer.probabilities.update({key: 0.4, 1: 0.1, 2: 0.5})
    assert (
        not SystemOneAdapter(RecordingPort(answer), model="test").run(task("score")).ok
    )


def test_score_legend_must_match_criteria() -> None:
    """A shifted rubric cannot pass as the requested score levels."""
    answer = ScoreAnswer(
        1.1, 0.5, {0: "wrong", 1: "mid", 2: "high"}, {0: 0.4, 1: 0.1, 2: 0.5}
    )
    assert (
        not SystemOneAdapter(RecordingPort(answer), model="test").run(task("score")).ok
    )


@pytest.mark.parametrize(
    "error,status",
    [
        (ProviderError("SECRET"), None),
        (JevError("SECRET", 429), 429),
        (ValueError("SECRET"), None),
    ],
)
def test_safe_failure(error: Exception, status: int | None) -> None:
    """Known failures retain only class and available HTTP status."""
    result = SystemOneAdapter(RecordingPort(error=error), model="test").run(task())
    assert not result.ok
    assert result.status == status
    assert "SECRET" not in json.dumps(asdict(result), allow_nan=False)


@pytest.mark.parametrize("field,value", [("usage", {}), ("model", None)])
def test_malformed_response_metadata(
    field: str, value: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Malformed metadata returns JSON-safe failure instead of raising."""
    response = SystemOneResponse("resolved", Usage(), {"decision": NoulAnswer(0.7)})
    object.__setattr__(response, field, value)
    port = RecordingPort()
    monkeypatch.setattr(port, "system_one", lambda **kwargs: response)
    result = SystemOneAdapter(port, model="test").run(task())
    assert not result.ok
    json.dumps(asdict(result), allow_nan=False)
