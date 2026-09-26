"""Shared JudgmentPort contract fixtures (judgevet-aligned shape).

Each fixture is **synthetic**: the author defines state, questions, scripted
answers or failures, and expected outcomes. Contract tests exercise offline
fakes that implement [typevet.ports.judgment.JudgmentPort][].
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from typevet.domain.errors import JudgmentError, JudgmentValidationError
from typevet.domain.judgment_answers import (
    Answer,
    ChoiceAnswer,
    NoulAnswer,
    ScoreAnswer,
)
from typevet.domain.judgment_questions import Choice, Noul, Question, Score
from typevet.domain.judgment_response import JudgmentResponse, TokenUsage


def _choice_answer(label: str, options: Mapping[str, Any]) -> ChoiceAnswer:
    keys = list(options.keys())
    if label not in keys:
        msg = f"choice '{label}' not in criteria keys {keys!r}"
        raise JudgmentValidationError(msg)
    n = len(keys)
    prob = 1.0 / n
    probabilities = {k: prob for k in keys}
    probabilities[label] = 1.0 - prob * (n - 1)
    return ChoiceAnswer(
        choice=label,
        confidence=probabilities[label],
        probabilities=probabilities,
    )


def _score_answer(level: int, criteria_len: int) -> ScoreAnswer:
    legend = {i: f"level-{i}" for i in range(criteria_len)}
    probabilities = {i: 0.0 for i in range(criteria_len)}
    probabilities[level] = 1.0
    return ScoreAnswer(
        score=float(level),
        confidence=1.0,
        legend=legend,
        probabilities=probabilities,
    )


class ContractJudgmentFake:
    """Offline fake that implements ``JudgmentPort`` for contract tests."""

    def __init__(
        self,
        *,
        answers: Mapping[str, Answer] | None = None,
        fail: JudgmentError | None = None,
        usage: TokenUsage | None = None,
    ) -> None:
        """Configure scripted answers, optional failure, and usage metadata.

        Args:
            answers: Per-question answers returned when a name is scripted.
            fail: When set, ``judge`` raises this error instead of succeeding.
            usage: Token usage attached to successful responses.
        """
        self._scripted = dict(answers or {})
        self._fail = fail
        self._usage = usage or TokenUsage()
        self.calls: list[
            tuple[Any, Mapping[str, Question | Mapping[str, Any]], str]
        ] = []

    def judge(
        self,
        state: str | dict[str, Any] | list[Any],
        questions: Mapping[str, Question | Mapping[str, Any]],
        model: str,
    ) -> JudgmentResponse:
        self.calls.append((state, questions, model))
        if self._fail is not None:
            raise self._fail
        if not model.strip():
            raise JudgmentValidationError("model must be non-empty")

        result: dict[str, Answer] = {}
        for name, question in questions.items():
            if name in self._scripted:
                answer = self._scripted[name]
                if (
                    isinstance(question, Choice)
                    and isinstance(answer, ChoiceAnswer)
                    and answer.choice not in question.criteria
                ):
                    msg = (
                        f"scripted choice '{answer.choice}' "
                        f"not in criteria {list(question.criteria)!r}"
                    )
                    raise JudgmentValidationError(msg)
                result[name] = answer
                continue
            if isinstance(question, Noul):
                result[name] = NoulAnswer(noul=0.5)
            elif isinstance(question, Choice):
                first = next(iter(question.criteria))
                result[name] = _choice_answer(first, question.criteria)
            elif isinstance(question, Score):
                mid = len(question.criteria) // 2
                result[name] = _score_answer(mid, len(question.criteria))
            else:
                msg = f"unsupported wire question for {name!r}"
                raise JudgmentValidationError(msg)
        return JudgmentResponse(model=model, usage=self._usage, answers=result)


def get_fixtures() -> list[dict[str, Any]]:
    """Return shared judgment contract fixtures."""
    questions = {
        "billing": Noul(instructions="Is this about billing?"),
        "route": Choice(
            criteria={"billing": "Money", "technical": "Bugs"},
            instructions="Pick a queue:",
        ),
        "quality": Score(
            criteria=["Poor", "Fair", "Good"],
            instructions="Rate clarity:",
        ),
    }
    return [
        {
            "name": "mixed_questions_default_seed",
            "label": "synthetic",
            "state": "I was charged twice on my card.",
            "model": "fake-judgment",
            "questions": questions,
            "fake": {},
            "expect": {
                "kind": "success",
                "nouls": {"billing": 0.5},
                "choices": {"route": "billing"},
                "scores": {"quality": 1.0},
            },
        },
        {
            "name": "scripted_noul_override",
            "label": "synthetic",
            "state": "text",
            "model": "m",
            "questions": {"flag": Noul(instructions="Is it urgent?")},
            "fake": {"answers": {"flag": NoulAnswer(noul=0.9)}},
            "expect": {"kind": "success", "nouls": {"flag": 0.9}},
        },
        {
            "name": "invalid_scripted_choice",
            "label": "synthetic",
            "state": "text",
            "model": "m",
            "questions": {
                "route": Choice(criteria={"a": "A", "b": "B"}, instructions="Pick:")
            },
            "fake": {
                "answers": {
                    "route": ChoiceAnswer(
                        choice="z",
                        confidence=1.0,
                        probabilities={"z": 1.0},
                    )
                }
            },
            "expect": {"kind": "error", "exc_type": "JudgmentValidationError"},
        },
        {
            "name": "configured_failure",
            "label": "synthetic",
            "state": "text",
            "model": "m",
            "questions": {"q": Noul()},
            "fake": {"fail": JudgmentError("offline failure")},
            "expect": {"kind": "error", "exc_type": "JudgmentError"},
        },
    ]


def fake_for(fixture: dict[str, Any]) -> ContractJudgmentFake:
    """Build the contract fake for a fixture."""
    fake_cfg = fixture.get("fake", {})
    return ContractJudgmentFake(
        answers=fake_cfg.get("answers"),
        fail=fake_cfg.get("fail"),
        usage=fake_cfg.get("usage"),
    )


def exc_type_from_name(name: str) -> type[JudgmentError]:
    mapping: dict[str, type[JudgmentError]] = {
        "JudgmentError": JudgmentError,
        "JudgmentValidationError": JudgmentValidationError,
    }
    try:
        return mapping[name]
    except KeyError as exc:
        msg = f"unknown exc_type: {name}"
        raise ValueError(msg) from exc
