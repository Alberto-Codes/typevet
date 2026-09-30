"""Offline contract: a check-versus-register request through the scoring adapter.

The fake scorer replaces the served model. These tests prove that one case
becomes one judgment with one image and four typed answers. They say
nothing about model quality or about real checks.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import pytest

from tests.fixtures.judgment_scoring_contract import SequentialScoringFake
from typevet.adapters.outbound.gemma import ServedTemplateClass
from typevet.adapters.outbound.judgment_scoring import ScoringJudgmentAdapter
from typevet.domain import Choice, ImageInput, Noul, Score
from typevet.domain.media import count_media_markers
from typevet_evals.check_match import (
    AMOUNTS_MATCH,
    LEGIBILITY,
    PAYEE_MATCHES,
    VERDICT,
    VERDICT_LABELS,
    CheckVariant,
    build_check_match_request,
    check_cases,
    check_match_questions,
    judge_check_match,
    render_check,
)

pytestmark = pytest.mark.contract

MODEL = "fake-check-match"
CASE = next(
    case for case in check_cases() if case.variant is CheckVariant.PAYEE_CHANGED
)


def _tokenizer() -> Callable[[str], tuple[int, ...]]:
    """Return a tokenizer that gives each distinct text one unique token id."""
    ids: dict[str, int] = {}

    def tokenize(text: str) -> tuple[int, ...]:
        if not text:
            return ()
        return (ids.setdefault(text, len(ids) + 1),)

    return tokenize


def _logs(probabilities: dict[str, float]) -> dict[str, float]:
    return {label: math.log(p) for label, p in probabilities.items()}


def _adapter() -> tuple[ScoringJudgmentAdapter, SequentialScoringFake]:
    fake = SequentialScoringFake(
        [
            _logs({"True": 0.2, "False": 0.8}),
            _logs({"True": 0.9, "False": 0.1}),
            _logs(
                {
                    "consistent": 0.1,
                    "payee_mismatch": 0.6,
                    "amount_mismatch": 0.1,
                    "date_mismatch": 0.1,
                    "unsigned": 0.05,
                    "cannot_tell": 0.05,
                }
            ),
            _logs({"0": 0.05, "1": 0.05, "2": 0.1, "3": 0.2, "4": 0.6}),
        ]
    )
    adapter = ScoringJudgmentAdapter(
        fake,
        tokenize_content=_tokenizer(),
        served_template=ServedTemplateClass.NATIVE_GEMMA4_TURN,
    )
    return adapter, fake


def test_questions_are_two_nouls_one_choice_and_one_score() -> None:
    questions = check_match_questions()
    assert list(questions) == [PAYEE_MATCHES, AMOUNTS_MATCH, VERDICT, LEGIBILITY]
    assert isinstance(questions[PAYEE_MATCHES], Noul)
    assert isinstance(questions[AMOUNTS_MATCH], Noul)
    verdict = questions[VERDICT]
    assert isinstance(verdict, Choice)
    assert tuple(verdict.criteria) == VERDICT_LABELS
    assert VERDICT_LABELS == (
        "consistent",
        "payee_mismatch",
        "amount_mismatch",
        "date_mismatch",
        "unsigned",
        "cannot_tell",
    )
    unsigned = verdict.criteria["unsigned"]
    assert isinstance(unsigned, str)
    assert "handwritten" in unsigned
    assert "VOID do not count" in unsigned
    legibility = questions[LEGIBILITY]
    assert isinstance(legibility, Score)
    assert len(legibility.criteria) == 5


def test_request_carries_one_png_and_the_register_row_as_text() -> None:
    image = render_check(CASE)
    request = build_check_match_request(CASE, image=image)
    assert request.media == (ImageInput(data=image, mime_type="image/png"),)
    assert request.case == CASE
    assert request.case_id == CASE.case_id
    assert CASE.row.as_text() in request.state
    assert CASE.face.payee not in request.state


def test_fake_port_sees_one_image_and_returns_four_typed_answers() -> None:
    adapter, fake = _adapter()
    image = render_check(CASE)
    request = build_check_match_request(CASE, image=image)
    response = judge_check_match(adapter, request, MODEL)

    assert len(fake.calls) == 4
    for call in fake.calls:
        assert [item.data for item in call.media] == [image]
        assert count_media_markers(call.prefix) == 1
        assert CASE.row.payee in call.prefix
    assert response.nouls[PAYEE_MATCHES].noul == pytest.approx(0.2)
    assert response.nouls[AMOUNTS_MATCH].noul == pytest.approx(0.9)
    assert response.choices[VERDICT].choice == "payee_mismatch"
    assert sorted(response.scores[LEGIBILITY].legend) == [0, 1, 2, 3, 4]
    assert set(response.answers) == {PAYEE_MATCHES, AMOUNTS_MATCH, VERDICT, LEGIBILITY}
