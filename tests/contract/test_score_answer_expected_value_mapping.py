"""Contract: ScoreAnswer.score is probability-weighted expected level ([#177][i177])."""

from __future__ import annotations

import pytest

from tests.fixtures.judgment_score_ev_contract import (
    SCORE_EV_EXPECTED,
    SCORE_EV_LOGPROBS,
    SCORE_EV_MODAL_LEVEL,
    score_ev_question,
)
from tests.fixtures.judgment_scoring_contract import adapter_for


@pytest.mark.contract
def test_score_answer_expected_value_is_fractional_not_modal_level() -> None:
    """ScoreAnswer.score stays at 1.35 EV, not modal level 2 or int rounding."""
    adapter, fake = adapter_for(logprobs_by_call=[SCORE_EV_LOGPROBS])
    response = adapter.judge("text", {"quality": score_ev_question()}, "fake")
    answer = response.scores["quality"]
    assert answer.score == pytest.approx(SCORE_EV_EXPECTED)
    assert answer.score != float(SCORE_EV_MODAL_LEVEL)
    assert int(answer.score) != answer.score
    assert len(fake.calls) == 1
