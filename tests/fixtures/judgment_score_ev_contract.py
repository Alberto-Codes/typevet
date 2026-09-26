"""Shared ScoreAnswer expected-value contract fixture ([#177][i177]).

Levels 0-2 with probabilities (0.15, 0.35, 0.50) yield expected score **1.35**,
which is neither the modal level (2) nor an integer rounding of the EV.

[i177]: https://github.com/Alberto-Codes/typevet/issues/177
"""

from __future__ import annotations

import math
from typing import Final

from typevet.domain.judgment_questions import Score

SCORE_EV_LEVELS: Final[tuple[str, ...]] = ("Poor", "Fair", "Good")
SCORE_EV_PROBS: Final[tuple[float, float, float]] = (0.15, 0.35, 0.50)
SCORE_EV_EXPECTED: Final[float] = 1.35
SCORE_EV_MODAL_LEVEL: Final[int] = 2

SCORE_EV_LOGPROBS: Final[dict[str, float]] = {
    str(i): math.log(p) for i, p in enumerate(SCORE_EV_PROBS)
}


def score_ev_question() -> Score:
    """Return the rubric used by the EV contract fixture.

    Returns:
        Three-level Score question matching ``SCORE_EV_PROBS``.
    """
    return Score(criteria=list(SCORE_EV_LEVELS), instructions="Rate clarity:")
