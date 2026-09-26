"""Unit tests: consumer dispatch ledger ([#177][i177]).

Examples:
    ```bash
    uv run pytest -q tests/unit/test_psai_vision_consumer_dispatch.py
    ```

See Also:
    - [typevet.evaluation.psai_vision_consumer_dispatch][]: ledger helpers
"""

from __future__ import annotations

import pytest

from typevet.domain.candidate_scoring_request import (
    CandidateScoringRequest,
    CandidateTokenSpec,
)
from typevet.domain.scoring_stage import ScoreStage
from typevet.evaluation.psai_vision_consumer_accounting import ConsumerCallBudgetError
from typevet.evaluation.psai_vision_consumer_dispatch import (
    ConsumerDispatchLedger,
    wrap_judgment_port,
    wrap_scoring_port,
)
from typevet.evaluation.psai_vision_consumer_protocol import FROZEN_SCORING_REQUESTS
from typevet.ports.judgment import JudgmentPort
from typevet.testing import ScriptedScoringFake


class _RaisingJudgmentPort(JudgmentPort):
    def judge(self, state, questions, model, *, media=None):
        """Raise to exercise failure accounting on the judgment wrapper.

        Raises:
            ValueError: Always raised for this test double.
        """
        msg = "boom"
        raise ValueError(msg)


@pytest.mark.unit
def test_scoring_budget_raises_when_exceeded() -> None:
    """Scoring wrapper enforces the frozen scoring ceiling."""
    ledger = ConsumerDispatchLedger()
    ledger.scoring_requests = FROZEN_SCORING_REQUESTS
    fake = ScriptedScoringFake(logprobs={"True": -0.2, "False": -1.0})
    port = wrap_scoring_port(fake, ledger)
    request = CandidateScoringRequest(
        model="m",
        prefix="p",
        candidates=(CandidateTokenSpec("True", (1,)),),
        stage=ScoreStage.PRE_SAMPLING,
    )
    with pytest.raises(ConsumerCallBudgetError):
        port.score_candidates(request)


@pytest.mark.unit
def test_judgment_failure_increments_failed_attempts() -> None:
    """Judgment wrapper records failed attempts when inner port raises."""
    ledger = ConsumerDispatchLedger()
    port = wrap_judgment_port(_RaisingJudgmentPort(), ledger)
    with pytest.raises(ValueError, match="boom"):
        port.judge("s", {}, "m")
    assert ledger.failed_attempts == 1


@pytest.mark.unit
def test_metadata_and_tokenizer_auxiliary_counters() -> None:
    """Metadata and tokenizer HTTP use independent auxiliary counters."""
    ledger = ConsumerDispatchLedger()
    ledger.before_metadata_http()
    ledger.before_tokenizer_http()
    assert ledger.auxiliary_http_total == 2
