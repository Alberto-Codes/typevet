"""Attempt ledger and budget-enforcing port wrappers ([#177][i177]).

Examples:
    ```python
    from typevet_evals.psai_vision_consumer.dispatch import (
        ConsumerDispatchLedger,
        wrap_judgment_port,
        wrap_scoring_port,
    )
    from typevet.testing import ScriptedScoringFake

    ledger = ConsumerDispatchLedger()
    fake = ScriptedScoringFake(logprobs={"True": -0.2, "False": -1.0})
    scoring = wrap_scoring_port(fake, ledger)
    assert scoring is not None
    ```

See Also:
    - [typevet_evals.psai_vision_consumer.accounting][]: frozen totals
    - [typevet_evals.psai_vision_consumer.protocol][]: ceilings

[i177]: https://github.com/Alberto-Codes/typevet/issues/177
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from typevet.domain.candidate_scoring_request import CandidateScoringRequest
from typevet.domain.candidate_scoring_response import CandidateScoringResult
from typevet.domain.judgment_questions import Question
from typevet.domain.judgment_response import JudgmentResponse
from typevet.domain.media import ImageInput
from typevet.ports.judgment import JudgmentPort
from typevet.ports.scoring import CandidateScoringPort
from typevet_evals.psai_vision_consumer.http_accounting import DispatchAccounting


class ConsumerDispatchLedger(DispatchAccounting):
    """PSAI ledger with frozen 14/16/3/128/16 dispatch ceilings.

    Examples:
        ```python
        ledger = ConsumerDispatchLedger()
        ledger.before_judgment_dispatch()
        assert ledger.judgment_attempts == 1
        ```
    """


class _BudgetScoringPort:
    """Scoring port wrapper that enforces frozen consumer budgets.

    Attributes:
        _inner (CandidateScoringPort): Wrapped scoring port.
        _ledger (ConsumerDispatchLedger): Dispatch counters for the run.

    Examples:
        ```python
        from typevet_evals.psai_vision_consumer.dispatch import (
            ConsumerDispatchLedger,
            _BudgetScoringPort,
        )
        from typevet.testing import ScriptedScoringFake

        ledger = ConsumerDispatchLedger()
        _BudgetScoringPort(ScriptedScoringFake(logprobs={"True": -0.1}), ledger)
        ```
    """

    def __init__(
        self,
        inner: CandidateScoringPort,
        ledger: DispatchAccounting,
    ) -> None:
        self._inner = inner
        self._ledger = ledger

    def score_candidates(
        self, request: CandidateScoringRequest
    ) -> CandidateScoringResult:
        """Dispatch one scoring request under the frozen ceiling.

        Raises:
            ConsumerCallBudgetError: When re-raised from the ledger guard.

        Returns:
            Scoring result from the inner port.
        """
        self._ledger.before_scoring_dispatch()
        try:
            result = self._inner.score_candidates(request)
        except Exception:
            self._ledger.record_failed_attempt()
            raise
        self._ledger.record_scoring_success()
        return result


class _BudgetJudgmentPort:
    """Judgment port wrapper that enforces frozen consumer budgets.

    Attributes:
        _inner (JudgmentPort): Wrapped judgment port.
        _ledger (ConsumerDispatchLedger): Dispatch counters for the run.

    Examples:
        ```python
        from typevet_evals.psai_vision_consumer.dispatch import (
            ConsumerDispatchLedger,
            _BudgetJudgmentPort,
        )

        ledger = ConsumerDispatchLedger()
        assert _BudgetJudgmentPort.__doc__
        ```
    """

    def __init__(self, inner: JudgmentPort, ledger: DispatchAccounting) -> None:
        self._inner = inner
        self._ledger = ledger

    def judge(
        self,
        state: str | dict[str, Any] | list[Any],
        questions: Mapping[str, Question | Mapping[str, Any]],
        model: str,
        *,
        media: tuple[ImageInput, ...] | None = None,
        off_option_threshold: float | None = None,
    ) -> JudgmentResponse:
        """Dispatch one judgment call under the frozen ceiling.

        ``media`` and ``off_option_threshold`` go to the inner port unchanged.

        Raises:
            ConsumerCallBudgetError: When re-raised from the ledger guard.

        Returns:
            Judgment response from the inner port.
        """
        self._ledger.before_judgment_dispatch()
        try:
            response = self._inner.judge(
                state,
                questions,
                model,
                media=media,
                off_option_threshold=off_option_threshold,
            )
        except Exception:
            self._ledger.record_failed_attempt()
            raise
        self._ledger.record_judgment_success()
        return response


def wrap_scoring_port(
    inner: CandidateScoringPort,
    ledger: DispatchAccounting,
) -> CandidateScoringPort:
    """Return a scoring port that records dispatch attempts on ``ledger``.

    Args:
        inner: Port under test.
        ledger: Shared PSAI or variant attempt and success ledger.

    Returns:
        Wrapped port enforcing the frozen scoring ceiling.
    """
    return _BudgetScoringPort(inner, ledger)


def wrap_judgment_port(
    inner: JudgmentPort,
    ledger: DispatchAccounting,
) -> JudgmentPort:
    """Return a judgment port that records dispatch attempts on ``ledger``.

    Args:
        inner: Port under test.
        ledger: Shared PSAI or variant attempt and success ledger.

    Returns:
        Wrapped port enforcing the frozen judgment ceiling.
    """
    return _BudgetJudgmentPort(inner, ledger)


def counting_tokenizer(
    client_post: Callable[..., Any],
    *,
    ledger: DispatchAccounting,
    model: str,
) -> Callable[[str], tuple[int, ...]]:
    """Build a tokenize closure that counts auxiliary HTTP on ``ledger``.

    Args:
        client_post: Bound ``httpx.Client.post`` used for ``/tokenize``.
        ledger: Shared attempt ledger receiving tokenizer HTTP reservations.
        model: Model id sent to the router.

    Returns:
        Callable matching ``ScoringJudgmentAdapter`` tokenize hook shape.
    """

    def tokenize_content(text: str) -> tuple[int, ...]:
        """Tokenize one prompt string through the router.

        Returns:
            Token id tuple from the router ``/tokenize`` endpoint.
        """
        ledger.before_tokenizer_http()
        payload = (
            client_post(
                "/tokenize",
                json={"model": model, "content": text, "add_special": False},
            )
            .raise_for_status()
            .json()
        )
        return tuple(payload["tokens"])

    return tokenize_content
