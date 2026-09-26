"""Dispatch ledger and budget-enforcing port wrappers ([#177][i177]).

Examples:
    ```python
    from typevet.evaluation.psai_vision_consumer_dispatch import (
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
    - [typevet.evaluation.psai_vision_consumer_accounting][]: frozen totals
    - [typevet.evaluation.psai_vision_consumer_protocol][]: ceilings

[i177]: https://github.com/Alberto-Codes/typevet/issues/177
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from typevet.domain.candidate_scoring_request import CandidateScoringRequest
from typevet.domain.candidate_scoring_response import CandidateScoringResult
from typevet.domain.judgment_questions import Question
from typevet.domain.judgment_response import JudgmentResponse
from typevet.domain.media import ImageInput
from typevet.evaluation.psai_vision_consumer_accounting import ConsumerCallBudgetError
from typevet.evaluation.psai_vision_consumer_protocol import (
    FROZEN_AUXILIARY_METADATA_HTTP,
    FROZEN_AUXILIARY_TOKENIZER_HTTP_CEILING,
    FROZEN_JUDGMENT_CALLS,
    FROZEN_SCORING_REQUESTS,
)
from typevet.ports.judgment import JudgmentPort
from typevet.ports.scoring import CandidateScoringPort


@dataclass(slots=True)
class ConsumerDispatchLedger:
    """Observed judgment, scoring and auxiliary HTTP for one consumer run.

    Attributes:
        judgment_calls (int): ``judge`` invocations completed.
        scoring_requests (int): ``score_candidates`` calls completed.
        auxiliary_metadata_http (int): Health, capability and template probes.
        auxiliary_tokenizer_http (int): Router ``/tokenize`` calls.
        failed_attempts (int): Judgment or probe attempts that raised.

    Examples:
        ```python
        ledger = ConsumerDispatchLedger()
        ledger.before_judgment_dispatch()
        ```
    """

    judgment_calls: int = 0
    scoring_requests: int = 0
    auxiliary_metadata_http: int = 0
    auxiliary_tokenizer_http: int = 0
    failed_attempts: int = 0

    @property
    def auxiliary_http_total(self) -> int:
        """Return metadata plus tokenizer auxiliary HTTP."""
        return self.auxiliary_metadata_http + self.auxiliary_tokenizer_http

    def before_metadata_http(self) -> None:
        """Reserve one metadata auxiliary HTTP slot.

        Raises:
            ConsumerCallBudgetError: When metadata auxiliary would exceed budget.
        """
        next_count = self.auxiliary_metadata_http + 1
        if next_count > FROZEN_AUXILIARY_METADATA_HTTP:
            msg = (
                f"auxiliary metadata HTTP {next_count} "
                f"exceed budget {FROZEN_AUXILIARY_METADATA_HTTP}"
            )
            raise ConsumerCallBudgetError(msg)
        self.auxiliary_metadata_http = next_count

    def before_tokenizer_http(self) -> None:
        """Reserve one tokenizer auxiliary HTTP slot.

        Raises:
            ConsumerCallBudgetError: When tokenizer auxiliary would exceed budget.
        """
        next_count = self.auxiliary_tokenizer_http + 1
        if next_count > FROZEN_AUXILIARY_TOKENIZER_HTTP_CEILING:
            msg = (
                f"auxiliary tokenizer HTTP {next_count} "
                f"exceed budget {FROZEN_AUXILIARY_TOKENIZER_HTTP_CEILING}"
            )
            raise ConsumerCallBudgetError(msg)
        self.auxiliary_tokenizer_http = next_count

    def before_judgment_dispatch(self) -> None:
        """Reserve one judgment dispatch slot.

        Raises:
            ConsumerCallBudgetError: When judgment calls would exceed budget.
        """
        next_count = self.judgment_calls + 1
        if next_count > FROZEN_JUDGMENT_CALLS:
            msg = f"judgment_calls {next_count} exceed budget {FROZEN_JUDGMENT_CALLS}"
            raise ConsumerCallBudgetError(msg)

    def before_scoring_dispatch(self) -> None:
        """Reserve one scoring dispatch slot.

        Raises:
            ConsumerCallBudgetError: When scoring requests would exceed budget.
        """
        next_count = self.scoring_requests + 1
        if next_count > FROZEN_SCORING_REQUESTS:
            msg = (
                f"scoring_requests {next_count} exceed budget {FROZEN_SCORING_REQUESTS}"
            )
            raise ConsumerCallBudgetError(msg)

    def record_judgment_success(self) -> None:
        """Increment judgment count after a successful ``judge``."""
        self.judgment_calls += 1

    def record_scoring_success(self) -> None:
        """Increment scoring count after a successful ``score_candidates``."""
        self.scoring_requests += 1

    def record_failed_attempt(self) -> None:
        """Increment failed attempts after a caught dispatch error."""
        self.failed_attempts += 1


class _BudgetScoringPort:
    """Scoring port wrapper that enforces frozen consumer budgets.

    Attributes:
        _inner (CandidateScoringPort): Wrapped scoring port.
        _ledger (ConsumerDispatchLedger): Dispatch counters for the run.

    Examples:
        ```python
        from typevet.evaluation.psai_vision_consumer_dispatch import (
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
        ledger: ConsumerDispatchLedger,
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
        from typevet.evaluation.psai_vision_consumer_dispatch import (
            ConsumerDispatchLedger,
            _BudgetJudgmentPort,
        )

        ledger = ConsumerDispatchLedger()
        assert _BudgetJudgmentPort.__doc__
        ```
    """

    def __init__(self, inner: JudgmentPort, ledger: ConsumerDispatchLedger) -> None:
        self._inner = inner
        self._ledger = ledger

    def judge(
        self,
        state: str | dict[str, Any] | list[Any],
        questions: Mapping[str, Question | Mapping[str, Any]],
        model: str,
        *,
        media: tuple[ImageInput, ...] | None = None,
    ) -> JudgmentResponse:
        """Dispatch one judgment call under the frozen ceiling.

        Raises:
            ConsumerCallBudgetError: When re-raised from the ledger guard.

        Returns:
            Judgment response from the inner port.
        """
        self._ledger.before_judgment_dispatch()
        try:
            response = self._inner.judge(state, questions, model, media=media)
        except Exception:
            self._ledger.record_failed_attempt()
            raise
        self._ledger.record_judgment_success()
        return response


def wrap_scoring_port(
    inner: CandidateScoringPort,
    ledger: ConsumerDispatchLedger,
) -> CandidateScoringPort:
    """Return a scoring port that records dispatch attempts on ``ledger``.

    Args:
        inner: Port under test.
        ledger: Mutable dispatch ledger for the run.

    Returns:
        Wrapped port enforcing the frozen scoring ceiling.
    """
    return _BudgetScoringPort(inner, ledger)


def wrap_judgment_port(
    inner: JudgmentPort,
    ledger: ConsumerDispatchLedger,
) -> JudgmentPort:
    """Return a judgment port that records dispatch attempts on ``ledger``.

    Args:
        inner: Port under test.
        ledger: Mutable dispatch ledger for the run.

    Returns:
        Wrapped port enforcing the frozen judgment ceiling.
    """
    return _BudgetJudgmentPort(inner, ledger)


def counting_tokenizer(
    client_post: Callable[..., Any],
    *,
    ledger: ConsumerDispatchLedger,
    model: str,
) -> Callable[[str], tuple[int, ...]]:
    """Build a tokenize closure that counts auxiliary HTTP on ``ledger``.

    Args:
        client_post: Bound ``httpx.Client.post`` used for ``/tokenize``.
        ledger: Dispatch ledger receiving tokenizer counts.
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
