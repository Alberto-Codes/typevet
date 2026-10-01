"""Record the request id that each vLLM judgment question sends (#356).

``open_vllm_judgment`` wraps its port in ``RequestIdJudgmentPort`` and the
scoring port in ``RequestIdScoringPort``. ``ScoringJudgmentAdapter`` calls
the scoring port once per question, in question order. Each such call opens
one slot. ``VllmCandidateScoringAdapter`` calls ``record_request_id`` after
each POST, and the id goes into the open slot. The judgment port then maps
each question name to the last id in its slot.

The slots live in a ``ContextVar``, so concurrent ``judge`` calls on other
threads or tasks do not mix. A scoring wrapper that posts on another thread
records no id for that question. When no slot holds an id, for example
without a request-id header, the response keeps empty ``request_ids``.
typevet never logs a request id.

Examples:
    ```python
    from typevet.adapters.outbound.vllm.request_ids import RequestIdJudgmentPort

    port = RequestIdJudgmentPort(scoring_judgment_adapter)
    response = port.judge("state", questions, "served-model")
    response.request_ids  # {"q": "<32 hex characters>"} with the header set
    ```

See Also:
    - [typevet.adapters.outbound.vllm.judgment_factory][]: Wires both wrappers
    - [typevet.adapters.outbound.vllm.scoring][]: Records each scoring id
    - [typevet.adapters.outbound.vllm.http_mapping][]: ``post_json_traced``
    - [typevet.domain.judgment_response][]: ``JudgmentResponse.request_ids``
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import replace
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping

    from typevet.adapters.outbound.judgment_scoring import ScoringJudgmentAdapter
    from typevet.domain.candidate_scoring_request import CandidateScoringRequest
    from typevet.domain.candidate_scoring_response import CandidateScoringResult
    from typevet.domain.judgment_questions import Question
    from typevet.domain.judgment_response import JudgmentResponse
    from typevet.domain.media import ImageInput
    from typevet.ports.scoring import CandidateScoringPort

_Slots = list[list[str | None]]
_SLOTS: ContextVar[_Slots | None] = ContextVar("typevet_vllm_request_ids", default=None)


def record_request_id(request_id: str | None) -> None:
    """Add ``request_id`` to the open slot of the current ``judge`` call.

    Args:
        request_id: Id that a scoring POST sent, or ``None``.
    """
    slots = _SLOTS.get()
    if slots:
        slots[-1].append(request_id)


class RequestIdScoringPort:
    """Scoring port wrapper that opens one id slot per scoring call.

    Attributes:
        _port (CandidateScoringPort): Wrapped scoring port.

    Examples:
        ```python
        RequestIdScoringPort(VllmCandidateScoringAdapter(client=client))
        ```
    """

    def __init__(self, port: CandidateScoringPort) -> None:
        """Wrap ``port``.

        Args:
            port: Scoring port that ``ScoringJudgmentAdapter`` calls.
        """
        self._port = port

    def score_candidates(
        self, request: CandidateScoringRequest
    ) -> CandidateScoringResult:
        """Open a slot for this call, then score with the wrapped port.

        Args:
            request: Scoring ask for one question.

        Returns:
            The result of the wrapped port, unchanged.
        """
        slots = _SLOTS.get()
        if slots is not None:
            slots.append([])
        return self._port.score_candidates(request)


class RequestIdJudgmentPort:
    """Judgment port wrapper that sets ``request_ids`` on each response.

    Attributes:
        _port (ScoringJudgmentAdapter): Wrapped judgment adapter.

    Examples:
        ```python
        RequestIdJudgmentPort(scoring_judgment_adapter)
        ```
    """

    def __init__(self, port: ScoringJudgmentAdapter) -> None:
        """Wrap ``port``.

        Args:
            port: Judgment adapter whose scoring port is a
                ``RequestIdScoringPort``.
        """
        self._port = port

    def judge(
        self,
        state: str | dict[str, Any] | list[Any],
        questions: Mapping[str, Question | Mapping[str, Any]],
        model: str,
        *,
        media: tuple[ImageInput, ...] | None = None,
        off_option_threshold: float | None = None,
    ) -> JudgmentResponse:
        """Judge, then map each question name to its scoring request id.

        Args:
            state: Content under evaluation.
            questions: Named native questions.
            model: Served model name.
            media: Images to condition every scored field on, in order.
            off_option_threshold: Forwarded to the wrapped adapter.

        Returns:
            The wrapped response. Its ``request_ids`` maps each question name
            to the last id that its scoring sent. It stays empty when no
            question sent an id.
        """
        slots: _Slots = []
        token = _SLOTS.set(slots)
        try:
            response = self._port.judge(
                state,
                questions,
                model,
                media=media,
                off_option_threshold=off_option_threshold,
            )
        finally:
            _SLOTS.reset(token)
        ids = [slot[-1] if slot else None for slot in slots]
        if all(value is None for value in ids):
            return response
        return replace(response, request_ids=dict(zip(questions, ids, strict=True)))
