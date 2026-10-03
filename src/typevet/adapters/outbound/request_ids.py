"""Record the request id that each judgment question sends (#356, #411).

Both serving backends share this module, so the "Serving backends stay
independent" contract holds. A client request hook calls
``stamp_request_id``: it sets a fresh UUID4 hex value in the configured
request-id header when the request does not have one, and records the sent
value in ``request.extensions`` under ``REQUEST_ID_EXTENSION``.
``request_id_of`` reads that value back.

``open_vllm_judgment`` and ``open_gemma_native_vision_judgment`` wrap their
port in ``RequestIdJudgmentPort`` and the scoring port in
``RequestIdScoringPort``. ``ScoringJudgmentAdapter`` calls the scoring port
once per question, in question order. Each such call opens one slot. The
vLLM and llama.cpp scoring adapters call ``record_request_id`` after each
scoring POST, and the id goes into the open slot. The judgment port then
maps each question name to the last id in its slot.

The slots live in a ``ContextVar``, so concurrent ``judge`` calls on other
threads or tasks do not mix. A scoring wrapper that posts on another thread
records no id for that question. When no slot holds an id, for example
without a request-id header, the response keeps empty ``request_ids``.
typevet never logs a request id.

Attributes:
    REQUEST_ID_EXTENSION (str): ``request.extensions`` key of the request id.

Examples:
    ```python
    from typevet.adapters.outbound.request_ids import RequestIdJudgmentPort

    port = RequestIdJudgmentPort(scoring_judgment_adapter)
    response = port.judge("state", questions, "served-model")
    response.request_ids  # {"q": "<32 hex characters>"} with the header set
    ```

See Also:
    - [typevet.adapters.outbound.vllm.judgment_factory][]: Wires both wrappers
    - [typevet.adapters.outbound.vllm.scoring][]: Records each vLLM scoring id
    - [typevet.adapters.outbound.llama_cpp.scoring][]: Records each
      ``/completion`` id
    - [typevet.adapters.inbound.gateway_headers][]: vLLM request hook
    - [typevet.adapters.inbound.settings][]: llama.cpp request hook
    - [typevet.domain.judgment_response][]: ``JudgmentResponse.request_ids``
"""

from __future__ import annotations

import uuid
from contextvars import ContextVar
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:
    from collections.abc import Mapping

    import httpx

    from typevet.adapters.outbound.judgment_scoring import ScoringJudgmentAdapter
    from typevet.domain.candidate_scoring_request import CandidateScoringRequest
    from typevet.domain.candidate_scoring_response import CandidateScoringResult
    from typevet.domain.judgment_questions import Question
    from typevet.domain.judgment_response import JudgmentResponse
    from typevet.domain.media import ImageInput
    from typevet.ports.scoring import CandidateScoringPort

REQUEST_ID_EXTENSION: Final[str] = "typevet.request_id"

_Slots = list[list[str | None]]
_SLOTS: ContextVar[_Slots | None] = ContextVar("typevet_request_ids", default=None)


def stamp_request_id(request: httpx.Request, header: str | None) -> None:
    """Set the request-id header when it is configured and absent.

    The sent value is recorded in ``request.extensions`` under
    ``REQUEST_ID_EXTENSION``. A value that the caller set is kept.

    Args:
        request: Outgoing request.
        header: Header that gets a fresh UUID4 hex value, or ``None`` to do
            nothing.
    """
    if header is None:
        return
    if header not in request.headers:
        request.headers[header] = uuid.uuid4().hex
    request.extensions[REQUEST_ID_EXTENSION] = request.headers[header]


def request_id_of(request: httpx.Request) -> str | None:
    """Return the request id that the request hook recorded on ``request``.

    Args:
        request: Outgoing request.

    Returns:
        The id under ``REQUEST_ID_EXTENSION``, or ``None`` when no hook set
        one.
    """
    value = request.extensions.get(REQUEST_ID_EXTENSION)
    return value if isinstance(value, str) else None


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
            off_option_threshold: Forwarded to the wrapped adapter only when
                set, so a wrapped ``judge`` without that keyword still works.

        Returns:
            The wrapped response. Its ``request_ids`` maps each question name
            to the last id that its scoring sent. It stays empty when no
            question sent an id.
        """
        options: dict[str, Any] = {"media": media}
        if off_option_threshold is not None:
            options["off_option_threshold"] = off_option_threshold
        slots: _Slots = []
        token = _SLOTS.set(slots)
        try:
            response = self._port.judge(state, questions, model, **options)
        finally:
            _SLOTS.reset(token)
        ids = [slot[-1] if slot else None for slot in slots]
        if all(value is None for value in ids):
            return response
        return replace(response, request_ids=dict(zip(questions, ids, strict=True)))
