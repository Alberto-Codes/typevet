"""vLLM ``/v1/chat/completions`` adapter for pre-sampling candidate logprobs.

The vLLM server applies the served chat template, so the scoring prefix is
sent as the content of one user message. ``ChatContentFraming`` composes that
content without turn markers. The adapter asks for the requested token ids
through ``logprob_token_ids`` and reads their logprobs from the first
generated position. A request with images sends the content as a list of
``text`` and ``image_url`` blocks, one ``data:`` URI per image. The response
holds at most ``MAX_LOGPROB_TOKEN_IDS`` ids, not the full distribution, so the
result reports ``off_option_mass`` as ``None`` (unavailable). An adapter
without a caller client creates one client on first use, safely across
threads.

Examples:
    ```python
    from typevet.adapters.outbound.vllm.scoring import (
        VllmCandidateScoringAdapter,
    )
    from typevet.domain.candidate_scoring_request import (
        CandidateScoringRequest,
        CandidateTokenSpec,
    )

    request = CandidateScoringRequest(
        model="served-model",
        prefix="Answer:",
        candidates=(CandidateTokenSpec("a", (42,)),),
    )
    with VllmCandidateScoringAdapter("http://127.0.0.1:8000") as port:
        port.score_candidates(request)
    ```

See Also:
    - [typevet.adapters.outbound.llama_cpp.scoring][]: llama.cpp counterpart
    - [typevet.adapters.outbound.vllm.content][]: Image content blocks
    - [typevet.domain.candidate_scoring_validate][]: Fail-closed result assembly
    - [typevet.ports.framing][]: ModelFramingPort protocol
    - [typevet.ports.scoring][]: CandidateScoringPort protocol

Attributes:
    ID_FORM_PREFIX (str): Prefix of a token that vLLM returns as an id.
    MAX_LOGPROB_TOKEN_IDS (int): Largest ``logprob_token_ids`` list vLLM takes.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Self
from urllib.parse import urljoin

import httpx

from typevet.adapters.outbound.vllm.content import content_blocks
from typevet.adapters.outbound.vllm.http_mapping import post_json
from typevet.domain.candidate_scoring_validate import build_and_validate_result
from typevet.domain.errors import (
    GenerationError,
    ScoringUnsupportedCapabilityError,
    ScoringValidationError,
)
from typevet.domain.judgment_response import TokenUsage
from typevet.domain.scoring_stage import ScoreStage

if TYPE_CHECKING:
    from typevet.domain.candidate_scoring_request import CandidateScoringRequest
    from typevet.domain.candidate_scoring_response import CandidateScoringResult
    from typevet.domain.media import ImageInput

ID_FORM_PREFIX = "token_id:"
MAX_LOGPROB_TOKEN_IDS = 128


class ChatContentFraming:
    """Compose a scoring prefix as plain chat content, without turn markers.

    The server applies the chat template, so this framing only joins the
    context and the field block. Media markers already in the context stay
    in place.

    Examples:
        ```python
        ChatContentFraming().compose_prefix(context="c", field_block="f", media=())
        ```
    """

    def compose_prefix(
        self,
        *,
        context: str,
        field_block: str,
        media: tuple[ImageInput, ...],
    ) -> str:
        """Join the context and the field block with one blank line.

        Args:
            context: Rendered state context; holds one media marker per image.
            field_block: Rendered field instructions.
            media: Images the context marks; not changed by this framing.

        Returns:
            The context, one blank line, then the field block.
        """
        del media
        return f"{context}\n\n{field_block}"


class VllmCandidateScoringAdapter:
    """Score single-token candidates via vLLM chat completions logprobs.

    Attributes:
        _base_url (str): Server root with trailing slash.
        _timeout (float): HTTP timeout in seconds.
        _client (httpx.Client | None): Shared or owned HTTP client.
        _owns_client (bool): Whether ``close`` should close the client.
        _client_lock (threading.Lock): Guards lazy creation of the owned
            client.

    Examples:
        ```python
        VllmCandidateScoringAdapter(base_url="http://127.0.0.1:8000")
        ```
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8000",
        *,
        timeout: float = 300.0,
        client: httpx.Client | None = None,
    ) -> None:
        """Create the scoring adapter.

        Args:
            base_url: vLLM server root URL.
            timeout: Request timeout in seconds for an owned client.
            client: Optional caller-built httpx client, for example one that
                carries authentication headers. The adapter does not close it.
                ``None`` makes the adapter create its own client on first
                use, under a lock, so threads share one client.
        """
        self._base_url = base_url.rstrip("/") + "/"
        self._timeout = timeout
        self._client = client
        self._owns_client = client is None
        self._client_lock = threading.Lock()

    def close(self) -> None:
        """Close the owned HTTP client when the adapter created it.

        The client is detached and closed under ``_client_lock``, so a racing
        ``_ensure_client`` never gets the closing client and builds at most
        one new client. An injected client stays open (#345).
        """
        if not self._owns_client:
            return
        with self._client_lock:
            client, self._client = self._client, None
            if client is not None:
                client.close()

    def __enter__(self) -> Self:
        """Enter a context that closes the owned client on exit."""
        return self

    def __exit__(self, *_exc: object) -> None:
        """Close resources."""
        self.close()

    def score_candidates(
        self, request: CandidateScoringRequest
    ) -> CandidateScoringResult:
        """POST ``/v1/chat/completions`` and map token-id logprobs to labels.

        Args:
            request: Model id, prefix, ordered single-token candidates and stage.

        Returns:
            Validated ``CandidateScoringResult`` with raw logprobs per label.
            ``off_option_mass`` is always ``None``: the response does not hold
            the full distribution (#297).

        Raises:
            ScoringUnsupportedCapabilityError: Non-``PRE_SAMPLING`` stage,
                multi-token candidates or more than ``MAX_LOGPROB_TOKEN_IDS``
                candidates.
            ScoringValidationError: A duplicate ``token_id:N`` entry, missing
                candidate scores or invalid logprob values.
            TransportError: When the HTTP client fails.
            BackendHttpError: When vLLM returns status 400 or above.
            GenerationError: When the body is not JSON or its shape is not usable.
        """
        _ensure_supported(request)
        url = urljoin(self._base_url, "v1/chat/completions")
        payload = post_json(self._ensure_client(), url, _request_body(request))
        token_logprobs = _extract_token_logprobs(payload)
        raw_by_label = {
            spec.label: token_logprobs[spec.token_ids[0]]
            for spec in request.candidates
            if spec.token_ids[0] in token_logprobs
        }
        return build_and_validate_result(
            request,
            raw_logprobs=raw_by_label,
            model=request.model,
            usage=_extract_usage(payload),
            off_option_mass=None,
        )

    def _ensure_client(self) -> httpx.Client:
        """Return the HTTP client, creating one when needed.

        Creation is double-checked under ``_client_lock``, so threads that
        race here share one client. The lock is never held across a request.

        Returns:
            An open ``httpx.Client``.
        """
        client = self._client
        if client is None:
            with self._client_lock:
                client = self._client
                if client is None:
                    client = httpx.Client(timeout=self._timeout)
                    self._client = client
        return client


def _ensure_supported(request: CandidateScoringRequest) -> None:
    """Refuse asks this adapter cannot score, before any HTTP call.

    Args:
        request: The scoring ask.

    Raises:
        ScoringUnsupportedCapabilityError: Non-``PRE_SAMPLING`` stage,
            multi-token candidates or more than ``MAX_LOGPROB_TOKEN_IDS``
            candidates.
    """
    if len(request.candidates) > MAX_LOGPROB_TOKEN_IDS:
        msg = (
            f"vLLM accepts at most {MAX_LOGPROB_TOKEN_IDS} logprob_token_ids; "
            f"got {len(request.candidates)} candidates"
        )
        raise ScoringUnsupportedCapabilityError(msg)
    if request.stage is not ScoreStage.PRE_SAMPLING:
        msg = (
            "VllmCandidateScoringAdapter supports PRE_SAMPLING only; "
            f"got {request.stage!r}"
        )
        raise ScoringUnsupportedCapabilityError(msg)
    for spec in request.candidates:
        if len(spec.token_ids) != 1:
            msg = (
                "VllmCandidateScoringAdapter supports single-token candidates "
                f"only; {spec.label!r} has {len(spec.token_ids)} ids"
            )
            raise ScoringUnsupportedCapabilityError(msg)


def _request_body(request: CandidateScoringRequest) -> dict[str, Any]:
    """Build the chat completions body for one scoring ask.

    A request with images sends content blocks from ``content_blocks``.

    Args:
        request: A supported scoring ask; its media marker count matches
            ``request.media``, as the domain requires.

    Returns:
        JSON body that asks for the logprobs of the requested token ids.
    """
    content: str | list[dict[str, Any]] = request.prefix
    if request.media:
        content = content_blocks(request.prefix, request.media)
    return {
        "model": request.model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 1,
        "temperature": 0,
        "logprobs": True,
        "logprob_token_ids": [spec.token_ids[0] for spec in request.candidates],
        "return_tokens_as_token_ids": True,
        "add_generation_prompt": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def _non_negative_int(value: Any) -> int | None:
    """Coerce a vLLM token count to a non-negative int, else ``None``.

    Args:
        value: Raw field from the ``usage`` object.

    Returns:
        The count, or ``None`` when it is absent or not a non-negative int.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value >= 0 else None


def _extract_usage(payload: Mapping[str, Any]) -> TokenUsage:
    """Read prompt and completion token counts from the ``usage`` object.

    Args:
        payload: Parsed JSON object that ``_extract_token_logprobs`` accepted.

    Returns:
        ``TokenUsage`` with unknown counts left as ``None``.
    """
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        return TokenUsage()
    return TokenUsage(
        input_tokens=_non_negative_int(usage.get("prompt_tokens")),
        output_tokens=_non_negative_int(usage.get("completion_tokens")),
    )


def _top_logprobs(payload: Any) -> list[Any]:
    """Return ``choices[0].logprobs.content[0].top_logprobs``.

    Args:
        payload: Parsed JSON body.

    Returns:
        The list of top logprob entries at the first generated position.

    Raises:
        GenerationError: When the root is not an object or a field is missing.
    """
    if not isinstance(payload, dict):
        msg = "vLLM chat completions response root must be an object"
        raise GenerationError(msg)
    try:
        top = payload["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    except (KeyError, IndexError, TypeError) as exc:
        msg = "vLLM response missing choices[0].logprobs.content[0].top_logprobs"
        raise GenerationError(msg) from exc
    if not isinstance(top, list):
        msg = "vLLM top_logprobs must be a list"
        raise GenerationError(msg)
    return top


def _extract_token_logprobs(payload: Any) -> dict[int, float]:
    """Build ``token_id -> logprob`` from ``token_id:N`` top logprob entries.

    Args:
        payload: Parsed JSON body from vLLM chat completions.

    Returns:
        Mapping from vocabulary token id to raw logprob.

    Raises:
        GenerationError: When an entry is not an object, lacks ``token`` or
            ``logprob``, has a logprob that is not a real number, or has a
            token not in ``token_id:N`` form.
        ScoringValidationError: When two entries name the same token id.
    """
    result: dict[int, float] = {}
    for item in _top_logprobs(payload):
        token_id, logprob = _parse_entry(item)
        if token_id in result:
            msg = f"vLLM top_logprobs holds a duplicate entry for token id {token_id}"
            raise ScoringValidationError(msg)
        result[token_id] = logprob
    return result


def _parse_entry(item: Any) -> tuple[int, float]:
    """Read the token id and logprob from one ``top_logprobs`` entry.

    Args:
        item: One raw entry from ``top_logprobs``.

    Returns:
        The vocabulary token id and its raw logprob.

    Raises:
        GenerationError: When the entry is not an object, lacks ``token`` or
            ``logprob``, has a logprob that is not a real number, or has a
            token not in ``token_id:N`` form.
    """
    if not isinstance(item, dict) or "token" not in item or "logprob" not in item:
        msg = "vLLM top_logprobs entry must be an object with token and logprob"
        raise GenerationError(msg)
    logprob = item["logprob"]
    if isinstance(logprob, bool) or not isinstance(logprob, int | float):
        msg = f"vLLM top_logprobs logprob {logprob!r} is not a real number"
        raise GenerationError(msg)
    token = str(item["token"])
    digits = token.removeprefix(ID_FORM_PREFIX)
    if digits == token or not digits.isdecimal():
        msg = (
            f"vLLM top_logprobs token {token!r} is not in {ID_FORM_PREFIX}N "
            "form; send return_tokens_as_token_ids"
        )
        raise GenerationError(msg)
    return int(digits), float(logprob)
