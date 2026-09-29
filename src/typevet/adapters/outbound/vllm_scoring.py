"""vLLM ``/v1/chat/completions`` adapter for pre-sampling candidate logprobs.

The vLLM server applies the served chat template, so the scoring prefix is
sent as the content of one user message. ``ChatContentFraming`` composes that
content without turn markers. The adapter asks for the requested token ids
through ``logprob_token_ids`` and reads their logprobs from the first
generated position. A request with images sends the content as a list of
``text`` and ``image_url`` blocks, one ``data:`` URI per image.

Examples:
    ```python
    from typevet.adapters.outbound.vllm_scoring import (
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
    - [typevet.adapters.outbound.llama_cpp_scoring][]: llama.cpp counterpart
    - [typevet.domain.candidate_scoring_validate][]: Fail-closed result assembly
    - [typevet.ports.framing][]: ModelFramingPort protocol
    - [typevet.ports.scoring][]: CandidateScoringPort protocol

Attributes:
    ID_FORM_PREFIX (str): Prefix of a token that vLLM returns as an id.
"""

from __future__ import annotations

import base64
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Self
from urllib.parse import urljoin

import httpx

from typevet.domain.candidate_scoring_validate import build_and_validate_result
from typevet.domain.errors import GenerationError, ScoringUnsupportedCapabilityError
from typevet.domain.judgment_response import TokenUsage
from typevet.domain.media import MEDIA_MARKER
from typevet.domain.scoring_stage import ScoreStage

if TYPE_CHECKING:
    from typevet.domain.candidate_scoring_request import CandidateScoringRequest
    from typevet.domain.candidate_scoring_response import CandidateScoringResult
    from typevet.domain.media import ImageInput

ID_FORM_PREFIX = "token_id:"


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
        """
        self._base_url = base_url.rstrip("/") + "/"
        self._timeout = timeout
        self._client = client
        self._owns_client = client is None

    def close(self) -> None:
        """Close the owned HTTP client when the adapter created it."""
        if self._owns_client and self._client is not None:
            self._client.close()
            self._client = None

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

        Raises:
            ScoringUnsupportedCapabilityError: Non-``PRE_SAMPLING`` stage or
                multi-token candidates.
            ScoringValidationError: Missing candidate scores or invalid logprob
                values (via ``build_and_validate_result``).
            GenerationError: When the body is not JSON or its shape is not usable.
        """
        _ensure_supported(request)
        url = urljoin(self._base_url, "v1/chat/completions")
        response = self._ensure_client().post(url, json=_request_body(request))
        try:
            payload = response.json()
        except ValueError as exc:
            msg = "vLLM chat completions response is not valid JSON"
            raise GenerationError(msg) from exc
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
        )

    def _ensure_client(self) -> httpx.Client:
        """Return the HTTP client, creating one when needed.

        Returns:
            An open ``httpx.Client``.
        """
        if self._client is None:
            self._client = httpx.Client(timeout=self._timeout)
        return self._client


def _ensure_supported(request: CandidateScoringRequest) -> None:
    """Refuse asks this adapter cannot score, before any HTTP call.

    Args:
        request: The scoring ask.

    Raises:
        ScoringUnsupportedCapabilityError: Non-``PRE_SAMPLING`` stage or
            multi-token candidates.
    """
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


def _content_blocks(prefix: str, media: tuple[ImageInput, ...]) -> list[dict[str, Any]]:
    """Split a prefix at its media markers into chat content blocks.

    Each marker becomes one ``image_url`` block, in ``media`` order. The text
    between markers becomes ``text`` blocks, verbatim except that one newline
    directly after a marker is dropped. Empty text blocks are left out.

    Args:
        prefix: Scoring prefix that holds one ``MEDIA_MARKER`` per image.
        media: Images the markers stand for, in marker order.

    Returns:
        Content blocks in prefix order.
    """
    parts = prefix.split(MEDIA_MARKER)
    blocks: list[dict[str, Any]] = []
    if parts[0]:
        blocks.append({"type": "text", "text": parts[0]})
    for image, part in zip(media, parts[1:], strict=True):
        encoded = base64.b64encode(image.data).decode("ascii")
        url = f"data:{image.mime_type};base64,{encoded}"
        blocks.append({"type": "image_url", "image_url": {"url": url}})
        text = part.removeprefix("\n")
        if text:
            blocks.append({"type": "text", "text": text})
    return blocks


def _request_body(request: CandidateScoringRequest) -> dict[str, Any]:
    """Build the chat completions body for one scoring ask.

    Args:
        request: A supported scoring ask; its media marker count matches
            ``request.media``, as the domain requires.

    Returns:
        JSON body that asks for the logprobs of the requested token ids.
    """
    content: str | list[dict[str, Any]] = request.prefix
    if request.media:
        content = _content_blocks(request.prefix, request.media)
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
        Mapping from vocabulary token id to raw logprob. The first entry for
        an id is kept.

    Raises:
        GenerationError: When an entry is not an object, lacks ``token`` or
            ``logprob``, or has a token not in ``token_id:N`` form.
    """
    result: dict[int, float] = {}
    for item in _top_logprobs(payload):
        if not isinstance(item, dict):
            msg = "vLLM top_logprobs entries must be objects"
            raise GenerationError(msg)
        try:
            token = str(item["token"])
            logprob = float(item["logprob"])
        except (KeyError, TypeError, ValueError) as exc:
            msg = "vLLM top_logprobs entry missing token or logprob"
            raise GenerationError(msg) from exc
        digits = token.removeprefix(ID_FORM_PREFIX)
        if digits == token or not digits.isdecimal():
            msg = (
                f"vLLM top_logprobs token {token!r} is not in {ID_FORM_PREFIX}N "
                "form; send return_tokens_as_token_ids"
            )
            raise GenerationError(msg)
        result.setdefault(int(digits), logprob)
    return result
