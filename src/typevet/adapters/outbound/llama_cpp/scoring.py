"""llama.cpp ``/completion`` adapter for pre-sampling candidate logprobs.

The adapter asks for the full-vocabulary distribution and reports the
off-option mass, the probability outside the candidate tokens, on the result.

Examples:
    ```python
    from typevet.adapters.outbound.llama_cpp.scoring import (
        LlamaCppCandidateScoringAdapter,
    )
    from typevet.domain.candidate_scoring_request import (
        CandidateScoringRequest,
        CandidateTokenSpec,
    )

    request = CandidateScoringRequest(
        model="local",
        prefix="Answer:",
        candidates=(CandidateTokenSpec("a", (42,)),),
    )
    with LlamaCppCandidateScoringAdapter() as port:
        port.score_candidates(request)
    ```

See Also:
    - [typevet.adapters.outbound.llama_cpp.http_mapping][]: Shared HTTP error
      mapping and the one-retry send for idempotent requests
    - [typevet.adapters.outbound.llama_cpp.multimodal][]: Media probe and shaping
    - [typevet.domain.candidate_scoring_validate][]: Fail-closed result assembly
    - [typevet.ports.scoring][]: CandidateScoringPort protocol

Attributes:
    DEFAULT_N_VOCAB (int): Default ``n_probs`` for Gemma 4 class vocab (262144).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any, Self
from urllib.parse import urljoin

import httpx

from typevet.adapters.outbound.llama_cpp.http_mapping import (
    ensure_success_status,
    parse_json_response,
    send_idempotent,
)
from typevet.adapters.outbound.llama_cpp.multimodal import (
    MediaCapability,
    fetch_media_capability,
    media_prompt_field,
)
from typevet.domain.candidate_scoring_request import CandidateScoringRequest
from typevet.domain.candidate_scoring_response import CandidateScoringResult
from typevet.domain.candidate_scoring_validate import build_and_validate_result
from typevet.domain.errors import (
    GenerationError,
    ScoringUnsupportedCapabilityError,
    ScoringValidationError,
)
from typevet.domain.judgment_response import TokenUsage
from typevet.domain.scoring_stage import ScoreStage

DEFAULT_N_VOCAB = 262144

_FULL_DISTRIBUTION_TOLERANCE = 1e-2
"""Largest distance of the returned total mass from 1 that still counts as the
full distribution.

The model computes its softmax in float32, so the total of all entries drifts
from 1. One local call on the Q2_K Gemma 4 file returned 262144 entries with a
total of 1.0005991 (review, 2026-09-30). This margin is about 16 times that
drift. It is a sanity guard only: the entry count is the completeness check.
"""


class LlamaCppCandidateScoringAdapter:
    """Score single-token candidates via llama.cpp pre-sampling ``/completion``.

    Attributes:
        _base_url (str): Router root with trailing slash.
        _timeout (float): HTTP timeout in seconds.
        _client (httpx.Client | None): Shared or owned HTTP client.
        _owns_client (bool): Whether ``close`` should close the client.
        _n_vocab (int): ``n_probs`` sent on each completion request.
        _media_capabilities (dict[str, MediaCapability]): Per-model props cache.

    Examples:
        ```python
        LlamaCppCandidateScoringAdapter(base_url="http://127.0.0.1:8090")
        ```
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8090",
        *,
        timeout: float = 300.0,
        client: httpx.Client | None = None,
        n_vocab: int = DEFAULT_N_VOCAB,
        media_capabilities: Mapping[str, MediaCapability] | None = None,
    ) -> None:
        """Create the scoring adapter with an empty media capability cache.

        Args:
            base_url: llama.cpp server root URL.
            timeout: Request timeout in seconds.
            client: Optional shared httpx client (tests inject a fake).
            media_capabilities: Optional initial capability cache, copied on construction.
            n_vocab: Vocabulary size for full ``n_probs`` pre-sampling receipt.
        """
        self._base_url = base_url.rstrip("/") + "/"
        self._timeout = timeout
        self._client = client
        self._owns_client = client is None
        self._n_vocab = n_vocab
        self._media_capabilities: dict[str, MediaCapability] = dict(
            media_capabilities or {}
        )

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
        """POST ``/completion`` and map top logprobs to requested candidates.

        A request that carries ``media`` probes ``/props`` for the model's image
        support and marker, then sends the nested object prompt that attaches
        the images. A text request sends a plain string ``prompt`` and never
        probes ``/props``. Every request sends ``cache_prompt: false`` so a
        cached KV prefix cannot shift the scores (#154, #155). When the router
        closes the connection (new or reused) before a response head, the request is sent once
        more on a fresh connection (#305).

        Args:
            request: Model id, prefix, ordered single-token candidates, stage,
                and optional images.

        Returns:
            Validated ``CandidateScoringResult`` with raw logprobs per label.
            ``off_option_mass`` is the mass outside the candidate tokens when
            the response holds the full distribution, else ``None`` (#297).

        Raises:
            ScoringUnsupportedCapabilityError: Non-``PRE_SAMPLING`` stage,
                multi-token candidates, or images for a text-only model.
            ScoringValidationError: Missing candidate token coverage or invalid
                logprob values (via ``build_and_validate_result``).
            TransportError: When the HTTP client fails before a response.
            BackendHttpError: When llama.cpp returns HTTP status 400 or above.
            GenerationError: When the JSON body shape is not usable.
        """
        if request.stage is not ScoreStage.PRE_SAMPLING:
            msg = (
                "LlamaCppCandidateScoringAdapter supports PRE_SAMPLING only; "
                f"got {request.stage!r}"
            )
            raise ScoringUnsupportedCapabilityError(msg)
        for spec in request.candidates:
            if len(spec.token_ids) != 1:
                msg = (
                    "LlamaCppCandidateScoringAdapter supports single-token "
                    f"candidates only; {spec.label!r} has {len(spec.token_ids)} ids"
                )
                raise ScoringUnsupportedCapabilityError(msg)

        body: dict[str, Any] = {
            "prompt": self._prompt_field(request),
            "model": request.model,
            "n_predict": 0,
            "n_probs": self._n_vocab,
            "temperature": 0,
            "top_k": 0,
            "top_p": 1,
            "post_sampling_probs": False,
            "stream": False,
            "cache_prompt": False,
        }
        url = urljoin(self._base_url, "completion")
        client = self._ensure_client()
        response = send_idempotent(lambda: client.post(url, json=body))
        ensure_success_status(response)
        payload = parse_json_response(response)
        # Validates the root is an object, which _extract_usage then assumes.
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
            off_option_mass=_off_option_mass(
                token_logprobs, raw_by_label.values(), n_vocab=self._n_vocab
            ),
        )

    def _prompt_field(self, request: CandidateScoringRequest) -> Any:
        """Return the ``prompt`` body value for a text or media request.

        Args:
            request: The scoring ask, with or without images.

        Returns:
            The prefix string for a text ask, or the nested object prompt that
            attaches ``request.media``.

        Raises:
            ScoringUnsupportedCapabilityError: When the model declares no image
                input modality.
        """
        if not request.media:
            return request.prefix
        capability = self._media_capability(request.model)
        if not capability.vision:
            msg = (
                f"llama.cpp model {request.model!r} does not declare image input "
                "support; scoring with media is refused"
            )
            raise ScoringUnsupportedCapabilityError(msg)
        return media_prompt_field(
            request.prefix,
            request.media,
            marker=capability.marker,
        )

    def _media_capability(self, model: str) -> MediaCapability:
        """Return the cached media capability for ``model``, probing once.

        Args:
            model: Router model id.

        Returns:
            Declared ``MediaCapability`` for that model.
        """
        cached = self._media_capabilities.get(model)
        if cached is not None:
            return cached
        capability = fetch_media_capability(
            self._ensure_client(), self._base_url, model
        )
        self._media_capabilities[model] = capability
        return capability

    def _ensure_client(self) -> httpx.Client:
        """Return the HTTP client, creating one when needed.

        Returns:
            An open ``httpx.Client``.
        """
        if self._client is None:
            self._client = httpx.Client(timeout=self._timeout)
        return self._client


def _off_option_mass(
    token_logprobs: Mapping[int, float],
    candidate_logprobs: Iterable[float],
    *,
    n_vocab: int,
) -> float | None:
    """Return the probability mass outside the candidate tokens, or ``None``.

    llama.cpp sends log-softmax values over the full vocabulary, so the
    off-option mass is ``1 - sum(exp(logprob))`` over the raw candidate
    logprobs, clamped to ``[0, 1]``. No renormalization over the candidates
    occurs.

    The response is complete only when it holds exactly ``n_vocab`` entries.
    ``n_vocab`` is the vocabulary size the caller gave the adapter; the
    response does not report the model vocabulary. llama.cpp caps ``n_probs``
    at the model vocabulary, so a smaller model gives fewer entries and the
    value is ``None`` (unavailable), not a partial number. A model vocabulary
    larger than ``n_vocab`` also returns ``n_vocab`` entries and cannot be
    detected by the count. The sum guard then catches it only when the
    missing mass is above ``_FULL_DISTRIBUTION_TOLERANCE``. A total that is
    not within that margin of 1 also gives ``None``.

    Args:
        token_logprobs: Every ``token_id -> logprob`` entry the router returned.
        candidate_logprobs: Raw logprobs of the requested candidate tokens.
        n_vocab: Vocabulary size the adapter asked for as ``n_probs``.

    Returns:
        The off-option mass in ``[0, 1]``, or ``None`` when the response does
        not hold the full distribution.
    """
    if len(token_logprobs) != n_vocab:
        return None
    returned_mass = math.fsum(math.exp(value) for value in token_logprobs.values())
    if abs(returned_mass - 1.0) > _FULL_DISTRIBUTION_TOLERANCE:
        return None
    candidate_mass = math.fsum(math.exp(value) for value in candidate_logprobs)
    return min(1.0, max(0.0, 1.0 - candidate_mass))


def _non_negative_int(value: Any) -> int | None:
    """Coerce a llama.cpp token count to a non-negative int, else ``None``.

    Args:
        value: Raw field from the ``/completion`` body.

    Returns:
        The count, or ``None`` when the router omits it or reports a value that
        is not a non-negative whole number.
    """
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if value >= 0 else None


def _extract_usage(payload: Mapping[str, Any]) -> TokenUsage:
    """Read prompt and completion token counts from a ``/completion`` body.

    ``tokens_evaluated`` is the only signal that distinguishes an attached
    image from a silently dropped one: both return HTTP 200 with a valid
    logprob distribution, but a dropped image leaves the count at the text
    baseline (#155 recipe). Callers pass a body that
    ``_extract_token_logprobs`` already proved is an object.

    Args:
        payload: Parsed JSON object from llama.cpp ``/completion``.

    Returns:
        ``TokenUsage`` with unknown counts left as ``None``.
    """
    return TokenUsage(
        input_tokens=_non_negative_int(payload.get("tokens_evaluated")),
        output_tokens=_non_negative_int(payload.get("tokens_predicted")),
    )


def _extract_token_logprobs(payload: Any) -> dict[int, float]:
    """Build ``token_id -> logprob`` from ``completion_probabilities``.

    Args:
        payload: Parsed JSON object from llama.cpp ``/completion``.

    Returns:
        Mapping from vocabulary token id to raw logprob.

    Raises:
        GenerationError: When required fields are missing or malformed.
        ScoringValidationError: When the same token id appears more than once.
    """
    if not isinstance(payload, dict):
        msg = "llama.cpp completion response root must be an object"
        raise GenerationError(msg)
    try:
        completion_probabilities = payload["completion_probabilities"]
        first = completion_probabilities[0]
        top_logprobs = first["top_logprobs"]
    except (KeyError, IndexError, TypeError) as exc:
        msg = "llama.cpp response missing completion_probabilities[0].top_logprobs"
        raise GenerationError(msg) from exc
    if not isinstance(top_logprobs, list):
        msg = "llama.cpp top_logprobs must be a list"
        raise GenerationError(msg)
    result: dict[int, float] = {}
    for item in top_logprobs:
        if not isinstance(item, dict):
            msg = "llama.cpp top_logprobs entries must be objects"
            raise GenerationError(msg)
        try:
            token_id = int(item["id"])
            logprob = float(item["logprob"])
        except (KeyError, TypeError, ValueError) as exc:
            msg = "llama.cpp top_logprobs entry missing id or logprob"
            raise GenerationError(msg) from exc
        if token_id in result:
            msg = f"duplicate token id in top_logprobs: {token_id!r}"
            raise ScoringValidationError(msg)
        result[token_id] = logprob
    return result
