"""llama.cpp ``/completion`` adapter for pre-sampling candidate logprobs.

Examples:
    ```python
    from typevet.adapters.outbound.llama_cpp_scoring import (
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
    - [typevet.adapters.outbound.llama_cpp_http][]: Shared HTTP error mapping
    - [typevet.domain.candidate_scoring_validate][]: Fail-closed result assembly
    - [typevet.ports.scoring][]: CandidateScoringPort protocol

Attributes:
    DEFAULT_N_VOCAB (int): Default ``n_probs`` for Gemma 4 class vocab (262144).
"""

from __future__ import annotations

from typing import Any, Self
from urllib.parse import urljoin

import httpx

from typevet.adapters.outbound.llama_cpp_http import (
    ensure_success_status,
    map_transport_error,
    parse_json_response,
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


class LlamaCppCandidateScoringAdapter:
    """Score single-token candidates via llama.cpp pre-sampling ``/completion``.

    Attributes:
        _base_url (str): Router root with trailing slash.
        _timeout (float): HTTP timeout in seconds.
        _client (httpx.Client | None): Shared or owned HTTP client.
        _owns_client (bool): Whether ``close`` should close the client.
        _n_vocab (int): ``n_probs`` sent on each completion request.

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
    ) -> None:
        """Create the scoring adapter.

        Args:
            base_url: llama.cpp server root URL.
            timeout: Request timeout in seconds.
            client: Optional shared httpx client (tests inject a fake).
            n_vocab: Vocabulary size for full ``n_probs`` pre-sampling receipt.
        """
        self._base_url = base_url.rstrip("/") + "/"
        self._timeout = timeout
        self._client = client
        self._owns_client = client is None
        self._n_vocab = n_vocab

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

        Args:
            request: Model id, prefix, ordered single-token candidates, stage.

        Returns:
            Validated ``CandidateScoringResult`` with raw logprobs per label.

        Raises:
            ScoringUnsupportedCapabilityError: Non-``PRE_SAMPLING`` stage or
                multi-token candidates.
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
            "prompt": request.prefix,
            "model": request.model,
            "n_predict": 0,
            "n_probs": self._n_vocab,
            "temperature": 0,
            "top_k": 0,
            "top_p": 1,
            "post_sampling_probs": False,
            "stream": False,
        }
        url = urljoin(self._base_url, "completion")
        client = self._ensure_client()
        try:
            response = client.post(url, json=body)
        except httpx.HTTPError as exc:
            raise map_transport_error(exc) from exc

        ensure_success_status(response)
        payload = parse_json_response(response)
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
            usage=TokenUsage(),
        )

    def _ensure_client(self) -> httpx.Client:
        """Return the HTTP client, creating one when needed.

        Returns:
            An open ``httpx.Client``.
        """
        if self._client is None:
            self._client = httpx.Client(timeout=self._timeout)
        return self._client


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
