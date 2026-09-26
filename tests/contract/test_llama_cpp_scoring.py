"""Contract tests for llama.cpp pre-sampling candidate scoring adapter."""

from __future__ import annotations

import json
import math

import httpx
import pytest

from typevet.adapters.outbound.llama_cpp_scoring import LlamaCppCandidateScoringAdapter
from typevet.domain.candidate_scoring_request import (
    CandidateScoringRequest,
    CandidateTokenSpec,
)
from typevet.domain.errors import (
    BackendHttpError,
    GenerationError,
    ScoringUnsupportedCapabilityError,
    ScoringValidationError,
    TransportError,
)
from typevet.domain.scoring_stage import ScoreStage

_DEFAULT_N_VOCAB = 262144


def _request(
    *,
    stage: ScoreStage = ScoreStage.PRE_SAMPLING,
    candidates: tuple[CandidateTokenSpec, ...] | None = None,
) -> CandidateScoringRequest:
    specs = candidates or (
        CandidateTokenSpec("billing", (101,)),
        CandidateTokenSpec("technical", (202,)),
    )
    return CandidateScoringRequest(
        model="gemma-test",
        prefix="Answer:",
        candidates=specs,
        stage=stage,
    )


def _completion_json(
    top_logprobs: list[dict[str, float | int]],
) -> dict[str, object]:
    return {
        "completion_probabilities": [
            {
                "top_logprobs": top_logprobs,
            }
        ]
    }


def _adapter(
    handler: httpx.MockTransport,
    *,
    n_vocab: int = _DEFAULT_N_VOCAB,
) -> LlamaCppCandidateScoringAdapter:
    client = httpx.Client(transport=handler, base_url="http://test")
    return LlamaCppCandidateScoringAdapter(
        base_url="http://test",
        client=client,
        n_vocab=n_vocab,
    )


@pytest.mark.contract
def test_llama_cpp_scoring_success() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/completion")
        body = json.loads(request.content.decode())
        captured["body"] = body
        assert body["prompt"] == "Answer:"
        assert body["n_predict"] == 0
        assert body["n_probs"] == _DEFAULT_N_VOCAB
        assert body["temperature"] == 0
        assert body["top_k"] == 0
        assert body["top_p"] == 1
        assert body["post_sampling_probs"] is False
        assert body["stream"] is False
        assert body["model"] == "gemma-test"
        return httpx.Response(
            200,
            json=_completion_json(
                [
                    {"id": 101, "logprob": -0.5},
                    {"id": 202, "logprob": -1.2},
                ]
            ),
        )

    adapter = _adapter(httpx.MockTransport(handler))
    result = adapter.score_candidates(_request())
    assert result.model == "gemma-test"
    assert [c.logprob for c in result.candidates] == [-0.5, -1.2]
    assert captured["body"] is not None


@pytest.mark.contract
def test_llama_cpp_scoring_omitted_low_ranked_candidate() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_completion_json([{"id": 101, "logprob": -0.1}]),
        )

    adapter = _adapter(httpx.MockTransport(handler))
    with pytest.raises(ScoringValidationError, match="missing scores"):
        adapter.score_candidates(_request())


@pytest.mark.contract
def test_llama_cpp_scoring_duplicate_top_logprob_token_id() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json=_completion_json(
                [
                    {"id": 101, "logprob": -0.5},
                    {"id": 101, "logprob": -9.0},
                    {"id": 202, "logprob": -1.0},
                ]
            ),
        )

    adapter = _adapter(httpx.MockTransport(handler))
    with pytest.raises(ScoringValidationError, match="duplicate token id"):
        adapter.score_candidates(_request())


@pytest.mark.contract
def test_llama_cpp_scoring_negative_infinity_logprob() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        body = json.dumps(
            _completion_json(
                [
                    {"id": 101, "logprob": float("-inf")},
                    {"id": 202, "logprob": -1.0},
                ]
            ),
            allow_nan=True,
        )
        return httpx.Response(
            200,
            content=body,
            headers={"content-type": "application/json"},
        )

    adapter = _adapter(httpx.MockTransport(handler))
    with pytest.raises(ScoringValidationError, match="non-finite"):
        adapter.score_candidates(_request())


@pytest.mark.contract
def test_llama_cpp_scoring_non_finite_logprob() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        body = json.dumps(
            _completion_json(
                [
                    {"id": 101, "logprob": float("nan")},
                    {"id": 202, "logprob": -1.0},
                ]
            ),
            allow_nan=True,
        )
        return httpx.Response(
            200,
            content=body,
            headers={"content-type": "application/json"},
        )

    adapter = _adapter(httpx.MockTransport(handler))
    with pytest.raises(ScoringValidationError, match="non-finite"):
        adapter.score_candidates(_request())


@pytest.mark.contract
def test_llama_cpp_scoring_timeout_maps_to_transport_error() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=_request)

    adapter = _adapter(httpx.MockTransport(handler))
    with pytest.raises(TransportError, match=r"llama\.cpp request failed"):
        adapter.score_candidates(_request())


@pytest.mark.contract
def test_llama_cpp_scoring_http_error() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="unavailable")

    adapter = _adapter(httpx.MockTransport(handler))
    with pytest.raises(BackendHttpError, match="HTTP 503") as exc_info:
        adapter.score_candidates(_request())
    assert exc_info.value.status_code == 503


@pytest.mark.contract
def test_llama_cpp_scoring_unsupported_stage() -> None:
    adapter = _adapter(httpx.MockTransport(lambda _r: httpx.Response(200, json={})))
    with pytest.raises(ScoringUnsupportedCapabilityError, match="PRE_SAMPLING"):
        adapter.score_candidates(_request(stage=ScoreStage.POST_SAMPLING))


@pytest.mark.contract
def test_llama_cpp_scoring_multi_token_candidate() -> None:
    request = _request(
        candidates=(CandidateTokenSpec("phrase", (1, 2)),),
    )
    adapter = _adapter(httpx.MockTransport(lambda _r: httpx.Response(200, json={})))
    with pytest.raises(ScoringUnsupportedCapabilityError, match="single-token"):
        adapter.score_candidates(request)


@pytest.mark.contract
def test_llama_cpp_scoring_bad_completion_probabilities_shape() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"completion_probabilities": []})

    adapter = _adapter(httpx.MockTransport(handler))
    with pytest.raises(GenerationError, match="completion_probabilities"):
        adapter.score_candidates(_request())


@pytest.mark.contract
def test_llama_cpp_scoring_missing_top_logprobs() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"completion_probabilities": [{}]},
        )

    adapter = _adapter(httpx.MockTransport(handler))
    with pytest.raises(GenerationError, match="top_logprobs"):
        adapter.score_candidates(_request())


@pytest.mark.contract
def test_llama_cpp_scoring_respects_custom_n_vocab() -> None:
    seen: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode())
        seen.append(int(body["n_probs"]))
        return httpx.Response(
            200,
            json=_completion_json(
                [
                    {"id": 101, "logprob": -0.5},
                    {"id": 202, "logprob": -1.0},
                ]
            ),
        )

    adapter = _adapter(httpx.MockTransport(handler), n_vocab=128256)
    adapter.score_candidates(_request())
    assert seen == [128256]
    assert math.isfinite(-0.5)
