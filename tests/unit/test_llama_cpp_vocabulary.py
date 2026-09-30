"""Unit tests for the llama.cpp model vocabulary read (#321).

The scoring adapter reads ``meta.n_vocab`` of the model's ``/v1/models`` entry
once, and uses it for ``n_probs`` and the off-option completeness count.
"""

from __future__ import annotations

import json
import math
import threading
import time
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from typevet.adapters.outbound.llama_cpp.scoring import (
    DEFAULT_N_VOCAB,
    LlamaCppCandidateScoringAdapter,
)
from typevet.domain.candidate_scoring_request import (
    CandidateScoringRequest,
    CandidateTokenSpec,
)

_MODEL = "gemma-test"


def _request() -> CandidateScoringRequest:
    return CandidateScoringRequest(
        model=_MODEL,
        prefix="Answer:",
        candidates=(CandidateTokenSpec("a", (0,)), CandidateTokenSpec("b", (1,))),
    )


def _completion_bytes(count: int, *, missing_mass: float = 0.0) -> bytes:
    """Return a ``/completion`` body with ``count`` entries.

    Token 0 has probability 0.5, token 1 has 0.3, and the rest share the
    remainder less ``missing_mass``.
    """
    rest = (0.2 - missing_mass) / (count - 2)
    entries = [
        {"id": 0, "logprob": math.log(0.5)},
        {"id": 1, "logprob": math.log(0.3)},
    ]
    rest_logprob = math.log(rest)
    entries.extend({"id": i, "logprob": rest_logprob} for i in range(2, count))
    return json.dumps(
        {"completion_probabilities": [{"top_logprobs": entries}]}
    ).encode()


def _models_body(n_vocab: object, *, model_id: str = _MODEL) -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {"id": "other", "meta": {"n_vocab": 32000}},
            {"id": model_id, "aliases": ["alias-a"], "meta": {"n_vocab": n_vocab}},
        ],
    }


class _Router:
    """Mock router that answers ``/completion`` and ``/v1/models``."""

    def __init__(
        self,
        completion: Callable[[dict[str, Any]], bytes],
        models: Callable[[], httpx.Response],
    ) -> None:
        self.completion = completion
        self.models = models
        self.paths: list[str] = []
        self.n_probs: list[int] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.paths.append(request.url.path)
        if request.url.path == "/v1/models":
            return self.models()
        assert request.url.path == "/completion"
        body = json.loads(request.content)
        self.n_probs.append(body["n_probs"])
        return httpx.Response(
            200,
            content=self.completion(body),
            headers={"content-type": "application/json"},
        )

    def adapter(self, **kwargs: Any) -> LlamaCppCandidateScoringAdapter:
        client = httpx.Client(
            transport=httpx.MockTransport(self.handler), base_url="http://test"
        )
        return LlamaCppCandidateScoringAdapter(
            base_url="http://test", client=client, **kwargs
        )


def _capped_completion(model_vocab: int) -> Callable[[dict[str, Any]], bytes]:
    """Answer like llama.cpp: ``min(n_probs, model_vocab)`` entries."""
    cache: dict[int, bytes] = {}

    def answer(body: dict[str, Any]) -> bytes:
        count = min(body["n_probs"], model_vocab)
        if count not in cache:
            # A cut-short list misses the mass of the dropped tokens.
            missing = 1e-3 if count < model_vocab else 0.0
            cache[count] = _completion_bytes(count, missing_mass=missing)
        return cache[count]

    return answer


def _json(body: object, status: int = 200) -> Callable[[], httpx.Response]:
    return lambda: httpx.Response(status, json=body)


@pytest.mark.unit
def test_model_vocabulary_larger_than_the_budget_reports_unavailable() -> None:
    larger = DEFAULT_N_VOCAB + 1
    router = _Router(_capped_completion(larger), _json(_models_body(larger)))
    result = router.adapter().score_candidates(_request())
    assert router.n_probs == [DEFAULT_N_VOCAB]
    assert result.off_option_mass is None
    assert [row.logprob for row in result.candidates] == [
        math.log(0.5),
        math.log(0.3),
    ]


@pytest.mark.unit
def test_model_vocabulary_equal_to_the_response_count_reports_the_mass() -> None:
    router = _Router(_capped_completion(8), _json(_models_body(8)))
    result = router.adapter().score_candidates(_request())
    assert result.off_option_mass == pytest.approx(0.2, abs=1e-12)


@pytest.mark.unit
def test_vocabulary_is_read_once_and_sets_n_probs_on_later_calls() -> None:
    router = _Router(_capped_completion(10), _json(_models_body(10)))
    adapter = router.adapter()
    adapter.score_candidates(_request())
    second = adapter.score_candidates(_request())
    third = adapter.score_candidates(_request())
    assert router.paths.count("/v1/models") == 1
    assert router.n_probs == [DEFAULT_N_VOCAB, 10, 10]
    assert second.off_option_mass == pytest.approx(0.2, abs=1e-12)
    assert third.off_option_mass == pytest.approx(0.2, abs=1e-12)


@pytest.mark.unit
def test_larger_vocabulary_is_used_for_n_probs_after_the_first_read() -> None:
    larger = DEFAULT_N_VOCAB + 1
    router = _Router(_capped_completion(larger), _json(_models_body(larger)))
    adapter = router.adapter()
    first = adapter.score_candidates(_request())
    second = adapter.score_candidates(_request())
    assert router.n_probs == [DEFAULT_N_VOCAB, larger]
    assert first.off_option_mass is None
    assert second.off_option_mass == pytest.approx(0.2, abs=1e-9)


@pytest.mark.unit
def test_alias_match_reads_the_aliased_entry() -> None:
    router = _Router(_capped_completion(8), _json(_models_body(8, model_id="x")))
    request = CandidateScoringRequest(
        model="alias-a",
        prefix="Answer:",
        candidates=(CandidateTokenSpec("a", (0,)),),
    )
    result = router.adapter().score_candidates(request)
    assert result.off_option_mass == pytest.approx(0.5, abs=1e-12)


@pytest.mark.unit
def test_single_model_server_uses_its_only_entry() -> None:
    body = {"data": [{"id": "/models/file.gguf", "meta": {"n_vocab": 8}}]}
    router = _Router(_capped_completion(8), _json(body))
    result = router.adapter().score_candidates(_request())
    assert result.off_option_mass == pytest.approx(0.2, abs=1e-12)


@pytest.mark.unit
@pytest.mark.parametrize(
    "models",
    [
        _json(_models_body(None)),
        _json(_models_body(0)),
        _json(_models_body(True)),
        _json(_models_body("8")),
        _json({"data": [{"id": _MODEL}]}),
        _json({"data": [{"id": "a", "meta": {"n_vocab": 8}}, {"id": "b"}]}),
        _json({"data": "not-a-list"}),
        _json(["not-an-object"]),
        _json({"error": "boom"}, status=500),
        lambda: httpx.Response(200, content=b"not json"),
    ],
    ids=[
        "null",
        "zero",
        "bool",
        "string",
        "no-meta",
        "no-matching-entry",
        "data-not-list",
        "root-not-object",
        "http-500",
        "non-json",
    ],
)
def test_unknown_vocabulary_reports_unavailable_and_keeps_scores(
    models: Callable[[], httpx.Response],
) -> None:
    router = _Router(_capped_completion(8), models)
    result = router.adapter().score_candidates(_request())
    assert result.off_option_mass is None
    assert [row.label for row in result.candidates] == ["a", "b"]
    assert [row.logprob for row in result.candidates] == [
        math.log(0.5),
        math.log(0.3),
    ]


@pytest.mark.unit
def test_transport_failure_on_the_vocabulary_read_keeps_scores() -> None:
    def models() -> httpx.Response:
        raise httpx.ConnectError("refused")

    router = _Router(_capped_completion(8), models)
    result = router.adapter().score_candidates(_request())
    assert result.off_option_mass is None
    assert len(result.candidates) == 2


@pytest.mark.unit
def test_failed_read_with_a_full_default_sized_response_reports_unavailable() -> None:
    router = _Router(
        _capped_completion(DEFAULT_N_VOCAB), _json({"error": "boom"}, status=500)
    )
    result = router.adapter().score_candidates(_request())
    assert router.n_probs == [DEFAULT_N_VOCAB]
    assert result.off_option_mass is None


@pytest.mark.unit
def test_no_size_read_with_a_full_default_sized_response_reports_unavailable() -> None:
    router = _Router(
        _capped_completion(DEFAULT_N_VOCAB), _json({"data": [{"id": _MODEL}]})
    )
    result = router.adapter().score_candidates(_request())
    assert router.n_probs == [DEFAULT_N_VOCAB]
    assert result.off_option_mass is None


class _YieldingCounts(dict[str, int]):
    """Read-count dict whose ``get`` sleeps to widen a check-then-set race."""

    def get(self, key: object, default: Any = None, /) -> Any:
        value = super().get(key, default)
        time.sleep(0.02)
        return value


@pytest.mark.unit
def test_concurrent_first_calls_make_at_most_three_reads() -> None:
    threads_count = 10
    lock = threading.Lock()
    reads: list[str] = []

    def models() -> httpx.Response:
        with lock:
            reads.append("read")
        time.sleep(0.05)
        return httpx.Response(503, json={})

    router = _Router(_capped_completion(8), models)
    adapter = router.adapter()
    # Every thread reaches the read-count check before any thread updates it.
    adapter._vocab_reads = _YieldingCounts()
    barrier = threading.Barrier(threads_count)
    results: list[float | None] = []

    def score() -> None:
        barrier.wait()
        result = adapter.score_candidates(_request())
        with lock:
            results.append(result.off_option_mass)

    workers = [threading.Thread(target=score) for _ in range(threads_count)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()
    assert results == [None] * threads_count
    assert len(reads) <= 3


@pytest.mark.unit
def test_absent_vocabulary_is_read_again_on_the_next_call() -> None:
    responses = iter([_json({"data": [{"id": _MODEL}]})(), _json(_models_body(8))()])
    router = _Router(_capped_completion(8), lambda: next(responses))
    adapter = router.adapter()
    first = adapter.score_candidates(_request())
    second = adapter.score_candidates(_request())
    assert router.paths.count("/v1/models") == 2
    assert first.off_option_mass is None
    assert second.off_option_mass == pytest.approx(0.2, abs=1e-12)


@pytest.mark.unit
@pytest.mark.parametrize(
    "models",
    [_json({}, status=503), _json({"data": [{"id": _MODEL}]})],
    ids=["http-503", "no-size"],
)
def test_vocabulary_reads_stop_after_three_attempts(
    models: Callable[[], httpx.Response],
) -> None:
    router = _Router(_capped_completion(8), models)
    adapter = router.adapter()
    results = [adapter.score_candidates(_request()) for _ in range(5)]
    assert router.paths.count("/v1/models") == 3
    assert [result.off_option_mass for result in results] == [None] * 5


@pytest.mark.unit
def test_failed_vocabulary_read_is_tried_again_on_the_next_call() -> None:
    responses = iter([_json({}, status=503)(), _json(_models_body(8))()])
    router = _Router(_capped_completion(8), lambda: next(responses))
    adapter = router.adapter()
    first = adapter.score_candidates(_request())
    second = adapter.score_candidates(_request())
    assert router.paths.count("/v1/models") == 2
    assert first.off_option_mass is None
    assert second.off_option_mass == pytest.approx(0.2, abs=1e-12)


@pytest.mark.unit
def test_explicit_n_vocab_overrides_the_read() -> None:
    router = _Router(_capped_completion(8), _json(_models_body(99)))
    result = router.adapter(n_vocab=8).score_candidates(_request())
    assert router.paths == ["/completion"]
    assert router.n_probs == [8]
    assert result.off_option_mass == pytest.approx(0.2, abs=1e-12)
