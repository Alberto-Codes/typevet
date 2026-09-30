"""Unit tests for the off-option mass on a candidate scoring result (#297)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import httpx
import pytest

from typevet.adapters.outbound.llama_cpp.scoring import LlamaCppCandidateScoringAdapter
from typevet.adapters.outbound.vllm.scoring import VllmCandidateScoringAdapter
from typevet.domain.candidate_scoring_request import (
    CandidateScoringRequest,
    CandidateTokenSpec,
)
from typevet.domain.candidate_scoring_response import (
    CandidateScoringResult,
    ScoredCandidate,
)
from typevet.domain.candidate_scoring_validate import (
    build_and_validate_result,
    validate_result_against_request,
)
from typevet.domain.errors import ScoringValidationError
from typevet.domain.scoring_stage import ScoreStage
from typevet.testing import ScriptedScoringFake

_VLLM_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "vllm"


def _request(*ids: int) -> CandidateScoringRequest:
    specs = tuple(
        CandidateTokenSpec(f"c{i}", (token_id,)) for i, token_id in enumerate(ids)
    )
    return CandidateScoringRequest(model="m", prefix="Answer:", candidates=specs)


def _llama_cpp(
    probs: dict[int, float], *, n_vocab: int, candidate_ids: tuple[int, ...]
) -> CandidateScoringResult:
    entries = [{"id": i, "logprob": math.log(p)} for i, p in probs.items()]
    payload = {"completion_probabilities": [{"top_logprobs": entries}]}

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    client = httpx.Client(
        transport=httpx.MockTransport(handler), base_url="http://test"
    )
    adapter = LlamaCppCandidateScoringAdapter(
        base_url="http://test", client=client, n_vocab=n_vocab
    )
    return adapter.score_candidates(_request(*candidate_ids))


def _candidate_mass(result: CandidateScoringResult) -> float:
    return sum(math.exp(row.logprob) for row in result.candidates)


@pytest.mark.unit
def test_llama_cpp_full_vocabulary_reports_off_option_mass() -> None:
    probs = {1: 0.4, 2: 0.25, 3: 0.2, 4: 0.1, 5: 0.05}
    result = _llama_cpp(probs, n_vocab=5, candidate_ids=(1, 2))
    assert result.off_option_mass == pytest.approx(0.35, abs=1e-12)
    assert result.off_option_mass is not None
    assert _candidate_mass(result) + result.off_option_mass <= 1.0 + 1e-12


@pytest.mark.unit
def test_llama_cpp_off_option_mass_uses_raw_not_renormalized_logprobs() -> None:
    probs = {1: 0.1, 2: 0.1, 3: 0.8}
    result = _llama_cpp(probs, n_vocab=3, candidate_ids=(1, 2))
    assert [row.logprob for row in result.candidates] == [math.log(0.1)] * 2
    assert result.off_option_mass == pytest.approx(0.8, abs=1e-12)


@pytest.mark.unit
def test_llama_cpp_partial_distribution_reports_unavailable() -> None:
    probs = {1: 0.4, 2: 0.25, 3: 0.2}
    result = _llama_cpp(probs, n_vocab=5, candidate_ids=(1, 2))
    assert result.off_option_mass is None
    assert [row.logprob for row in result.candidates] == [
        math.log(0.4),
        math.log(0.25),
    ]


def _with_total(total: float) -> dict[int, float]:
    """Return a 4-token distribution whose probabilities sum to ``total``."""
    return {1: 0.5, 2: 0.2, 3: 0.2, 4: total - 0.9}


@pytest.mark.unit
def test_llama_cpp_count_short_reports_unavailable_even_when_total_is_one() -> None:
    probs = {1: 0.5, 2: 0.3, 3: 0.2}
    result = _llama_cpp(probs, n_vocab=4, candidate_ids=(1, 2))
    assert result.off_option_mass is None


@pytest.mark.unit
def test_llama_cpp_count_above_vocabulary_reports_unavailable() -> None:
    probs = {1: 0.5, 2: 0.3, 3: 0.1, 4: 0.1}
    result = _llama_cpp(probs, n_vocab=3, candidate_ids=(1, 2))
    assert result.off_option_mass is None


@pytest.mark.unit
@pytest.mark.parametrize("total", [0.991, 1.009])
def test_llama_cpp_total_inside_sum_margin_reports_mass(total: float) -> None:
    result = _llama_cpp(_with_total(total), n_vocab=4, candidate_ids=(1, 2))
    assert result.off_option_mass == pytest.approx(0.3, abs=1e-12)


@pytest.mark.unit
@pytest.mark.parametrize("total", [0.989, 1.011])
def test_llama_cpp_total_outside_sum_margin_reports_unavailable(total: float) -> None:
    result = _llama_cpp(_with_total(total), n_vocab=4, candidate_ids=(1, 2))
    assert result.off_option_mass is None


@pytest.mark.unit
def test_llama_cpp_replay_of_float32_softmax_drift() -> None:
    """Replay the shape of one local Gemma 4 call (review, 2026-09-30).

    That call returned all 262144 entries with a total of 1.0005991. The two
    candidate logprobs below are the values it returned for ids 100 and
    236820. Six filler tokens carry the rest of the drifted total.
    """
    first, second = -0.14641161262989044, -2.848642110824585
    rest = (1.0005991 - math.exp(first) - math.exp(second)) / 6
    entries = [{"id": 100, "logprob": first}, {"id": 236820, "logprob": second}]
    entries += [{"id": 900 + i, "logprob": math.log(rest)} for i in range(6)]
    payload = {"completion_probabilities": [{"top_logprobs": entries}]}
    client = httpx.Client(
        transport=httpx.MockTransport(lambda _r: httpx.Response(200, json=payload)),
        base_url="http://test",
    )
    adapter = LlamaCppCandidateScoringAdapter(
        base_url="http://test", client=client, n_vocab=8
    )
    result = adapter.score_candidates(_request(100, 236820))
    assert result.off_option_mass == pytest.approx(0.07827500144064548, abs=1e-12)


@pytest.mark.unit
def test_llama_cpp_total_above_one_reports_unavailable() -> None:
    probs = {1: 0.7, 2: 0.6, 3: 0.2}
    result = _llama_cpp(probs, n_vocab=3, candidate_ids=(1, 2))
    assert result.off_option_mass is None
    assert [row.logprob for row in result.candidates] == [
        math.log(0.7),
        math.log(0.6),
    ]


@pytest.mark.unit
def test_llama_cpp_off_option_mass_is_clamped_at_zero() -> None:
    entries = [{"id": 1, "logprob": 1e-7}]
    payload = {"completion_probabilities": [{"top_logprobs": entries}]}
    client = httpx.Client(
        transport=httpx.MockTransport(lambda _r: httpx.Response(200, json=payload)),
        base_url="http://test",
    )
    adapter = LlamaCppCandidateScoringAdapter(
        base_url="http://test", client=client, n_vocab=1
    )
    assert adapter.score_candidates(_request(1)).off_option_mass == 0.0


@pytest.mark.unit
def test_vllm_response_reports_off_option_mass_unavailable() -> None:
    fixture = json.loads(
        (_VLLM_FIXTURES / "text_three_way.json").read_text(encoding="utf-8")
    )
    payload = fixture["response"]
    client = httpx.Client(
        transport=httpx.MockTransport(lambda _r: httpx.Response(200, json=payload)),
        base_url="http://test",
    )
    adapter = VllmCandidateScoringAdapter(base_url="http://test", client=client)
    result = adapter.score_candidates(_request(236771, 236770, 236778))
    assert len(result.candidates) == 3
    assert result.off_option_mass is None


@pytest.mark.unit
def test_result_field_defaults_to_unavailable() -> None:
    result = CandidateScoringResult(
        model="m",
        stage=ScoreStage.PRE_SAMPLING,
        candidates=(ScoredCandidate("c0", (1,), -1.0),),
    )
    assert result.off_option_mass is None


@pytest.mark.unit
def test_build_keeps_valid_off_option_mass() -> None:
    result = build_and_validate_result(
        _request(1), raw_logprobs={"c0": math.log(0.6)}, model="m", off_option_mass=0.4
    )
    assert result.off_option_mass == 0.4


@pytest.mark.unit
@pytest.mark.parametrize(
    ("value", "match"),
    [
        (float("nan"), "non-finite"),
        (float("inf"), "non-finite"),
        (-0.01, "outside"),
        (1.01, "outside"),
        (0.5, "above 1"),
    ],
)
def test_build_rejects_invalid_off_option_mass(value: float, match: str) -> None:
    with pytest.raises(ScoringValidationError, match=match):
        build_and_validate_result(
            _request(1),
            raw_logprobs={"c0": math.log(0.9)},
            model="m",
            off_option_mass=value,
        )


@pytest.mark.unit
def test_validate_result_rejects_mass_sum_above_one() -> None:
    result = CandidateScoringResult(
        model="m",
        stage=ScoreStage.PRE_SAMPLING,
        candidates=(ScoredCandidate("c0", (1,), math.log(0.9)),),
        off_option_mass=0.5,
    )
    with pytest.raises(ScoringValidationError, match="above 1"):
        validate_result_against_request(_request(1), result)


@pytest.mark.unit
def test_scripted_fake_carries_off_option_mass() -> None:
    request = _request(1)
    scripted: dict[str, Any] = {"logprobs": {"c0": math.log(0.75)}}
    assert (
        ScriptedScoringFake(**scripted).score_candidates(request).off_option_mass
        is None
    )
    with_mass = ScriptedScoringFake(**scripted, off_option_mass=0.25)
    assert with_mass.score_candidates(request).off_option_mass == 0.25
