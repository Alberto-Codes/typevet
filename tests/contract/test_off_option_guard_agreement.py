"""Contract: fakes and real adapters agree on the off-option guard (#353).

The llama.cpp pair reports a mass of 0.3. The vLLM pair reports ``None``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from tests.fixtures.scoring_contract import ContractScoringFake, get_off_option_fixture
from typevet.adapters.outbound.llama_cpp.scoring import LlamaCppCandidateScoringAdapter
from typevet.adapters.outbound.vllm.scoring import VllmCandidateScoringAdapter
from typevet.domain.candidate_scoring_request import CandidateTokenSpec
from typevet.domain.decision_execute import (
    CategoricalExecutionResult,
    apply_off_option_threshold,
    execute_categorical_decision,
)
from typevet.domain.decisions import Decision
from typevet.ports.scoring import CandidateScoringPort

_VLLM_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "vllm"
_VLLM_SPECS = (
    CandidateTokenSpec("zero", (236771,)),
    CandidateTokenSpec("one", (236770,)),
    CandidateTokenSpec("two", (236778,)),
)


def _client(payload: object) -> httpx.Client:
    return httpx.Client(
        transport=httpx.MockTransport(lambda _r: httpx.Response(200, json=payload)),
        base_url="http://test",
    )


def _run(
    port: CandidateScoringPort,
    specs: tuple[CandidateTokenSpec, ...],
    model: str,
    threshold: float | None,
) -> CategoricalExecutionResult:
    labels = tuple(spec.label for spec in specs)
    decision = Decision("c", "Pick.", labels, syntax="Choice")
    executed = execute_categorical_decision(
        decision, prefix="Answer:", candidates=specs, port=port, model=model
    )
    return apply_off_option_threshold(executed, threshold)


def _llama_cpp_pair() -> tuple[list[CandidateScoringPort], Any]:
    fixture = get_off_option_fixture()
    payload = {"completion_probabilities": [{"top_logprobs": fixture["top_logprobs"]}]}
    adapter = LlamaCppCandidateScoringAdapter(
        base_url="http://test", client=_client(payload), n_vocab=fixture["n_vocab"]
    )
    return [ContractScoringFake(**fixture["fake"]), adapter], fixture["request"]


def _vllm_pair() -> list[CandidateScoringPort]:
    raw = (_VLLM_FIXTURES / "text_three_way.json").read_text(encoding="utf-8")
    payload = json.loads(raw)["response"]
    adapter = VllmCandidateScoringAdapter(
        base_url="http://test", client=_client(payload)
    )
    content = payload["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    by_id = {int(i["token"].removeprefix("token_id:")): i["logprob"] for i in content}
    logprobs = {spec.label: by_id[spec.token_ids[0]] for spec in _VLLM_SPECS}
    return [ContractScoringFake(logprobs=logprobs), adapter]


@pytest.mark.contract
@pytest.mark.parametrize(("threshold", "flag"), [(0.25, True), (0.35, False)])
def test_llama_cpp_fake_and_adapter_flag_alike(threshold: float, flag: bool) -> None:
    ports, request = _llama_cpp_pair()
    for port in ports:
        receipt = _run(port, request.candidates, request.model, threshold).off_option
        assert receipt.off_option_mass == pytest.approx(0.3, abs=1e-12)
        assert receipt.off_option_threshold == threshold
        assert receipt.off_option_flag is flag


@pytest.mark.contract
def test_vllm_fake_and_adapter_never_flag_unavailable_mass() -> None:
    for port in _vllm_pair():
        receipt = _run(port, _VLLM_SPECS, "gemma-4-31b-it", 0.0).off_option
        assert receipt.as_dict() == {
            "off_option_mass": None,
            "off_option_threshold": 0.0,
            "off_option_flag": False,
        }
