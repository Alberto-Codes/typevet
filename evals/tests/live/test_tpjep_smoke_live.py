"""Opt-in live TPJEP eight-task smoke via ``ScoringJudgmentAdapter`` (#106)."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

from tests.fixtures.tpjep.live_acceptance import assert_tpjep_live_smoke_receipt
from tests.live.gate import gate_live
from typevet.adapters.inbound.settings import load_llama_settings
from typevet.adapters.outbound.judgment_scoring import ScoringJudgmentAdapter
from typevet.adapters.outbound.llama_cpp.scoring import LlamaCppCandidateScoringAdapter
from typevet_evals.tpjep.loader import load_eight_task_fixture
from typevet_evals.tpjep.records import records_to_jsonl
from typevet_evals.tpjep.runner import TpjepRunConfig, run_tpjep_with_receipt

_LLAMA = load_llama_settings()
_PINNED_MODEL = "gemma-4-31b-24gib-kv11-decoder"
_FIXTURE = (
    Path(__file__).resolve().parents[3]
    / "tests"
    / "fixtures"
    / "tpjep"
    / "eight_task_smoke.jsonl"
).read_text(encoding="utf-8")
_OUTPUT_DIR = Path(__file__).resolve().parents[3] / "scratchpad" / "tpjep"


@pytest.fixture
def live_tpjep_model() -> str:
    gate_live(_LLAMA)
    model = _LLAMA.default_model or _PINNED_MODEL
    return model


@pytest.mark.live
def test_tpjep_eight_task_smoke_live(live_tpjep_model: str) -> None:
    """Exercise pinned protocol on eight vendored tasks; writes replay JSONL."""
    tasks = load_eight_task_fixture(_FIXTURE)
    live_settings = replace(_LLAMA, timeout=max(_LLAMA.timeout, 900.0))
    base = live_settings.base_url.rstrip("/")
    config = TpjepRunConfig(
        model=live_tpjep_model,
        template_class="gemma",
    )
    with httpx.Client(base_url=base, timeout=live_settings.timeout) as client:

        def tokenize_content(text: str) -> tuple[int, ...]:
            payload = (
                client.post(
                    "/tokenize",
                    json={
                        "model": live_tpjep_model,
                        "content": text,
                        "add_special": False,
                    },
                )
                .raise_for_status()
                .json()
            )
            return tuple(payload["tokens"])

        with LlamaCppCandidateScoringAdapter(
            base_url=base,
            timeout=live_settings.timeout,
            n_vocab=262144,
        ) as scoring:
            port = ScoringJudgmentAdapter(scoring, tokenize_content=tokenize_content)
            receipt = run_tpjep_with_receipt(port, tasks, config=config)

    assert_tpjep_live_smoke_receipt(receipt)
    assert receipt.metadata.thinking is False
    assert receipt.metadata.permutations == 1
    assert receipt.metadata.manifest_hash
    assert receipt.metadata.local_concat_hash
    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (_OUTPUT_DIR / "live_attempts.jsonl").write_text(
        records_to_jsonl(receipt.records), encoding="utf-8"
    )
    (_OUTPUT_DIR / "live_summary.json").write_text(
        json.dumps(
            {
                "n_scheduled": receipt.summary.n_scheduled,
                "n_answered": receipt.summary.n_answered,
                "n_prob_valid": receipt.summary.n_prob_valid,
                "n_correct": receipt.summary.n_correct,
                "metadata": {
                    "protocol": receipt.metadata.protocol,
                    "model": receipt.metadata.model,
                    "manifest_hash": receipt.metadata.manifest_hash,
                    "local_concat_hash": receipt.metadata.local_concat_hash,
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
