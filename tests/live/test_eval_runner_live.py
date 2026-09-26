"""Opt-in live eval runner over BoolQ smoke (#98)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from tests.live.gate import gate_live
from typevet.adapters.inbound.settings import llama_cpp_adapter, load_llama_settings
from typevet.evaluation.runner.core import run_eval_tasks
from typevet.evaluation.runner.datasets import load_eval_tasks

_LLAMA = load_llama_settings()
BOOLQ_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "boolq"
    / "validation_smoke.jsonl"
).read_text(encoding="utf-8")


@pytest.fixture
def live_llama_router() -> str:
    """Skip or fail when the stock live path is not configured."""
    gate_live(_LLAMA)
    model = _LLAMA.default_model
    assert model is not None
    return model


@pytest.mark.live
def test_eval_runner_boolq_smoke(live_llama_router: str) -> None:
    tasks = load_eval_tasks("boolq", limit=2, boolq_jsonl_text=BOOLQ_FIXTURE)
    live_settings = replace(_LLAMA, timeout=max(_LLAMA.timeout, 600.0))
    with llama_cpp_adapter(live_settings) as port:
        report = run_eval_tasks(port, tasks, model=live_llama_router)
    assert report.attempted == 2
    assert report.schema_valid >= 0
    assert report.gold_match <= report.schema_valid
