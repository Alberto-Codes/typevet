"""Opt-in live eval runner over BoolQ smoke (#98)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from typevet.adapters.inbound.settings import llama_cpp_adapter, load_llama_settings
from typevet.evaluation.runner.core import run_eval_tasks
from typevet.evaluation.runner.datasets import load_eval_tasks
from typevet.evaluation.runner.live_gate import live_skip_reason

_LLAMA = load_llama_settings()
BOOLQ_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "boolq"
    / "validation_smoke.jsonl"
).read_text(encoding="utf-8")


@pytest.fixture
def live_llama_router() -> str:
    """Skip when the stock live path is not configured."""
    reason = live_skip_reason(_LLAMA)
    if reason is not None:
        pytest.skip(reason)
    model = _LLAMA.default_model
    if model is None:
        pytest.skip("TYPEVET_LLAMA__DEFAULT_MODEL (or TYPEVET_GEMMA_MODEL) not set")
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
