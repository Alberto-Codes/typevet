"""Offline tests for loader eval runner wiring (#98)."""

from __future__ import annotations

from pathlib import Path

import pytest

from typevet.adapters.inbound.settings import load_llama_settings
from typevet.adapters.outbound.fake import FakeGenerationAdapter
from typevet.domain.errors import SchemaValidationError
from typevet.domain.models import GenerationRequest
from typevet.eval_runner import run_eval_tasks
from typevet.eval_runner_datasets import load_eval_tasks
from typevet.eval_runner_live_gate import live_skip_reason
from typevet.eval_runner_report import EvalRunReport, format_report, merge_reports

BOOLQ_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "boolq"
    / "validation_smoke.jsonl"
).read_text(encoding="utf-8")
BANKING77_FIXTURE = (
    Path(__file__).resolve().parents[1] / "fixtures" / "banking77" / "test_subset.csv"
).read_text(encoding="utf-8")


@pytest.mark.unit
def test_load_boolq_tasks_from_fixture() -> None:
    tasks = load_eval_tasks(
        "boolq",
        limit=2,
        boolq_jsonl_text=BOOLQ_FIXTURE,
    )
    assert len(tasks) == 2
    assert tasks[0].noul_field == "answer"
    assert tasks[0].gold in {"yes", "no"}


@pytest.mark.unit
def test_load_banking77_tasks_from_fixture() -> None:
    tasks = load_eval_tasks(
        "banking77",
        limit=4,
        banking77_csv_text=BANKING77_FIXTURE,
    )
    assert len(tasks) == 4
    assert tasks[0].noul_field == "reports_unauthorized"
    assert isinstance(tasks[0].gold, bool)


@pytest.mark.unit
def test_run_eval_tasks_counts_schema_and_gold() -> None:
    tasks = load_eval_tasks(
        "boolq",
        limit=2,
        boolq_jsonl_text=BOOLQ_FIXTURE,
    )

    def responder(request: GenerationRequest):
        for task in tasks:
            if task.prompt == request.prompt:
                return {task.noul_field: task.gold}
        return {tasks[0].noul_field: "no"}

    port = FakeGenerationAdapter(responder=responder)
    report = run_eval_tasks(port, tasks, model="fake")
    assert report.attempted == 2
    assert report.schema_valid == 2
    assert report.gold_match == 2


@pytest.mark.unit
def test_run_eval_tasks_schema_invalid_not_counted() -> None:
    tasks = load_eval_tasks(
        "boolq",
        limit=2,
        boolq_jsonl_text=BOOLQ_FIXTURE,
    )
    tasks = tasks[:1]
    port = FakeGenerationAdapter(
        fail=SchemaValidationError("bad", payload={"answer": "maybe"}),
    )
    report = run_eval_tasks(port, tasks, model="fake")
    assert report.attempted == 1
    assert report.schema_valid == 0
    assert report.gold_match == 0


@pytest.mark.unit
def test_merge_reports_sums() -> None:
    first = EvalRunReport(
        dataset="boolq",
        metric_name="exact_match",
        limit=1,
        attempted=1,
        schema_valid=1,
        gold_match=0,
    )
    second = EvalRunReport(
        dataset="boolq",
        metric_name="exact_match",
        limit=1,
        attempted=1,
        schema_valid=0,
        gold_match=0,
    )
    merged = merge_reports([first, second])
    assert merged.attempted == 2
    assert merged.schema_valid == 1


@pytest.mark.unit
def test_format_report_line() -> None:
    tasks = load_eval_tasks(
        "boolq",
        limit=2,
        boolq_jsonl_text=BOOLQ_FIXTURE,
    )
    report = run_eval_tasks(
        FakeGenerationAdapter(value={"answer": tasks[0].gold}),
        tasks[:1],
        model="fake",
    )
    line = format_report(report)
    assert "dataset=boolq" in line
    assert "schema_valid=1" in line


@pytest.mark.unit
def test_live_skip_when_model_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TYPEVET_LLAMA__DEFAULT_MODEL", raising=False)
    monkeypatch.delenv("TYPEVET_GEMMA_MODEL", raising=False)
    settings = load_llama_settings()
    assert live_skip_reason(settings) is not None
