"""Offline tests for loader eval runner wiring (#98)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

from typevet.adapters.inbound.settings import LlamaSettings, load_llama_settings
from typevet.adapters.outbound.fake import FakeGenerationAdapter
from typevet.domain.errors import SchemaValidationError
from typevet.domain.models import GenerationRequest
from typevet_evals.runner.core import run_eval_tasks
from typevet_evals.runner.datasets import EvalTaskSpec, load_eval_tasks
from typevet_evals.runner.live_gate import live_skip_reason
from typevet_evals.runner.report import EvalRunReport, format_report, merge_reports

BOOLQ_FIXTURE = (
    Path(__file__).resolve().parents[3]
    / "tests"
    / "fixtures"
    / "boolq"
    / "validation_smoke.jsonl"
).read_text(encoding="utf-8")
BANKING77_FIXTURE = (
    Path(__file__).resolve().parents[3]
    / "tests"
    / "fixtures"
    / "banking77"
    / "test_subset.csv"
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


@pytest.mark.unit
def test_live_skip_when_router_unreachable() -> None:
    settings = LlamaSettings(base_url="http://127.0.0.1:1", default_model="gemma")
    with patch(
        "typevet_evals.runner.live_gate.httpx.get",
        side_effect=httpx.HTTPError("down"),
    ):
        reason = live_skip_reason(settings)
    assert reason == "llama.cpp router not reachable"


@pytest.mark.unit
def test_live_skip_when_model_not_in_catalog() -> None:
    settings = LlamaSettings(base_url="http://127.0.0.1:8090", default_model="missing")
    ok_response = MagicMock()
    ok_response.status_code = 200
    ok_response.json.return_value = {"data": [{"id": "other"}]}
    with patch("typevet_evals.runner.live_gate.httpx.get", return_value=ok_response):
        reason = live_skip_reason(settings)
    assert reason == "missing not in router catalog"


@pytest.mark.unit
def test_live_skip_none_when_router_lists_model() -> None:
    settings = LlamaSettings(base_url="http://127.0.0.1:8090", default_model="gemma")
    ok_response = MagicMock()
    ok_response.status_code = 200
    ok_response.json.return_value = {"data": [{"id": "gemma"}]}
    with patch("typevet_evals.runner.live_gate.httpx.get", return_value=ok_response):
        assert live_skip_reason(settings) is None


@pytest.mark.unit
def test_live_gate_model_list_handles_bad_json() -> None:
    settings = LlamaSettings(base_url="http://127.0.0.1:8090", default_model="gemma")
    bad_json = MagicMock()
    bad_json.status_code = 200
    bad_json.json.side_effect = ValueError("not json")
    with patch("typevet_evals.runner.live_gate.httpx.get", return_value=bad_json):
        reason = live_skip_reason(settings)
    assert reason == "gemma not in router catalog"


@pytest.mark.unit
def test_run_eval_tasks_rejects_empty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        run_eval_tasks(FakeGenerationAdapter(value={"answer": "yes"}), [], model="fake")


@pytest.mark.unit
def test_run_eval_tasks_rejects_mixed_datasets() -> None:
    tasks = [
        EvalTaskSpec(
            task_id="boolq:0",
            dataset="boolq",
            prompt="p1",
            schema={"type": "object", "properties": {}},
            noul_field="answer",
            gold="yes",
        ),
        EvalTaskSpec(
            task_id="banking77:0",
            dataset="banking77",
            prompt="p2",
            schema={"type": "object", "properties": {}},
            noul_field="reports_unauthorized",
            gold=True,
        ),
    ]
    with pytest.raises(ValueError, match="single dataset"):
        run_eval_tasks(
            FakeGenerationAdapter(value={"answer": "yes"}), tasks, model="fake"
        )


@pytest.mark.unit
def test_run_eval_tasks_banking77_uses_noul_metric() -> None:
    tasks = load_eval_tasks(
        "banking77",
        limit=4,
        banking77_csv_text=BANKING77_FIXTURE,
    )
    port = FakeGenerationAdapter(
        value={tasks[0].noul_field: tasks[0].gold},
    )
    report = run_eval_tasks(port, tasks[:1], model="fake")
    assert report.metric_name == "noul_agreement"


@pytest.mark.unit
def test_merge_reports_rejects_empty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        merge_reports([])


@pytest.mark.unit
def test_merge_reports_rejects_mismatched_keys() -> None:
    first = EvalRunReport(
        dataset="boolq",
        metric_name="exact_match",
        limit=1,
        attempted=1,
        schema_valid=1,
        gold_match=0,
    )
    second = EvalRunReport(
        dataset="banking77",
        metric_name="noul_agreement",
        limit=1,
        attempted=1,
        schema_valid=1,
        gold_match=0,
    )
    with pytest.raises(ValueError, match="same dataset"):
        merge_reports([first, second])
