"""Unit tests for opt-in eval runner CLI (#98)."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest

from typevet.adapters.inbound.settings import LlamaSettings
from typevet_evals.cli.eval_runner import _as_dataset, main
from typevet_evals.runner.report import EvalRunReport


@pytest.mark.unit
def test_as_dataset_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="unsupported eval dataset"):
        _as_dataset("not-a-dataset")


@pytest.mark.unit
def test_main_skips_when_live_gate_reports_reason(
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = LlamaSettings(default_model="gemma")
    with (
        patch(
            "typevet_evals.cli.eval_runner.load_llama_settings",
            return_value=settings,
        ),
        patch(
            "typevet_evals.cli.eval_runner.live_skip_reason",
            return_value="llama.cpp router not reachable",
        ),
    ):
        code = main([])
    assert code == 0
    assert "skip: llama.cpp router not reachable" in capsys.readouterr().err


@pytest.mark.unit
def test_main_skips_when_default_model_missing_after_gate(
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = LlamaSettings(default_model=None)
    with (
        patch(
            "typevet_evals.cli.eval_runner.load_llama_settings",
            return_value=settings,
        ),
        patch("typevet_evals.cli.eval_runner.live_skip_reason", return_value=None),
    ):
        code = main([])
    assert code == 0
    err = capsys.readouterr().err
    assert "skip:" in err
    assert "DEFAULT_MODEL" in err


@pytest.mark.unit
def test_main_runs_datasets_and_prints_reports(
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = LlamaSettings(default_model="gemma", timeout=30.0)
    report = EvalRunReport(
        dataset="boolq",
        metric_name="exact_match",
        limit=1,
        attempted=1,
        schema_valid=1,
        gold_match=1,
    )
    fake_port = MagicMock()

    @contextmanager
    def fake_adapter(_settings: LlamaSettings):
        yield fake_port

    with (
        patch(
            "typevet_evals.cli.eval_runner.load_llama_settings",
            return_value=settings,
        ),
        patch("typevet_evals.cli.eval_runner.live_skip_reason", return_value=None),
        patch("typevet_evals.cli.eval_runner.llama_cpp_adapter", fake_adapter),
        patch(
            "typevet_evals.cli.eval_runner.load_eval_tasks",
            return_value=[cast(Any, object())],
        ),
        patch("typevet_evals.cli.eval_runner.run_eval_tasks", return_value=report),
    ):
        code = main(["--dataset", "boolq", "--limit", "1", "--seed", "3"])
    assert code == 0
    out = capsys.readouterr().out
    assert "dataset=boolq" in out
    assert "gold_match=1" in out


@pytest.mark.unit
def test_main_default_dataset_is_boolq_when_omitted(
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = LlamaSettings(default_model="gemma", timeout=900.0)
    seen: list[str] = []

    def capture_load(dataset: str, **kwargs: object) -> list[object]:
        seen.append(dataset)
        return [cast(Any, object())]

    report = EvalRunReport(
        dataset="boolq",
        metric_name="exact_match",
        limit=4,
        attempted=0,
        schema_valid=0,
        gold_match=0,
    )

    @contextmanager
    def fake_adapter(_settings: LlamaSettings):
        yield MagicMock()

    with (
        patch(
            "typevet_evals.cli.eval_runner.load_llama_settings",
            return_value=settings,
        ),
        patch("typevet_evals.cli.eval_runner.live_skip_reason", return_value=None),
        patch("typevet_evals.cli.eval_runner.llama_cpp_adapter", fake_adapter),
        patch(
            "typevet_evals.cli.eval_runner.load_eval_tasks",
            side_effect=capture_load,
        ),
        patch("typevet_evals.cli.eval_runner.run_eval_tasks", return_value=report),
    ):
        main([])
    assert seen == ["boolq"]


def _live_ready_settings() -> LlamaSettings:
    return LlamaSettings(default_model="gemma", timeout=30.0)


def _fake_adapter():
    @contextmanager
    def _adapter(_settings: LlamaSettings):
        yield MagicMock()

    return _adapter


@contextmanager
def _patch_live_run(
    *,
    report: EvalRunReport,
    tasks: list[object] | None = None,
):
    if tasks is None:
        tasks = [cast(Any, object())]
    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "typevet_evals.cli.eval_runner.load_llama_settings",
                return_value=_live_ready_settings(),
            )
        )
        stack.enter_context(
            patch("typevet_evals.cli.eval_runner.live_skip_reason", return_value=None)
        )
        stack.enter_context(
            patch("typevet_evals.cli.eval_runner.llama_cpp_adapter", _fake_adapter())
        )
        stack.enter_context(
            patch("typevet_evals.cli.eval_runner.load_eval_tasks", return_value=tasks)
        )
        stack.enter_context(
            patch("typevet_evals.cli.eval_runner.run_eval_tasks", return_value=report)
        )
        yield


@pytest.mark.unit
def test_main_accepts_require_live_flag(
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = EvalRunReport(
        dataset="boolq",
        metric_name="exact_match",
        limit=1,
        attempted=1,
        schema_valid=1,
        gold_match=1,
    )
    with _patch_live_run(report=report):
        code = main(["--require-live", "--dataset", "boolq", "--limit", "1"])
    assert code == 0
    assert "dataset=boolq" in capsys.readouterr().out


@pytest.mark.unit
def test_main_limit_must_be_positive_before_settings_io() -> None:
    loader = MagicMock()
    with patch("typevet_evals.cli.eval_runner.load_llama_settings", loader):
        code = main(["--limit", "0"])
    assert code == 2
    loader.assert_not_called()


@pytest.mark.unit
def test_require_live_nonzero_when_live_gate_skips(
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = LlamaSettings(default_model="gemma")
    with (
        patch(
            "typevet_evals.cli.eval_runner.load_llama_settings",
            return_value=settings,
        ),
        patch(
            "typevet_evals.cli.eval_runner.live_skip_reason",
            return_value="llama.cpp router not reachable",
        ),
    ):
        code = main(["--require-live"])
    assert code != 0
    err = capsys.readouterr().err
    assert "llama.cpp router not reachable" in err


@pytest.mark.unit
def test_require_live_nonzero_when_default_model_missing(
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = LlamaSettings(default_model=None)
    with (
        patch(
            "typevet_evals.cli.eval_runner.load_llama_settings",
            return_value=settings,
        ),
        patch("typevet_evals.cli.eval_runner.live_skip_reason", return_value=None),
    ):
        code = main(["--require-live"])
    assert code != 0
    err = capsys.readouterr().err
    assert "DEFAULT_MODEL" in err


@pytest.mark.unit
def test_require_live_nonzero_on_empty_workload(
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = EvalRunReport(
        dataset="boolq",
        metric_name="exact_match",
        limit=1,
        attempted=0,
        schema_valid=0,
        gold_match=0,
    )
    with _patch_live_run(report=report, tasks=[]):
        code = main(["--require-live", "--limit", "1"])
    assert code != 0
    assert capsys.readouterr().err


@pytest.mark.unit
def test_require_live_nonzero_when_some_calls_not_schema_valid(
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = EvalRunReport(
        dataset="boolq",
        metric_name="exact_match",
        limit=3,
        attempted=3,
        schema_valid=2,
        gold_match=1,
    )
    with _patch_live_run(report=report):
        code = main(["--require-live", "--limit", "3"])
    assert code != 0


@pytest.mark.unit
def test_require_live_nonzero_when_all_calls_not_schema_valid(
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = EvalRunReport(
        dataset="boolq",
        metric_name="exact_match",
        limit=2,
        attempted=2,
        schema_valid=0,
        gold_match=0,
    )
    with _patch_live_run(report=report):
        code = main(["--require-live", "--limit", "2"])
    assert code != 0


@pytest.mark.unit
def test_require_live_succeeds_with_schema_valid_wrong_gold(
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = EvalRunReport(
        dataset="boolq",
        metric_name="exact_match",
        limit=2,
        attempted=2,
        schema_valid=2,
        gold_match=0,
    )
    with _patch_live_run(report=report):
        code = main(["--require-live", "--limit", "2"])
    assert code == 0
    out = capsys.readouterr().out
    assert "gold_match=0" in out
