"""Contract tests for the ``typevet.evaluation`` package and root eval shims (#147)."""

from __future__ import annotations

import importlib
import runpy
import sys
from pathlib import Path

import pytest

from typevet.adapters.inbound import eval_cli
from typevet.evaluation.datasets import clinc_shard
from typevet.evaluation.runner import (
    SUPPORTED_DATASETS,
    EvalRunReport,
    EvalTaskSpec,
    format_report,
    live_skip_reason,
    load_eval_tasks,
    run_eval_tasks,
)
from typevet.evaluation.tpjep import (
    TPJEP_MANIFEST_HASH,
    TPJEP_PROTOCOL_V0,
    TpjepRunConfig,
    load_eight_task_fixture,
    outcome_from_answer,
    run_tpjep_tasks,
    summarize_tpjep_records,
)

# Old root module -> new home. Every entry must keep working as an import path.
SHIM_TARGETS: dict[str, str] = {
    "typevet.eval_tpjep_loader": "typevet.evaluation.tpjep.loader",
    "typevet.eval_tpjep_outcome": "typevet.evaluation.tpjep.outcome",
    "typevet.eval_tpjep_records": "typevet.evaluation.tpjep.records",
    "typevet.eval_tpjep_runner": "typevet.evaluation.tpjep.runner",
    "typevet.eval_runner": "typevet.evaluation.runner.core",
    "typevet.eval_runner_datasets": "typevet.evaluation.runner.datasets",
    "typevet.eval_runner_live_gate": "typevet.evaluation.runner.live_gate",
    "typevet.eval_runner_report": "typevet.evaluation.runner.report",
    "typevet.eval_runner_cli": "typevet.adapters.inbound.eval_cli",
    "typevet.eval_banking77": "typevet.evaluation.datasets.banking77",
    "typevet.eval_boolq": "typevet.evaluation.datasets.boolq",
    "typevet.eval_boolq_download": "typevet.evaluation.datasets.boolq_download",
    "typevet.eval_civil_comments": "typevet.evaluation.datasets.civil_comments",
    "typevet.eval_clinc": "typevet.evaluation.datasets.clinc",
    "typevet.eval_clinc_download": "typevet.evaluation.datasets.clinc_download",
    "typevet.eval_clinc_rows": "typevet.evaluation.datasets.clinc_rows",
    "typevet.eval_clinc_shard": "typevet.evaluation.datasets.clinc_shard",
    "typevet.eval_difraud": "typevet.evaluation.datasets.difraud",
    "typevet.eval_go_emotions": "typevet.evaluation.datasets.go_emotions",
    "typevet.eval_go_emotions_download": (
        "typevet.evaluation.datasets.go_emotions_download"
    ),
    "typevet.eval_hyperpartisan": "typevet.evaluation.datasets.hyperpartisan",
    "typevet.eval_partner_guard": "typevet.evaluation.datasets.partner_guard",
    "typevet.eval_psai": "typevet.evaluation.datasets.psai",
    "typevet.eval_psai_download": "typevet.evaluation.datasets.psai_download",
    "typevet.eval_psai_schema": "typevet.evaluation.datasets.psai_schema",
    "typevet.eval_psai_stream": "typevet.evaluation.datasets.psai_stream",
    "typevet.eval_pubmedqa": "typevet.evaluation.datasets.pubmedqa",
}

# Legacy symbols callers depend on, per root shim module.
LEGACY_SYMBOLS: dict[str, tuple[str, ...]] = {
    "typevet.eval_tpjep_loader": (
        "TPJEP_DATASET_GIT_COMMIT",
        "TPJEP_MANIFEST_HASH",
        "TpjepScheduledTask",
        "load_eight_task_fixture",
        "model_inputs_for_task",
    ),
    "typevet.eval_tpjep_outcome": ("outcome_from_answer", "prob_valid"),
    "typevet.eval_tpjep_records": (
        "TPJEP_PROTOCOL_V0",
        "TpjepAttemptRecord",
        "TpjepRunSummary",
        "summarize_tpjep_records",
    ),
    "typevet.eval_tpjep_runner": (
        "TpjepRunConfig",
        "TpjepRunReceipt",
        "run_tpjep_tasks",
        "run_tpjep_with_receipt",
    ),
    "typevet.eval_runner": ("run_eval_tasks",),
    "typevet.eval_runner_datasets": ("SUPPORTED_DATASETS", "load_eval_tasks"),
    "typevet.eval_runner_live_gate": ("live_skip_reason",),
    "typevet.eval_runner_report": ("EvalRunReport", "format_report", "merge_reports"),
    "typevet.eval_runner_cli": ("main",),
    "typevet.eval_clinc_shard": ("domain_intent_map", "plus_intent_names"),
    "typevet.eval_partner_guard": ("scan_tree_paths",),
}


@pytest.mark.contract
def test_tpjep_package_exports_public_names() -> None:
    assert TPJEP_MANIFEST_HASH
    assert TPJEP_PROTOCOL_V0
    assert callable(load_eight_task_fixture)
    assert callable(outcome_from_answer)
    assert callable(run_tpjep_tasks)
    assert callable(summarize_tpjep_records)
    assert TpjepRunConfig is not None


@pytest.mark.contract
def test_runner_package_exports_public_names() -> None:
    assert "boolq" in SUPPORTED_DATASETS
    assert EvalRunReport is not None
    assert EvalTaskSpec is not None
    assert callable(format_report)
    assert callable(live_skip_reason)
    assert callable(load_eval_tasks)
    assert callable(run_eval_tasks)


@pytest.mark.contract
def test_eval_cli_lives_under_inbound_adapters() -> None:
    assert callable(eval_cli.main)


@pytest.mark.contract
@pytest.mark.parametrize(("old_name", "new_name"), sorted(SHIM_TARGETS.items()))
def test_root_shim_reexports_new_module(old_name: str, new_name: str) -> None:
    shim = importlib.import_module(old_name)
    target = importlib.import_module(new_name)
    exported = getattr(shim, "__all__", None)
    assert exported, f"{old_name} must declare __all__"
    for name in exported:
        assert getattr(shim, name) is getattr(target, name), (
            f"{old_name}.{name} must be {new_name}.{name}"
        )


@pytest.mark.contract
@pytest.mark.parametrize(("old_name", "symbols"), sorted(LEGACY_SYMBOLS.items()))
def test_root_shim_keeps_legacy_symbols(
    old_name: str, symbols: tuple[str, ...]
) -> None:
    shim = importlib.import_module(old_name)
    for name in symbols:
        assert hasattr(shim, name), f"{old_name} lost {name}"


@pytest.mark.contract
def test_eval_runner_cli_module_still_runs_as_main(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr("sys.argv", ["typevet.eval_runner_cli", "--limit", "0"])
    monkeypatch.delitem(sys.modules, "typevet.eval_runner_cli", raising=False)
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("typevet.eval_runner_cli", run_name="__main__")
    assert exit_info.value.code == 2
    assert "--limit must be positive" in capsys.readouterr().err


@pytest.mark.contract
def test_clinc_json_resources_ship_with_the_dataset_package() -> None:
    package_dir = Path(clinc_shard.__file__).resolve().parent
    assert (package_dir / "clinc_domains.json").is_file()
    assert (package_dir / "clinc_plus_intent_names.json").is_file()
    assert len(clinc_shard.domain_intent_map()) == 10
    assert clinc_shard.plus_intent_names()
