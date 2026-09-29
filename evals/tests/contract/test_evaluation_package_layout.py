"""Contract tests for the removed root ``eval_*`` shims (#147, #256).

The root ``eval_*`` shims are removed before 0.1.0 (#256). Each old root path
must not resolve, and each current home must import.
"""

from __future__ import annotations

import importlib
import importlib.util
import runpy
import sys
from pathlib import Path

import pytest

from typevet_evals.cli import eval_runner as eval_cli
from typevet_evals.datasets import clinc_shard
from typevet_evals.runner import (
    SUPPORTED_DATASETS,
    EvalRunReport,
    EvalTaskSpec,
    format_report,
    live_skip_reason,
    load_eval_tasks,
    run_eval_tasks,
)
from typevet_evals.tpjep import (
    TPJEP_MANIFEST_HASH,
    TPJEP_PROTOCOL_V0,
    TpjepRunConfig,
    load_eight_task_fixture,
    outcome_from_answer,
    run_tpjep_tasks,
    summarize_tpjep_records,
)

# Removed root module -> current home. The old path must not resolve.
REMOVED_SHIM_HOMES: dict[str, str] = {
    "typevet.eval_tpjep_loader": "typevet_evals.tpjep.loader",
    "typevet.eval_tpjep_outcome": "typevet_evals.tpjep.outcome",
    "typevet.eval_tpjep_records": "typevet_evals.tpjep.records",
    "typevet.eval_tpjep_runner": "typevet_evals.tpjep.runner",
    "typevet.eval_runner": "typevet_evals.runner.core",
    "typevet.eval_runner_datasets": "typevet_evals.runner.datasets",
    "typevet.eval_runner_live_gate": "typevet_evals.runner.live_gate",
    "typevet.eval_runner_report": "typevet_evals.runner.report",
    "typevet.eval_runner_cli": "typevet_evals.cli.eval_runner",
    "typevet.eval_banking77": "typevet_evals.datasets.banking77",
    "typevet.eval_boolq": "typevet_evals.datasets.boolq",
    "typevet.eval_boolq_download": "typevet_evals.datasets.boolq_download",
    "typevet.eval_civil_comments": "typevet_evals.datasets.civil_comments",
    "typevet.eval_clinc": "typevet_evals.datasets.clinc",
    "typevet.eval_clinc_download": "typevet_evals.datasets.clinc_download",
    "typevet.eval_clinc_rows": "typevet_evals.datasets.clinc_rows",
    "typevet.eval_clinc_shard": "typevet_evals.datasets.clinc_shard",
    "typevet.eval_difraud": "typevet_evals.datasets.difraud",
    "typevet.eval_go_emotions": "typevet_evals.datasets.go_emotions",
    "typevet.eval_go_emotions_download": (
        "typevet_evals.datasets.go_emotions_download"
    ),
    "typevet.eval_hyperpartisan": "typevet_evals.datasets.hyperpartisan",
    "typevet.eval_partner_guard": "typevet_evals.datasets.partner_guard",
    "typevet.eval_psai": "typevet_evals.datasets.psai",
    "typevet.eval_psai_download": "typevet_evals.datasets.psai_download",
    "typevet.eval_psai_schema": "typevet_evals.datasets.psai_schema",
    "typevet.eval_psai_stream": "typevet_evals.datasets.psai_stream",
    "typevet.eval_pubmedqa": "typevet_evals.datasets.pubmedqa",
}

# Legacy symbols callers depend on, per current home of a removed shim.
LEGACY_SYMBOLS: dict[str, tuple[str, ...]] = {
    "typevet_evals.tpjep.loader": (
        "TPJEP_DATASET_GIT_COMMIT",
        "TPJEP_MANIFEST_HASH",
        "TpjepScheduledTask",
        "load_eight_task_fixture",
        "model_inputs_for_task",
    ),
    "typevet_evals.tpjep.outcome": ("outcome_from_answer", "prob_valid"),
    "typevet_evals.tpjep.records": (
        "TPJEP_PROTOCOL_V0",
        "TpjepAttemptRecord",
        "TpjepRunSummary",
        "summarize_tpjep_records",
    ),
    "typevet_evals.tpjep.runner": (
        "TpjepRunConfig",
        "TpjepRunReceipt",
        "run_tpjep_tasks",
        "run_tpjep_with_receipt",
    ),
    "typevet_evals.runner.core": ("run_eval_tasks",),
    "typevet_evals.runner.datasets": ("SUPPORTED_DATASETS", "load_eval_tasks"),
    "typevet_evals.runner.live_gate": ("live_skip_reason",),
    "typevet_evals.runner.report": (
        "EvalRunReport",
        "format_report",
        "merge_reports",
    ),
    "typevet_evals.cli.eval_runner": ("main",),
    "typevet_evals.datasets.clinc_shard": (
        "domain_intent_map",
        "plus_intent_names",
    ),
    "typevet_evals.datasets.partner_guard": ("scan_tree_paths",),
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
def test_eval_cli_lives_in_the_evals_member() -> None:
    assert callable(eval_cli.main)


@pytest.mark.contract
@pytest.mark.parametrize(("old_name", "new_name"), sorted(REMOVED_SHIM_HOMES.items()))
def test_root_shim_is_removed_and_home_imports(old_name: str, new_name: str) -> None:
    assert importlib.util.find_spec(old_name) is None
    importlib.import_module(new_name)


@pytest.mark.contract
@pytest.mark.parametrize(("module_name", "symbols"), sorted(LEGACY_SYMBOLS.items()))
def test_current_home_keeps_legacy_symbols(
    module_name: str, symbols: tuple[str, ...]
) -> None:
    module = importlib.import_module(module_name)
    for name in symbols:
        assert hasattr(module, name), f"{module_name} lost {name}"


@pytest.mark.contract
def test_eval_cli_module_runs_as_main(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    module_name = "typevet_evals.cli.eval_runner"
    monkeypatch.setattr("sys.argv", [module_name, "--limit", "0"])
    monkeypatch.delitem(sys.modules, module_name, raising=False)
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module(module_name, run_name="__main__")
    assert exit_info.value.code == 2
    assert "--limit must be positive" in capsys.readouterr().err


@pytest.mark.contract
def test_clinc_json_resources_ship_with_the_dataset_package() -> None:
    package_dir = Path(clinc_shard.__file__).resolve().parent
    assert (package_dir / "clinc_domains.json").is_file()
    assert (package_dir / "clinc_plus_intent_names.json").is_file()
    assert len(clinc_shard.domain_intent_map()) == 10
    assert clinc_shard.plus_intent_names()
