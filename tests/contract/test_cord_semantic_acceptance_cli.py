"""Contract tests for the CORD semantic acceptance operator CLI (#184)."""

from __future__ import annotations

from pathlib import Path

import pytest

from typevet.adapters.inbound.cord_semantic_acceptance_cli import main
from typevet.cord_semantic_acceptance_cli import main as shim_main

pytestmark = pytest.mark.contract

FIXTURE_DIR = (
    Path(__file__).resolve().parents[1] / "fixtures" / "cord" / "semantic_acceptance"
)
EXPENSE_FAIL = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "cord"
    / "expense_smoke"
    / "gemma4_kv9_direct_receipt.json"
)
HISTORICAL_PASS = FIXTURE_DIR / "gemma4_post187_combined_pass.json"


def test_cli_accepts_synthetic_pass_fixture(capsys: pytest.CaptureFixture[str]) -> None:
    path = FIXTURE_DIR / "labeled_synthetic_pass.json"
    code = main([str(path)])
    out = capsys.readouterr().out
    assert code == 0
    assert "accepted: true" in out
    assert "| answerable_accuracy |" in out
    assert "PASS" in out


def test_cli_rejects_vendored_gemma4_fail_receipt(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main([str(EXPENSE_FAIL)])
    out = capsys.readouterr().out
    assert code == 1
    assert "accepted: false" in out
    assert "FAIL" in out


@pytest.mark.skipif(not HISTORICAL_PASS.is_file(), reason="historical PASS fixture")
def test_cli_accepts_vendored_post187_pass_receipt(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main([str(HISTORICAL_PASS)])
    out = capsys.readouterr().out
    assert code == 0
    assert "accepted: true" in out


def test_cli_nonzero_on_malformed_json(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert main([str(bad)]) == 2


def test_cli_nonzero_on_non_object_json_root(tmp_path: Path) -> None:
    bad = tmp_path / "array.json"
    bad.write_text("[1, 2]", encoding="utf-8")
    assert main([str(bad)]) == 2


def test_cli_nonzero_on_missing_path() -> None:
    assert main(["/no/such/receipt.json"]) == 2


def test_cli_usage_error_without_receipt() -> None:
    assert main([]) == 2


def test_shim_module_main_matches_inbound(
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = FIXTURE_DIR / "labeled_synthetic_pass.json"
    assert shim_main([str(path)]) == 0
    assert "accepted: true" in capsys.readouterr().out


def test_report_lists_every_failure_reason(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main([str(FIXTURE_DIR / "receipt_425e6c9.json")])
    err = capsys.readouterr()
    assert code == 1
    assert err.out.count("FAIL") >= 1
    assert "failures:" in err.out
