"""Contract tests for the CORD semantic acceptance operator CLI (#184)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from typevet.adapters.inbound.cord_semantic_acceptance_cli import main

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


def test_report_lists_every_failure_reason(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main([str(FIXTURE_DIR / "receipt_425e6c9.json")])
    err = capsys.readouterr()
    assert code == 1
    assert err.out.count("FAIL") >= 1
    assert "failures:" in err.out


def test_cli_reports_a_stopped_receipt_without_a_floor_table(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    receipt = {
        "issue": 170,
        "stopped": "model call cap 5 reached",
        "passed": False,
        "cases": [],
        "combined": {},
    }
    path = tmp_path / "stopped.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    code = main([str(path)])
    captured = capsys.readouterr()
    assert code == 1
    assert captured.err.strip() == "stopped: model call cap 5 reached"
    assert captured.out == ""


def test_cli_null_stopped_keeps_the_floor_table(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    source = FIXTURE_DIR / "labeled_synthetic_pass.json"
    receipt = json.loads(source.read_text(encoding="utf-8"))
    path = tmp_path / "null_stopped.json"
    path.write_text(json.dumps({**receipt, "stopped": None}), encoding="utf-8")
    assert main([str(path)]) == 0
    assert "accepted: true" in capsys.readouterr().out


def _pass_receipt_with(tmp_path: Path, stopped: object) -> Path:
    source = FIXTURE_DIR / "labeled_synthetic_pass.json"
    receipt = json.loads(source.read_text(encoding="utf-8"))
    path = tmp_path / "stopped_variant.json"
    path.write_text(json.dumps({**receipt, "stopped": stopped}), encoding="utf-8")
    return path


@pytest.mark.parametrize("stopped", [True, {"r": "x"}, 5])
def test_cli_rejects_a_non_string_stopped_as_malformed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], stopped: object
) -> None:
    code = main([str(_pass_receipt_with(tmp_path, stopped))])
    captured = capsys.readouterr()
    assert code == 2
    assert "malformed receipt" in captured.err
    assert captured.out == ""


@pytest.mark.parametrize("stopped", ["", "   "])
def test_cli_reports_a_blank_stopped_as_unspecified(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], stopped: str
) -> None:
    code = main([str(_pass_receipt_with(tmp_path, stopped))])
    captured = capsys.readouterr()
    assert code == 1
    assert captured.err.strip() == "stopped: <unspecified>"
    assert captured.out == ""
