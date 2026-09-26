"""CORD smoke receipt wiring uses evaluated-input snapshots ([#186][i186])."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.live import test_cord_expense_smoke_live as cord_live
from typevet.evaluation.experiment_identity import (
    ReceiptAlreadyExistsError,
    RunIdentityStart,
    RuntimeBuild,
    WorkingTreeState,
    begin_run_identity,
    finalize_experiment_identity,
    snapshot_evaluated_inputs,
    write_receipt_exclusive,
)

pytestmark = pytest.mark.contract

_RECEIPT_KEYS = tuple(
    f"receipt_{rid}" for rid in ("R01", "R02", "R03", "R04", "R05", "R06")
)


def test_cord_snapshot_includes_all_receipt_png_digests_before_scoring() -> None:
    """Pre-scoring snapshot must pin every receipt PNG used by _receipt_image."""
    evaluated = cord_live._snapshot_evaluated_inputs_before_scoring()
    assert "expense_smoke_manifest" in evaluated.fixture_digests
    for key in _RECEIPT_KEYS:
        assert key in evaluated.fixture_digests, key
        assert len(evaluated.fixture_digests[key]) == 64


def test_cord_snapshot_keeps_receipt_png_digest_after_post_snapshot_mutation(
    tmp_path: Path,
) -> None:
    """Receipt PNG bytes mutated after snapshot must not change fixture digests."""
    receipt = tmp_path / "R01.png"
    receipt.write_bytes(b"\x89PNG\r\n\x1a\nreceipt-v1")
    evaluated = snapshot_evaluated_inputs(
        prompts=(cord_live._prompt_spec_from_expense_question(),),
        code_paths={},
        fixture_paths={
            "expense_smoke_manifest": cord_live.FIXTURE_DIR / "manifest.json",
            "receipt_R01": receipt,
        },
    )
    digest_at_snapshot = evaluated.fixture_digests["receipt_R01"]
    receipt.write_bytes(b"\x89PNG\r\n\x1a\nreceipt-v2")
    identity = finalize_experiment_identity(
        run_start=begin_run_identity(
            repo_root=cord_live._REPO_ROOT,
            runtime=RuntimeBuild("m", "native_gemma4_turn", "unknown"),
            working_tree=WorkingTreeState("b" * 40, False, (), ""),
            run_id="png-mutate",
        ),
        evaluated=evaluated,
        arm_call_counts={"combined": 1},
    )
    assert identity.fixture_digests["receipt_R01"] == digest_at_snapshot


def test_cord_write_receipt_uses_exclusive_writer_and_snapshot_finalize(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Harness path snapshots before scoring and writes receipts exclusively."""
    monkeypatch.setattr(cord_live, "_OUTPUT_DIR", tmp_path)
    code_copy = tmp_path / "cord_expense_smoke_live.py"
    code_copy.write_text("# harness stub\n", encoding="utf-8")
    fixture_copy = tmp_path / "manifest.json"
    fixture_copy.write_text('{"cases": []}\n', encoding="utf-8")
    run_start: RunIdentityStart = begin_run_identity(
        repo_root=cord_live._REPO_ROOT,
        runtime=RuntimeBuild("m", "native_gemma4_turn", "unknown"),
        working_tree=WorkingTreeState("b" * 40, False, (), ""),
        run_id="cord-attempt",
    )
    evaluated = snapshot_evaluated_inputs(
        prompts=(cord_live._prompt_spec_from_expense_question(),),
        code_paths={"cord_expense_smoke_live": code_copy},
        fixture_paths={"expense_smoke_manifest": fixture_copy},
    )
    digest_at_snapshot = evaluated.code_path_digests["cord_expense_smoke_live"]
    code_copy.write_text("# mutated after snapshot\n", encoding="utf-8")
    identity = finalize_experiment_identity(
        run_start=run_start,
        evaluated=evaluated,
        arm_call_counts={"text_only": 18, "image_only": 6, "combined": 18},
    )
    assert identity.code_path_digests["cord_expense_smoke_live"] == digest_at_snapshot
    receipt = {
        "experiment_identity": identity.to_receipt_mapping(),
        "requests": 42,
    }
    cord_live._write_receipt(run_start, receipt)
    path = cord_live.cord_expense_receipt_path(tmp_path, run_start.run_id)
    assert json.loads(path.read_text(encoding="utf-8"))["requests"] == 42
    with pytest.raises(ReceiptAlreadyExistsError):
        write_receipt_exclusive(path, {"requests": 99})
