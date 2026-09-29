"""Snapshot and exclusive receipt writer for experiment identity ([#186][i186])."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from typevet_evals.experiment_identity import (
    ExperimentIdentityRequest,
    PromptSpec,
    ReceiptAlreadyExistsError,
    RunIdentityStart,
    RuntimeBuild,
    WorkingTreeState,
    begin_run_identity,
    capture_experiment_identity,
    finalize_experiment_identity,
    snapshot_evaluated_inputs,
    write_receipt_exclusive,
)

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[3]


def _tree() -> WorkingTreeState:
    return WorkingTreeState("a" * 40, False, (), "")


def _runtime() -> RuntimeBuild:
    return RuntimeBuild("model", "native_gemma3_turn", "unknown")


def _run_start() -> RunIdentityStart:
    return begin_run_identity(
        repo_root=_REPO,
        runtime=_runtime(),
        working_tree=_tree(),
        run_id="run-fixed",
    )


def test_finalize_keeps_snapshot_digests_after_receipt_image_mutation(
    tmp_path: Path,
) -> None:
    """Mutating a receipt PNG after snapshot must not change fixture digests."""
    receipt = tmp_path / "R01.png"
    receipt.write_bytes(b"\x89PNG\r\n\x1a\nreceipt-bytes-v1")
    fixture = tmp_path / "manifest.json"
    fixture.write_text("{}", encoding="utf-8")
    prompt = PromptSpec("expense", ("a",), "inst", {"a": "rule"})
    evaluated = snapshot_evaluated_inputs(
        prompts=(prompt,),
        code_paths={},
        fixture_paths={
            "expense_smoke_manifest": fixture,
            "receipt_R01": receipt,
        },
    )
    digest_before = evaluated.fixture_digests["receipt_R01"]
    receipt.write_bytes(b"\x89PNG\r\n\x1a\nreceipt-bytes-v2")
    identity = finalize_experiment_identity(
        run_start=_run_start(),
        evaluated=evaluated,
        arm_call_counts={"combined": 1},
    )
    assert identity.fixture_digests["receipt_R01"] == digest_before


def test_finalize_keeps_snapshot_digests_after_source_mutation(tmp_path: Path) -> None:
    """Mutating evaluated files after snapshot must not change receipt digests."""
    source = tmp_path / "evaluated.py"
    source.write_text("version-one", encoding="utf-8")
    fixture = tmp_path / "fixture.json"
    fixture.write_text('{"v": 1}', encoding="utf-8")
    prompt = PromptSpec("expense", ("a",), "inst", {"a": "rule"})
    evaluated = snapshot_evaluated_inputs(
        prompts=(prompt,),
        code_paths={"evaluated": source},
        fixture_paths={"fixture": fixture},
    )
    digest_before = evaluated.code_path_digests["evaluated"]
    source.write_text("version-two", encoding="utf-8")
    fixture.write_text('{"v": 2}', encoding="utf-8")
    identity = finalize_experiment_identity(
        run_start=_run_start(),
        evaluated=evaluated,
        arm_call_counts={"combined": 18},
    )
    assert identity.code_path_digests["evaluated"] == digest_before
    assert identity.fixture_digests["fixture"] == evaluated.fixture_digests["fixture"]


def test_write_receipt_exclusive_leaves_first_bytes_on_duplicate(
    tmp_path: Path,
) -> None:
    """A second write for the same run id must fail and keep the first receipt."""
    path = tmp_path / "receipt-run-fixed.json"
    first = {"run_id": "run-fixed", "score": 1}
    second = {"run_id": "run-fixed", "score": 2}
    write_receipt_exclusive(path, first)
    with pytest.raises(ReceiptAlreadyExistsError):
        write_receipt_exclusive(path, second)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded == first


def test_capture_experiment_identity_still_matches_snapshot_finalize() -> None:
    """Legacy capture must agree with snapshot then finalize."""
    prompt = PromptSpec("expense", ("a",), "i", {"a": "c"})
    tree = _tree()
    request = ExperimentIdentityRequest(
        repo_root=_REPO,
        prompts=(prompt,),
        code_paths={
            "cord_expense": _REPO / "src/typevet/evaluation/datasets/cord_expense.py"
        },
        fixture_paths={
            "manifest": _REPO / "tests/fixtures/cord/expense_smoke/manifest.json"
        },
        runtime=_runtime(),
        arm_call_counts={"combined": 1},
        working_tree=tree,
    )
    captured = capture_experiment_identity(request, run_id="same")
    evaluated = snapshot_evaluated_inputs(
        prompts=request.prompts,
        code_paths=request.code_paths,
        fixture_paths=request.fixture_paths,
    )
    finalized = finalize_experiment_identity(
        run_start=begin_run_identity(
            repo_root=_REPO,
            runtime=_runtime(),
            working_tree=tree,
            run_id="same",
        ),
        evaluated=evaluated,
        arm_call_counts=request.arm_call_counts,
    )
    assert captured.prompt_digests == finalized.prompt_digests
    assert captured.code_path_digests == finalized.code_path_digests
    assert captured.fixture_digests == finalized.fixture_digests
