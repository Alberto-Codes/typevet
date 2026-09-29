"""Unit tests for evaluation run identity fingerprints ([#186][i186])."""

from __future__ import annotations

from pathlib import Path

import pytest

from typevet_evals.experiment_identity import (
    ExperimentIdentity,
    ExperimentIdentityRequest,
    PromptSpec,
    RuntimeBuild,
    WorkingTreeState,
    begin_run_identity,
    capture_experiment_identity,
    capture_working_tree_at_run_start,
    cord_expense_receipt_path,
    identity_digest,
    prompt_digest,
    working_tree_from_porcelain,
)

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[3]


def _baseline_tree() -> WorkingTreeState:
    return WorkingTreeState(
        baseline_commit="deadbeef" * 5,
        dirty=False,
        dirty_paths=(),
        dirty_digest="",
    )


def _runtime() -> RuntimeBuild:
    return RuntimeBuild(
        model="gemma-3-4b-it-q4km-mm",
        served_template="native_gemma3_turn",
        server_build="unknown",
    )


def _request(
    *,
    prompts: tuple[PromptSpec, ...],
    tree: WorkingTreeState,
    arm_call_counts: dict[str, int] | None = None,
) -> ExperimentIdentityRequest:
    return ExperimentIdentityRequest(
        repo_root=_REPO,
        prompts=prompts,
        code_paths={
            "cord_expense": _REPO / "evals/src/typevet_evals/datasets/cord_expense.py"
        },
        fixture_paths={
            "manifest": _REPO / "tests/fixtures/cord/expense_smoke/manifest.json"
        },
        runtime=_runtime(),
        arm_call_counts=arm_call_counts or {"combined": 18},
        working_tree=tree,
    )


def test_two_prompt_variants_at_same_baseline_do_not_share_identity_digest() -> None:
    """Two instruction variants must not collide on ``identity_digest``."""
    tree = _baseline_tree()
    prompt_a = PromptSpec(
        name="expense",
        label_order=("insufficient_evidence", "mismatch", "match"),
        instructions="variant A",
        criteria={"match": "equals total"},
    )
    prompt_b = PromptSpec(
        name="expense",
        label_order=("insufficient_evidence", "mismatch", "match"),
        instructions="variant B",
        criteria={"match": "equals total"},
    )
    id_a = capture_experiment_identity(
        _request(prompts=(prompt_a,), tree=tree), run_id="run-a"
    )
    id_b = capture_experiment_identity(
        _request(prompts=(prompt_b,), tree=tree), run_id="run-b"
    )
    assert id_a.run_id != id_b.run_id
    assert identity_digest(id_a) != identity_digest(id_b)


def test_dirty_working_tree_changes_identity_digest() -> None:
    """Dirty bytes must change ``identity_digest`` at a fixed commit."""
    clean = WorkingTreeState("c" * 40, False, (), "")
    dirty = WorkingTreeState(
        "c" * 40,
        True,
        ("evals/src/typevet_evals/datasets/cord_expense.py",),
        "abc",
    )
    prompt = PromptSpec(
        name="expense",
        label_order=("a",),
        instructions="same",
        criteria={"a": "rule"},
    )
    assert identity_digest(
        capture_experiment_identity(
            _request(prompts=(prompt,), tree=clean, arm_call_counts={"text_only": 1}),
            run_id="fixed",
        )
    ) != identity_digest(
        capture_experiment_identity(
            _request(prompts=(prompt,), tree=dirty, arm_call_counts={"text_only": 1}),
            run_id="fixed",
        )
    )


def test_prompt_digest_covers_label_order_instructions_and_criteria() -> None:
    """Label order is part of the prompt fingerprint."""
    base = PromptSpec("q", ("l1", "l2"), "inst", {"l1": "a", "l2": "b"})
    reordered = PromptSpec("q", ("l2", "l1"), "inst", {"l1": "a", "l2": "b"})
    assert prompt_digest(base) != prompt_digest(reordered)


def test_injected_working_tree_pins_baseline_commit() -> None:
    """Callers inject the working-tree fingerprint; capture does not call git."""
    tree = WorkingTreeState("a" * 40, False, (), "")
    identity = capture_experiment_identity(
        _request(
            prompts=(PromptSpec("expense", ("a",), "i", {"a": "c"}),),
            tree=tree,
            arm_call_counts={"combined": 0},
        )
    )
    assert identity.baseline_commit == "a" * 40
    assert identity.working_tree.baseline_commit == identity.baseline_commit


def test_capture_working_tree_at_run_start_reports_dirty_checkout() -> None:
    """Injected porcelain must mark the captured working tree dirty."""
    tree = capture_working_tree_at_run_start(
        _REPO,
        porcelain=" M tests/unit/test_experiment_identity.py\n",
    )
    assert tree.baseline_commit != "unknown"
    assert tree.dirty is True
    assert "tests/unit/test_experiment_identity.py" in tree.dirty_paths


def test_begin_run_identity_freezes_run_id_and_runtime() -> None:
    """Run-start capture keeps the runtime labels and assigns a run id."""
    runtime = _runtime()
    tree = WorkingTreeState("a" * 40, True, ("src/foo.py",), "digest")
    start = begin_run_identity(
        repo_root=_REPO,
        runtime=runtime,
        working_tree=tree,
        run_id="fixed-run",
    )
    assert start.run_id == "fixed-run"
    assert start.runtime == runtime
    assert start.working_tree == tree


def test_working_tree_from_porcelain_marks_dirty_for_staged_and_unstaged() -> None:
    """Porcelain lines must set dirty and list repo-relative paths."""
    porcelain = " M src/foo.py\nA  new.txt\n?? untracked.py\n"
    tree = working_tree_from_porcelain(
        baseline_commit="b" * 40,
        porcelain=porcelain,
        repo_root=_REPO,
    )
    assert tree.dirty is True
    assert "src/foo.py" in tree.dirty_paths
    assert "new.txt" in tree.dirty_paths
    assert "untracked.py" in tree.dirty_paths
    assert tree.dirty_digest


def test_working_tree_from_porcelain_clean_when_empty() -> None:
    """An empty porcelain string means a clean tree at the baseline commit."""
    tree = working_tree_from_porcelain(
        baseline_commit="c" * 40,
        porcelain="",
        repo_root=_REPO,
    )
    assert tree.dirty is False
    assert tree.dirty_paths == ()
    assert tree.dirty_digest == ""


def test_cord_expense_receipt_path_is_unique_per_attempt() -> None:
    """Each run attempt gets its own receipt filename under the output dir."""
    out = _REPO / "scratchpad" / "cord-expense"
    first = cord_expense_receipt_path(out, "aaa")
    second = cord_expense_receipt_path(out, "bbb")
    assert first != second
    assert first.name == "receipt-aaa.json"
    assert second.name == "receipt-bbb.json"


def test_runtime_receipt_records_unknown_projector_and_config() -> None:
    """Runtime pins unknown template/projector/config fields honestly."""
    identity = ExperimentIdentity(
        run_id="run-1",
        baseline_commit="b" * 40,
        working_tree=_baseline_tree(),
        prompt_digests={"expense": "p" * 64},
        code_path_digests={"cord_expense": "c" * 64},
        fixture_digests={"manifest": "f" * 64},
        runtime=RuntimeBuild(
            model="gemma-4-31b-kv9-q4km-mm",
            served_template="native_gemma4_turn",
            server_build="unknown",
        ),
        arm_call_counts={"combined": 18},
    )
    runtime = identity.to_receipt_mapping()["runtime"]
    assert isinstance(runtime, dict)
    assert runtime["template_identity"] == "unknown"
    assert runtime["projector_identity"] == "unknown"
    assert runtime["config_identity"] == "unknown"


def test_to_receipt_mapping_includes_arm_call_counts_and_runtime_unknown() -> None:
    """Receipt JSON carries call counts and an unknown server build."""
    identity = ExperimentIdentity(
        run_id="run-1",
        baseline_commit="b" * 40,
        working_tree=_baseline_tree(),
        prompt_digests={"expense": "p" * 64},
        code_path_digests={"cord_expense": "c" * 64},
        fixture_digests={"manifest": "f" * 64},
        runtime=_runtime(),
        arm_call_counts={"text_only": 18, "combined": 18},
    )
    receipt = identity.to_receipt_mapping()
    runtime = receipt["runtime"]
    assert receipt["run_id"] == "run-1"
    assert isinstance(runtime, dict)
    assert runtime["server_build"] == "unknown"
    assert receipt["arm_call_counts"] == {"text_only": 18, "combined": 18}
    assert receipt["identity_digest"] == identity_digest(identity)
