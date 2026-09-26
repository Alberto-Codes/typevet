"""Unit tests for evaluation run identity fingerprints ([#186][i186])."""

from __future__ import annotations

from pathlib import Path

import pytest

from typevet.evaluation.experiment_identity import (
    ExperimentIdentity,
    ExperimentIdentityRequest,
    PromptSpec,
    RuntimeBuild,
    WorkingTreeState,
    capture_experiment_identity,
    identity_digest,
    prompt_digest,
)

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[2]


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
            "cord_expense": _REPO / "src/typevet/evaluation/datasets/cord_expense.py"
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
        ("src/typevet/evaluation/datasets/cord_expense.py",),
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
