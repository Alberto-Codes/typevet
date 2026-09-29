"""Offline instruction-variant consumer proof orchestration ([#177][i177]).

Examples:
    ```python
    from pathlib import Path

    from typevet_evals.instruction_variant.offline import (
        run_offline_instruction_variant_proof,
    )

    result = run_offline_instruction_variant_proof(
        fixture_root=Path("tests/fixtures/psai/vision_smoke"),
    )
    assert result.exit_code == 0
    ```

See Also:
    - [typevet_evals.instruction_variant.matrix][]: matrix legs
    - [typevet_evals.instruction_variant.receipt][]: receipt writer
    - [typevet_evals.psai_vision_consumer.accounting][]: call budgets

``finalize_variant_proof`` compares saved outcomes, assembles the receipt, and
calls ``persist_variant_receipt`` when ``VariantProofRunContext.out_dir`` is set.

[i177]: https://github.com/Alberto-Codes/typevet/issues/177
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from typevet_evals.instruction_variant.matrix import gold_labels
from typevet_evals.instruction_variant.protocol import (
    DEFAULT_CANDIDATE_INSTRUCTION,
    DEFAULT_SEED_INSTRUCTION,
    plan_instruction_variant_calls,
)
from typevet_evals.instruction_variant.receipt import (
    VariantFinalizeInputs,
    VariantProofRunContext,
    acceptance_failures,
    assemble_variant_receipt,
    persist_variant_receipt,
    variant_receipt_assembly_from_run,
)
from typevet_evals.instruction_variant.run import (
    VariantMatrixRun,
    run_variant_matrix,
)
from typevet_evals.outcome_replay_metrics import (
    compare_matched_prompt_outcomes,
    replay_identical_reports,
)
from typevet_evals.psai_vision_consumer.accounting import (
    ConsumerCallBudgetError,
    ConsumerCallCounts,
    enforce_consumer_call_budget,
)

_NOUL_LABELS: tuple[str, ...] = ("false", "true")
_EXIT_ACCEPTANCE_FAIL = 1


@dataclass(frozen=True, slots=True)
class InstructionVariantProofResult:
    """Outcome of one instruction-variant consumer proof run.

    Attributes:
        exit_code (int): Process exit code.
        receipt (dict[str, Any]): JSON-serializable proof artifact.
        replay_report (dict[str, Any]): Offline metrics compare block.
        receipt_path (Path | None): Exclusive receipt path when written.

    Examples:
        ```python
        result = InstructionVariantProofResult(
            exit_code=0, receipt={}, replay_report={}, receipt_path=None
        )
        assert result.exit_code == 0
        ```
    """

    exit_code: int
    receipt: dict[str, Any]
    replay_report: dict[str, Any]
    receipt_path: Path | None = None


_REPO_ROOT = Path(__file__).resolve().parents[4]


def _git_head() -> str:
    proc = subprocess.run(
        ["/usr/bin/git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
    )
    return proc.stdout.strip() if proc.returncode == 0 else "unknown"


def _replay_report(
    gold: dict[str, str],
    seed_outcomes: dict[str, Any],
    candidate_outcomes: dict[str, Any],
) -> tuple[dict[str, Any], bool]:
    report = compare_matched_prompt_outcomes(
        gold,
        seed_outcomes,
        candidate_outcomes,
        labels=_NOUL_LABELS,
    )
    second = compare_matched_prompt_outcomes(
        gold,
        seed_outcomes,
        candidate_outcomes,
        labels=_NOUL_LABELS,
    )
    return dict(report), replay_identical_reports(report, second)


def _enforce_budget(
    plan: ConsumerCallCounts,
    run: VariantMatrixRun,
) -> tuple[bool, str | None]:
    observed = ConsumerCallCounts(
        judgment_calls=run.ledger.judgment_calls,
        scoring_requests=run.ledger.scoring_requests,
        failed_attempts=run.ledger.failed_attempts,
    )
    try:
        enforce_consumer_call_budget(observed, plan)
    except ConsumerCallBudgetError as exc:
        return False, str(exc)
    return True, None


def finalize_variant_proof(
    plan: ConsumerCallCounts,
    run: VariantMatrixRun,
    *,
    seed_instruction: str,
    candidate_instruction: str,
    context: VariantProofRunContext,
) -> InstructionVariantProofResult:
    """Assemble replay metrics and receipt from a completed matrix run.

    Args:
        plan: Scheduled call totals.
        run: Completed matrix payload.
        seed_instruction: Baseline instruction text.
        candidate_instruction: Candidate instruction text.
        context: Model, wheel, and optional exclusive receipt directory.

    Returns:
        Proof result with acceptance failures when checks fail.
    """
    gold = gold_labels(run.controls)
    replay_report, replay_check = _replay_report(
        gold, run.seed_outcomes, run.candidate_outcomes
    )
    budget_ok, budget_error = _enforce_budget(plan, run)
    failures = acceptance_failures(
        invalid=run.invalid,
        negative=run.negative,
        budget_ok=budget_ok,
        budget_error=budget_error,
        ledger=run.ledger,
        replay_check=replay_check,
    )
    inputs = VariantFinalizeInputs(
        seed_instruction=seed_instruction,
        candidate_instruction=candidate_instruction,
        context=context,
        replay_report=replay_report,
        replay_check=replay_check,
        failures=failures,
    )
    receipt = assemble_variant_receipt(
        variant_receipt_assembly_from_run(plan, run, inputs)
    )
    receipt_path = persist_variant_receipt(
        receipt,
        fixture_root=run.fixture_root,
        context=context,
        git_head=_git_head(),
    )
    exit_code = 0 if not failures else _EXIT_ACCEPTANCE_FAIL
    return InstructionVariantProofResult(
        exit_code=exit_code,
        receipt=receipt,
        replay_report=replay_report,
        receipt_path=receipt_path,
    )


def run_offline_instruction_variant_proof(
    *,
    fixture_root: Path,
    seed_instruction: str = DEFAULT_SEED_INSTRUCTION,
    candidate_instruction: str = DEFAULT_CANDIDATE_INSTRUCTION,
    read_image: Callable[[str], bytes] | None = None,
    context: VariantProofRunContext | None = None,
) -> InstructionVariantProofResult:
    """Run the frozen instruction-variant matrix offline with scripted scoring.

    Args:
        fixture_root: Committed ``vision_smoke`` directory.
        seed_instruction: Baseline instruction text.
        candidate_instruction: Candidate instruction text.
        read_image: Optional image loader override for tests.
        context: Model, wheel digest, and optional exclusive receipt directory.

    Returns:
        Proof result with replay metrics and retained failures.
    """
    run_context = context or VariantProofRunContext(
        model_id="offline-instruction-variant-fake",
    )
    plan = plan_instruction_variant_calls()
    run = run_variant_matrix(
        fixture_root=fixture_root,
        seed_instruction=seed_instruction,
        candidate_instruction=candidate_instruction,
        model_id=run_context.model_id,
        read_image=read_image,
    )
    return finalize_variant_proof(
        plan,
        run,
        seed_instruction=seed_instruction,
        candidate_instruction=candidate_instruction,
        context=run_context,
    )
