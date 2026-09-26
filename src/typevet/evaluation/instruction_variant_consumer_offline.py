"""Offline instruction-variant consumer proof orchestration ([#177][i177]).

Examples:
    ```python
    from pathlib import Path

    from typevet.evaluation.instruction_variant_consumer_offline import (
        run_offline_instruction_variant_proof,
    )

    result = run_offline_instruction_variant_proof(
        fixture_root=Path("tests/fixtures/psai/vision_smoke"),
    )
    assert result.exit_code == 0
    ```

See Also:
    - [typevet.evaluation.instruction_variant_consumer_matrix][]: matrix legs
    - [typevet.evaluation.instruction_variant_consumer_receipt][]: receipt writer
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from typevet.evaluation.instruction_variant_consumer_matrix import gold_labels
from typevet.evaluation.instruction_variant_consumer_protocol import (
    DEFAULT_CANDIDATE_INSTRUCTION,
    DEFAULT_SEED_INSTRUCTION,
    plan_instruction_variant_calls,
)
from typevet.evaluation.instruction_variant_consumer_receipt import (
    VariantReceiptAssembly,
    acceptance_failures,
    assemble_variant_receipt,
)
from typevet.evaluation.instruction_variant_consumer_run import (
    VariantMatrixRun,
    run_variant_matrix,
)
from typevet.evaluation.outcome_replay_metrics import (
    compare_matched_prompt_outcomes,
    replay_identical_reports,
)
from typevet.evaluation.psai_vision_consumer_accounting import (
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

    Examples:
        ```python
        result = InstructionVariantProofResult(
            exit_code=0, receipt={}, replay_report={}
        )
        assert result.exit_code == 0
        ```
    """

    exit_code: int
    receipt: dict[str, Any]
    replay_report: dict[str, Any]


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
    wheel_sha256: str | None,
    typevet_install_path: str | None,
) -> InstructionVariantProofResult:
    """Assemble replay metrics and receipt from a completed matrix run.

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
    receipt = assemble_variant_receipt(
        VariantReceiptAssembly(
            fixture_root=run.fixture_root,
            plan=plan,
            ledger=run.ledger,
            seed_instruction=seed_instruction,
            candidate_instruction=candidate_instruction,
            seed_rows=run.seed_rows,
            candidate_rows=run.candidate_rows,
            seed_outcomes=run.seed_outcomes,
            candidate_outcomes=run.candidate_outcomes,
            gold=gold,
            invalid=run.invalid,
            negative=run.negative,
            replay_report=replay_report,
            replay_check=replay_check,
            acceptance_failures=failures,
            wheel_sha256=wheel_sha256,
            typevet_install_path=typevet_install_path,
        )
    )
    exit_code = 0 if not failures else _EXIT_ACCEPTANCE_FAIL
    return InstructionVariantProofResult(
        exit_code=exit_code,
        receipt=receipt,
        replay_report=replay_report,
    )


def run_offline_instruction_variant_proof(
    *,
    fixture_root: Path,
    seed_instruction: str = DEFAULT_SEED_INSTRUCTION,
    candidate_instruction: str = DEFAULT_CANDIDATE_INSTRUCTION,
    model_id: str = "offline-instruction-variant-fake",
    read_image: Callable[[str], bytes] | None = None,
    wheel_sha256: str | None = None,
    typevet_install_path: str | None = None,
) -> InstructionVariantProofResult:
    """Run the frozen instruction-variant matrix offline with scripted scoring.

    Returns:
        Proof result with replay metrics and retained failures.
    """
    plan = plan_instruction_variant_calls()
    run = run_variant_matrix(
        fixture_root=fixture_root,
        seed_instruction=seed_instruction,
        candidate_instruction=candidate_instruction,
        model_id=model_id,
        read_image=read_image,
    )
    return finalize_variant_proof(
        plan,
        run,
        seed_instruction=seed_instruction,
        candidate_instruction=candidate_instruction,
        wheel_sha256=wheel_sha256,
        typevet_install_path=typevet_install_path,
    )
