"""Receipt assembly and acceptance for instruction-variant proof ([#177][i177]).

Examples:
    ```python
    from typevet.evaluation.instruction_variant_consumer_receipt import (
        comparison_verdict,
    )

    assert comparison_verdict({"candidate_improved": False}) != "candidate_improved"
    ```

See Also:
    - [typevet.evaluation.instruction_variant_consumer_offline][]: orchestration
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from typevet.evaluation.instruction_variant_consumer_protocol import (
    FROZEN_VARIANT_SCORING_REQUESTS,
    INSTRUCTION_VARIANT_PROTOCOL_REVISION,
    VariantDispatchLedger,
)
from typevet.evaluation.outcome_replay_metrics import SavedPromptOutcome
from typevet.evaluation.psai_vision_consumer_accounting import ConsumerCallCounts


@dataclass(frozen=True, slots=True)
class VariantReceiptAssembly:
    """Inputs for ``assemble_variant_receipt``.

    Attributes:
        fixture_root (Path): Committed fixture directory.
        plan (ConsumerCallCounts): Scheduled call totals.
        ledger (VariantDispatchLedger): Observed dispatch counters.
        seed_instruction (str): Baseline instruction text.
        candidate_instruction (str): Candidate instruction text.
        seed_rows (Sequence[Mapping[str, Any]]): Seed matrix rows.
        candidate_rows (Sequence[Mapping[str, Any]]): Candidate matrix rows.
        seed_outcomes (Mapping[str, SavedPromptOutcome]): Seed distributions.
        candidate_outcomes (Mapping[str, SavedPromptOutcome]): Candidate distributions.
        gold (Mapping[str, str]): Gold labels per case id.
        invalid (Mapping[str, Any]): Invalid-model probe payload.
        negative (Mapping[str, Any]): Unsupported-template probe payload.
        replay_report (Mapping[str, Any]): Offline compare block.
        replay_check (bool): Idempotent replay flag.
        acceptance_failures (Sequence[str]): Human-readable failures.
        wheel_sha256 (str | None): Wheel digest when isolated.
        typevet_install_path (str | None): Resolved install path when known.

    Examples:
        ```python
        assert VariantReceiptAssembly.__dataclass_fields__
        ```
    """

    fixture_root: Path
    plan: ConsumerCallCounts
    ledger: VariantDispatchLedger
    seed_instruction: str
    candidate_instruction: str
    seed_rows: Sequence[Mapping[str, Any]]
    candidate_rows: Sequence[Mapping[str, Any]]
    seed_outcomes: Mapping[str, SavedPromptOutcome]
    candidate_outcomes: Mapping[str, SavedPromptOutcome]
    gold: Mapping[str, str]
    invalid: Mapping[str, Any]
    negative: Mapping[str, Any]
    replay_report: Mapping[str, Any]
    replay_check: bool
    acceptance_failures: Sequence[str]
    wheel_sha256: str | None
    typevet_install_path: str | None


def comparison_verdict(replay_report: Mapping[str, Any]) -> str:
    """Summarize seed vs candidate replay without overstating improvement.

    Returns:
        One of ``candidate_improved``, ``tie``, ``candidate_worse_or_mixed``,
        or ``insufficient_valid_metrics``.
    """
    if replay_report.get("candidate_improved") is True:
        return "candidate_improved"
    delta = replay_report.get("delta_candidate_minus_seed") or {}
    brier = delta.get("mean_brier")
    loss = delta.get("mean_log_loss")
    if brier is None or loss is None:
        return "insufficient_valid_metrics"
    if brier == 0.0 and loss == 0.0:
        return "tie"
    if (isinstance(brier, float) and brier > 0) or (
        isinstance(loss, float) and loss > 0
    ):
        return "candidate_worse_or_mixed"
    return "tie"


def acceptance_failures(
    *,
    invalid: Mapping[str, Any],
    negative: Mapping[str, Any],
    budget_ok: bool,
    budget_error: str | None,
    ledger: VariantDispatchLedger,
    replay_check: bool,
) -> list[str]:
    """Return human-readable acceptance failures for one proof run.

    Returns:
        Empty list when the proof passes acceptance checks.
    """
    failures: list[str] = []
    if not invalid["ok"]:
        failures.append("invalid model probe did not fail closed")
    if not negative["ok"]:
        failures.append("unsupported template probe did not fail closed")
    if not budget_ok:
        failures.append(budget_error or "call budget mismatch")
    if ledger.scoring_requests > FROZEN_VARIANT_SCORING_REQUESTS:
        failures.append("scoring_requests exceed hard cap")
    if not replay_check:
        failures.append("replay metrics not idempotent")
    if ledger.failed_attempts < 1:
        failures.append("expected at least one retained failed_attempt")
    return failures


def assemble_variant_receipt(assembly: VariantReceiptAssembly) -> dict[str, Any]:
    """Build the JSON receipt for one proof run.

    Returns:
        JSON-serializable receipt mapping.
    """
    return {
        "instruction_variant_protocol_revision": INSTRUCTION_VARIANT_PROTOCOL_REVISION,
        "fixture_root": str(assembly.fixture_root.resolve()),
        "instruction_variants": {
            "seed": assembly.seed_instruction,
            "candidate": assembly.candidate_instruction,
        },
        "judgment_call_count": assembly.plan.judgment_calls,
        "scoring_request_count": assembly.plan.scoring_requests,
        "scoring_requests_observed": assembly.ledger.scoring_requests,
        "failed_attempts": assembly.ledger.failed_attempts,
        "matrix_rows": list(assembly.seed_rows) + list(assembly.candidate_rows),
        "invalid_model_probe": dict(assembly.invalid),
        "unsupported_template_probe": dict(assembly.negative),
        "gold_labels": dict(assembly.gold),
        "saved_outcomes": {
            "seed": {
                key: {"probabilities": dict(val.probabilities)}
                for key, val in assembly.seed_outcomes.items()
            },
            "candidate": {
                key: {"probabilities": dict(val.probabilities)}
                for key, val in assembly.candidate_outcomes.items()
            },
        },
        "replay_report": dict(assembly.replay_report),
        "replay_idempotent": assembly.replay_check,
        "comparison_verdict": comparison_verdict(assembly.replay_report),
        "wheel_sha256": assembly.wheel_sha256,
        "typevet_install_path": assembly.typevet_install_path,
        "harness_entrypoint": "scripts/run_consumer_instruction_variant_proof.py",
        "acceptance_failures": list(assembly.acceptance_failures),
    }
