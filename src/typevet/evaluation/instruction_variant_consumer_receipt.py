"""Receipt assembly and acceptance for instruction-variant proof ([#177][i177]).

Examples:
    ```python
    from typevet.evaluation.instruction_variant_consumer_receipt import (
        descriptive_replay_label,
    )

    assert descriptive_replay_label({"shared_valid_case_ids": []}) == (
        "no_shared_valid_cases"
    )
    ```

See Also:
    - [typevet.evaluation.instruction_variant_consumer_offline][]: orchestration

Exclusive commits use ``write_receipt_exclusive`` with manifest and image pins
from ``variant_fixture_identity_pins``. ``persist_variant_receipt`` enriches
run metadata when ``VariantProofRunContext.out_dir`` is set.

[i177]: https://github.com/Alberto-Codes/typevet/issues/177
"""

from __future__ import annotations

import importlib.metadata
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from typevet.evaluation.experiment_identity import write_receipt_exclusive
from typevet.evaluation.instruction_variant_consumer_matrix import gold_labels
from typevet.evaluation.instruction_variant_consumer_protocol import (
    FROZEN_VARIANT_SCORING_REQUESTS,
    INSTRUCTION_VARIANT_PROTOCOL_REVISION,
    VariantDispatchLedger,
)
from typevet.evaluation.instruction_variant_consumer_run import VariantMatrixRun
from typevet.evaluation.outcome_replay_metrics import SavedPromptOutcome
from typevet.evaluation.psai_vision_consumer_accounting import ConsumerCallCounts
from typevet.evaluation.psai_vision_consumer_offline import (
    consumer_fixture_identity_pins,
)

_VARIANT_HARNESS = "scripts/run_consumer_instruction_variant_proof.py"


@dataclass(frozen=True, slots=True)
class VariantFinalizeInputs:
    """Inputs to build a variant receipt assembly from a matrix run.

    Attributes:
        seed_instruction (str): Baseline instruction text.
        candidate_instruction (str): Candidate instruction text.
        context (VariantProofRunContext): Run metadata and optional out dir.
        replay_report (dict[str, Any]): Offline compare block.
        replay_check (bool): Idempotent replay flag.
        failures (list[str]): Acceptance failure messages.

    Examples:
        ```python
        inputs = VariantFinalizeInputs(
            seed_instruction="a",
            candidate_instruction="b",
            context=VariantProofRunContext(model_id="m"),
            replay_report={},
            replay_check=True,
            failures=[],
        )
        assert inputs.failures == []
        ```
    """

    seed_instruction: str
    candidate_instruction: str
    context: VariantProofRunContext
    replay_report: dict[str, Any]
    replay_check: bool
    failures: list[str]


@dataclass(frozen=True, slots=True)
class VariantProofRunContext:
    """Run metadata for variant receipt assembly and optional commit.

    Attributes:
        model_id (str): Model id on the receipt.
        wheel_sha256 (str | None): Wheel digest when isolated.
        typevet_install_path (str | None): Resolved install path when known.
        out_dir (Path | None): When set, write an exclusive receipt here.
        evidence_kind (str): Evidence label on the receipt.
        require_live (bool): Whether live gate was required for this run.

    Examples:
        ```python
        ctx = VariantProofRunContext(model_id="offline-instruction-variant-fake")
        assert ctx.require_live is False
        ```
    """

    model_id: str
    wheel_sha256: str | None = None
    typevet_install_path: str | None = None
    out_dir: Path | None = None
    evidence_kind: str = "instruction_variant_offline"
    require_live: bool = False


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
        model (str): Model id recorded on the receipt.

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
    model: str


def variant_fixture_identity_pins(fixture_root: Path) -> dict[str, Any]:
    """Return manifest and image pins for instruction-variant receipts.

    Returns:
        Pin block with the slice harness entrypoint.
    """
    pins = dict(consumer_fixture_identity_pins(fixture_root))
    pins["harness_entrypoint"] = _VARIANT_HARNESS
    return pins


def variant_receipt_basename(
    *,
    protocol_revision: int,
    wheel_sha256: str | None,
    model_id: str,
) -> str:
    """Return a stable instruction-variant receipt filename.

    Returns:
        Filename including ``.json`` suffix.
    """
    wheel_part = (wheel_sha256 or "unknown-wheel")[:16]
    model_slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", model_id)[:48]
    return (
        f"instruction-variant-receipt-p{protocol_revision}-"
        f"{wheel_part}-{model_slug}.json"
    )


def resolve_variant_receipt_write_path(
    out_dir: Path,
    *,
    protocol_revision: int,
    wheel_sha256: str | None,
    model_id: str,
) -> Path:
    """Pick a non-colliding variant receipt path under ``out_dir``.

    Returns:
        Path that does not yet exist (appends ``-attempt-N`` when needed).

    Raises:
        OSError: When no free attempt suffix is available.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / variant_receipt_basename(
        protocol_revision=protocol_revision,
        wheel_sha256=wheel_sha256,
        model_id=model_id,
    )
    if not base.exists():
        return base
    for attempt in range(1, 1000):
        candidate = base.with_name(base.stem + f"-attempt-{attempt}" + base.suffix)
        if not candidate.exists():
            return candidate
    msg = f"could not allocate variant receipt path under {out_dir}"
    raise OSError(msg)


def enrich_variant_receipt_metadata(
    receipt: dict[str, Any],
    *,
    fixture_root: Path,
    model: str,
    evidence_kind: str,
    require_live: bool,
    git_head: str,
) -> dict[str, Any]:
    """Merge identity pins and run metadata onto a variant receipt.

    Returns:
        Updated receipt mapping (mutates and returns ``receipt``).
    """
    try:
        package_version = importlib.metadata.version("typevet")
    except importlib.metadata.PackageNotFoundError:
        package_version = "unknown"
    receipt["model"] = model
    receipt["evidence_kind"] = evidence_kind
    receipt["require_live"] = require_live
    receipt["git_head"] = git_head
    receipt["typevet_version"] = package_version
    receipt.update(variant_fixture_identity_pins(fixture_root))
    return receipt


def persist_variant_receipt(
    receipt: dict[str, Any],
    *,
    fixture_root: Path,
    context: VariantProofRunContext,
    git_head: str,
) -> Path | None:
    """Enrich and optionally write one variant receipt.

    Returns:
        Receipt path when ``context.out_dir`` is set; otherwise ``None``.
    """
    enrich_variant_receipt_metadata(
        receipt,
        fixture_root=fixture_root,
        model=context.model_id,
        evidence_kind=context.evidence_kind,
        require_live=context.require_live,
        git_head=git_head,
    )
    if context.out_dir is None:
        return None
    return write_variant_receipt_exclusive(
        context.out_dir,
        receipt,
        wheel_sha256=context.wheel_sha256,
        model_id=context.model_id,
    )


def write_variant_receipt_exclusive(
    out_dir: Path,
    receipt: Mapping[str, Any],
    *,
    wheel_sha256: str | None,
    model_id: str,
) -> Path:
    """Write one variant receipt under ``out_dir`` (fail if path exists).

    Returns:
        Path written.
    """
    path = resolve_variant_receipt_write_path(
        out_dir,
        protocol_revision=INSTRUCTION_VARIANT_PROTOCOL_REVISION,
        wheel_sha256=wheel_sha256,
        model_id=model_id,
    )
    write_receipt_exclusive(path, receipt)
    return path


def descriptive_replay_label(replay_report: Mapping[str, Any]) -> str:
    """Return a non-promotional label for a descriptive replay report.

    Returns:
        ``no_shared_valid_cases`` when pairing has no shared valid distributions,
        otherwise ``shared_metrics_computed``. Legacy receipts with
        ``candidate_improved`` are labeled ``legacy_promotion_schema`` for readers
        only; new reports must use ``replay_report_schema`` descriptive v2.
    """
    if replay_report.get("candidate_improved") is not None:
        return "legacy_promotion_schema"
    shared = replay_report.get("shared_valid_case_ids")
    if not shared:
        return "no_shared_valid_cases"
    return "shared_metrics_computed"


def comparison_verdict(replay_report: Mapping[str, Any]) -> str:
    """Deprecated alias for historical receipt readers.

    Returns:
        Same string as ``descriptive_replay_label``.
    """
    return descriptive_replay_label(replay_report)


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


def variant_receipt_assembly_from_run(
    plan: ConsumerCallCounts,
    run: VariantMatrixRun,
    inputs: VariantFinalizeInputs,
) -> VariantReceiptAssembly:
    """Build receipt assembly from one completed variant matrix run.

    Returns:
        ``VariantReceiptAssembly`` ready for ``assemble_variant_receipt``.
    """
    context = inputs.context
    return VariantReceiptAssembly(
        fixture_root=run.fixture_root,
        plan=plan,
        ledger=run.ledger,
        seed_instruction=inputs.seed_instruction,
        candidate_instruction=inputs.candidate_instruction,
        seed_rows=run.seed_rows,
        candidate_rows=run.candidate_rows,
        seed_outcomes=run.seed_outcomes,
        candidate_outcomes=run.candidate_outcomes,
        gold=gold_labels(run.controls),
        invalid=run.invalid,
        negative=run.negative,
        replay_report=inputs.replay_report,
        replay_check=inputs.replay_check,
        acceptance_failures=inputs.failures,
        wheel_sha256=context.wheel_sha256,
        typevet_install_path=context.typevet_install_path,
        model=context.model_id,
    )


def assemble_variant_receipt(assembly: VariantReceiptAssembly) -> dict[str, Any]:
    """Build the JSON receipt for one proof run.

    Returns:
        JSON-serializable receipt with matrix rows, replay metrics,
        ``replay_descriptive_label``, wheel digest fields, and
        ``acceptance_failures`` (empty when acceptance passes).
    """
    return {
        "instruction_variant_protocol_revision": INSTRUCTION_VARIANT_PROTOCOL_REVISION,
        "model": assembly.model,
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
        "replay_descriptive_label": descriptive_replay_label(assembly.replay_report),
        "replay_report_schema": assembly.replay_report.get("replay_report_schema"),
        "wheel_sha256": assembly.wheel_sha256,
        "typevet_install_path": assembly.typevet_install_path,
        "harness_entrypoint": _VARIANT_HARNESS,
        "acceptance_failures": list(assembly.acceptance_failures),
    }
