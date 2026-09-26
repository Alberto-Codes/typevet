"""Offline-first PSAI consumer proof harness ([#177][i177], [#184][i184]).

Examples:
    ```python
    from pathlib import Path

    from typevet.evaluation.psai_vision_consumer_harness import (
        consumer_proof_main,
        run_offline_consumer_proof,
    )

    result = run_offline_consumer_proof(
        fixture_root=Path("tests/fixtures/psai/vision_smoke"),
    )
    assert result.exit_code in {0, 1}
    ```

See Also:
    - [typevet.evaluation.psai_vision_consumer_accounting][]: call budgets
    - [typevet.evaluation.psai_vision_consumer_offline][]: matrix runner

[i177]: https://github.com/Alberto-Codes/typevet/issues/177
[i184]: https://github.com/Alberto-Codes/typevet/issues/184
"""

from __future__ import annotations

import argparse
import importlib.metadata
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from typevet.adapters.inbound.cord_semantic_acceptance_cli import main as cord_cli_main
from typevet.adapters.outbound.gemma import ServedTemplateClass
from typevet.evaluation.datasets.psai_vision_controls import paired_image_ordering
from typevet.evaluation.experiment_identity import (
    ReceiptAlreadyExistsError,
    write_receipt_exclusive,
)
from typevet.evaluation.psai_vision_consumer_accounting import (
    ANNOTATION_QUESTIONS_PER_JUDGE_CALL,
    ConsumerCallBudgetError,
    ConsumerCallCounts,
    enforce_consumer_call_budget,
    plan_frozen_consumer_calls,
)
from typevet.evaluation.psai_vision_consumer_offline import (
    build_offline_consumer_port,
    consumer_fixture_identity_pins,
    frozen_consumer_controls,
    load_frozen_consumer_fixture,
    run_offline_consumer_matrix,
)
from typevet.evaluation.psai_vision_consumer_receipt import (
    consumer_receipt_basename,
    evaluate_consumer_receipt_acceptance,
)

PROTOCOL_REVISION = 2
_EXIT_ACCEPTANCE_FAIL = 1
_EXIT_INVALID = 2


@dataclass(frozen=True, slots=True)
class _OfflineReceiptAssembly:
    """Inputs for ``_offline_receipt_payload`` (keeps arity under the ruff cap).

    Attributes:
        plan (ConsumerCallCounts): Scheduled judgment and scoring totals.
        matrix_rows (list[dict[str, Any]]): Serialized judgment rows.
        pairs (Sequence[Any]): Paired ordering summary rows.
        negative (dict[str, Any]): Unsupported-template probe outcome.
        scoring_observed (int): ``score_candidates`` calls observed.
        fixture_root (Path): Committed fixture directory.
        wheel_sha256 (str | None): Built wheel digest when known.
        typevet_install_path (str | None): Resolved ``typevet.__file__`` when known.

    Examples:
        ```python
        assembly = _OfflineReceiptAssembly(
            plan=ConsumerCallCounts(judgment_calls=14, scoring_requests=16),
            matrix_rows=[],
            pairs=(),
            negative={"ok": True},
            scoring_observed=0,
            fixture_root=Path("tests/fixtures/psai/vision_smoke"),
            wheel_sha256=None,
        )
        assert assembly.plan.judgment_calls == 14
        ```
    """

    plan: ConsumerCallCounts
    matrix_rows: list[dict[str, Any]]
    pairs: Sequence[Any]
    negative: dict[str, Any]
    scoring_observed: int
    fixture_root: Path
    wheel_sha256: str | None
    typevet_install_path: str | None = None


@dataclass(frozen=True, slots=True)
class ConsumerProofResult:
    """Outcome of one harness run.

    Attributes:
        exit_code (int): Process exit (0 pass, 1 acceptance fail, 2 invalid).
        receipt (dict[str, Any]): Serialized receipt payload.
        receipt_path (Path | None): Path written when ``out_dir`` was set.

    Examples:
        ```python
        result = ConsumerProofResult(exit_code=0, receipt={}, receipt_path=None)
        assert result.exit_code == 0
        ```
    """

    exit_code: int
    receipt: dict[str, Any]
    receipt_path: Path | None = None


def _offline_receipt_payload(assembly: _OfflineReceiptAssembly) -> dict[str, Any]:
    try:
        package_version = importlib.metadata.version("typevet")
    except importlib.metadata.PackageNotFoundError:
        package_version = "unknown"
    payload: dict[str, Any] = {
        "consumer_live_protocol_revision": PROTOCOL_REVISION,
        "require_live": False,
        "model": "offline-consumer-fake",
        "wheel_sha256": assembly.wheel_sha256,
        "typevet_version": package_version,
        "typevet_install_path": assembly.typevet_install_path or "unknown",
        "fixture_root": str(assembly.fixture_root),
        "questions_per_judge_call_annotation": ANNOTATION_QUESTIONS_PER_JUDGE_CALL,
        "judgment_call_count": assembly.plan.judgment_calls,
        "scoring_request_count": assembly.plan.scoring_requests,
        "scoring_requests_observed": assembly.scoring_observed,
        "auxiliary_http_count": 0,
        "failed_attempts": 0,
        "capability": {"vision": True, "offline_stub": True},
        "served_template": ServedTemplateClass.NATIVE_GEMMA4_TURN.name,
        "unsupported_capability_negative": assembly.negative,
        "matrix_rows": assembly.matrix_rows,
        "paired_ordering": [
            {
                "unique_data_id": p.unique_data_id,
                "ordered": p.ordered,
                "margin": p.margin,
            }
            for p in assembly.pairs
        ],
    }
    payload.update(consumer_fixture_identity_pins(assembly.fixture_root))
    return payload


def run_offline_consumer_proof(
    *,
    fixture_root: Path,
    out_dir: Path | None = None,
    wheel_sha256: str | None = None,
    typevet_install_path: str | None = None,
    force_acceptance_fail: bool = False,
) -> ConsumerProofResult:
    """Run the full offline consumer harness and optionally write a receipt.

    Args:
        fixture_root: Committed ``vision_smoke`` fixture directory.
        out_dir: When set, write a versioned receipt JSON file here.
        wheel_sha256: Optional wheel digest stored on the receipt.
        typevet_install_path: Resolved ``typevet.__file__`` when known.
        force_acceptance_fail: When true, mark paired ordering failed (tests).

    Returns:
        ``ConsumerProofResult`` with exit code and receipt payload.

    Raises:
        FileNotFoundError: Missing fixture manifest.
        ValueError: Missing frozen protocol rows.
        ConsumerCallBudgetError: Schedule exceeds budget.
    """
    fixture = load_frozen_consumer_fixture(fixture_root)
    controls = frozen_consumer_controls(fixture)
    plan = plan_frozen_consumer_calls(visual_control_rows=len(controls))
    enforce_consumer_call_budget(plan, plan)

    port, _controls, fake = build_offline_consumer_port(fixture_root)
    matrix_rows, probabilities, negative = run_offline_consumer_matrix(
        port,
        fixture_root=fixture_root,
    )
    pairs = paired_image_ordering(controls, probabilities)
    if force_acceptance_fail and pairs:
        pairs = tuple(
            replace(p, ordered=False, margin=-1.0) if i == 0 else p
            for i, p in enumerate(pairs)
        )

    receipt = _offline_receipt_payload(
        _OfflineReceiptAssembly(
            plan=plan,
            matrix_rows=matrix_rows,
            pairs=pairs,
            negative=negative,
            scoring_observed=len(fake.calls),
            fixture_root=fixture_root,
            wheel_sha256=wheel_sha256,
            typevet_install_path=typevet_install_path,
        ),
    )
    accepted, failures = evaluate_consumer_receipt_acceptance(receipt)
    if not accepted:
        receipt["failure_attempt"] = {
            "checks_failed": failures,
            "timestamp_unix": time.time(),
        }
    exit_code = 0 if accepted else _EXIT_ACCEPTANCE_FAIL
    receipt_path: Path | None = None
    if out_dir is not None:
        receipt_path = out_dir / consumer_receipt_basename(
            protocol_revision=PROTOCOL_REVISION,
            wheel_sha256=wheel_sha256,
            model_id=str(receipt["model"]),
        )
        write_receipt_exclusive(receipt_path, receipt)
    return ConsumerProofResult(
        exit_code=exit_code, receipt=receipt, receipt_path=receipt_path
    )


def cord_semantic_cli_exit_code(receipt_path: Path) -> int:
    """Run the #184 operator CLI on a saved CORD combined receipt.

    Args:
        receipt_path: Path to combined receipt JSON.

    Returns:
        CLI exit code (0 pass, 1 fail, 2 invalid).
    """
    return cord_cli_main([str(receipt_path)])


def consumer_proof_main(argv: Sequence[str] | None = None) -> int:
    """CLI entry for offline consumer proof.

    Args:
        argv: Optional argument vector; defaults to ``sys.argv[1:]``.

    Returns:
        Exit code 0 on acceptance pass, 1 on acceptance fail, 2 on invalid input.
    """
    parser = argparse.ArgumentParser(description="PSAI consumer proof harness (#177)")
    parser.add_argument(
        "--fixture-root",
        type=Path,
        default=Path("tests/fixtures/psai/vision_smoke"),
        help="Committed vision_smoke directory (no scratchpad graft)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Write a versioned receipt JSON under this directory",
    )
    parser.add_argument(
        "--wheel-sha256",
        default=None,
        help="Optional wheel digest recorded on the receipt",
    )
    parser.add_argument(
        "--force-fail",
        action="store_true",
        help="Test-only: force paired ordering acceptance failure",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        result = run_offline_consumer_proof(
            fixture_root=args.fixture_root.resolve(),
            out_dir=args.out_dir.resolve() if args.out_dir else None,
            wheel_sha256=args.wheel_sha256,
            force_acceptance_fail=args.force_fail,
        )
    except (
        FileNotFoundError,
        ValueError,
        ConsumerCallBudgetError,
        ReceiptAlreadyExistsError,
    ) as exc:
        print(f"FAIL_CLOSED: {exc}", file=sys.stderr)
        return _EXIT_INVALID
    if result.exit_code != 0:
        failed = result.receipt.get("failure_attempt", {})
        for msg in failed.get("checks_failed") or []:
            print(f"FAIL: {msg}", file=sys.stderr)
    elif result.receipt_path is not None:
        print(f"wrote {result.receipt_path}")
    return result.exit_code
