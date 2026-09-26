"""Live PSAI consumer matrix with retained dispatch attempts ([#177][i177]).

Examples:
    ```python
    from pathlib import Path

    from typevet.evaluation.psai_vision_consumer_live import run_live_consumer_proof

    result = run_live_consumer_proof(
        fixture_root=Path("tests/fixtures/psai/vision_smoke"),
        wheel_sha256="abc",
    )
    assert result.exit_code in {0, 1, 2}
    ```

See Also:
    - [typevet.evaluation.psai_vision_consumer_live_identity][]: pre-dispatch snapshot
    - [typevet.evaluation.psai_vision_consumer_live_router][]: router HTTP leg
    - [scripts.psai_vision_consumer_wheel_proof][]: isolated wheel entry

Live proof snapshots router identity before the matrix, enforces
independent judgment/scoring/auxiliary budgets via a dispatch ledger,
and may write exclusive failure receipts when dispatch aborts early.

[i177]: https://github.com/Alberto-Codes/typevet/issues/177
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx

from typevet.adapters.inbound.settings import LlamaSettings, load_llama_settings
from typevet.domain.errors import GenerationError, JudgmentError
from typevet.evaluation.datasets.psai_vision import VisionSmokeFixture
from typevet.evaluation.datasets.psai_vision_controls import (
    VisualControl,
    paired_image_ordering,
)
from typevet.evaluation.experiment_identity import (
    ReceiptAlreadyExistsError,
    write_receipt_exclusive,
)
from typevet.evaluation.psai_vision_consumer_accounting import (
    ConsumerCallBudgetError,
    ConsumerCallCounts,
    plan_frozen_consumer_calls,
)
from typevet.evaluation.psai_vision_consumer_dispatch import ConsumerDispatchLedger
from typevet.evaluation.psai_vision_consumer_harness import (
    _EXIT_ACCEPTANCE_FAIL,
    _EXIT_INVALID,
    PROTOCOL_REVISION,
    ConsumerProofResult,
)
from typevet.evaluation.psai_vision_consumer_live_receipt import (
    LiveReceiptContext,
    build_live_receipt_payload,
)
from typevet.evaluation.psai_vision_consumer_live_router import (
    ConsumerLiveMatrixResult,
    run_consumer_live_matrix,
)
from typevet.evaluation.psai_vision_consumer_offline import (
    consumer_fixture_identity_pins,
    frozen_consumer_controls,
    load_frozen_consumer_fixture,
    run_negative_template_probe,
)
from typevet.evaluation.psai_vision_consumer_receipt import (
    consumer_receipt_basename,
    evaluate_consumer_receipt_acceptance,
    resolve_receipt_write_path,
)
from typevet.evaluation.runner.live_gate import (
    TYPEVET_REQUIRE_LIVE_ENV,
    require_live_enabled,
)

_DISPATCH_ERRORS = (
    ConsumerCallBudgetError,
    ValueError,
    httpx.HTTPError,
    GenerationError,
    JudgmentError,
)

_MODEL_ENV = ("TYPEVET_GEMMA_MODEL", "TYPEVET_LLAMA__DEFAULT_MODEL")
_REPO_ROOT = Path(__file__).resolve().parents[3]


def _typevet_install_path() -> str:
    spec = importlib.util.find_spec("typevet")
    if spec is None or not spec.origin:
        return "unknown"
    return str(Path(spec.origin).resolve())


def _git_head() -> str:
    proc = subprocess.run(
        ["/usr/bin/git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
    )
    return proc.stdout.strip() if proc.returncode == 0 else "unknown"


def _resolve_model(settings: LlamaSettings) -> str:
    for key in _MODEL_ENV:
        val = os.environ.get(key)
        if val and val.strip():
            return val.strip()
    if settings.multimodal_model:
        return settings.multimodal_model
    if settings.default_model:
        return settings.default_model
    msg = "no model id in env or settings"
    raise ValueError(msg)


def _prepare_live_run(
    fixture_root: Path,
) -> tuple[
    LlamaSettings,
    str,
    ConsumerCallCounts,
    tuple[VisualControl, ...],
    VisionSmokeFixture,
]:
    settings = load_llama_settings()
    model = _resolve_model(settings)
    timeout = max(settings.timeout, 900.0)
    settings = LlamaSettings(
        base_url=settings.base_url,
        timeout=timeout,
        default_model=settings.default_model,
        multimodal_model=settings.multimodal_model,
    )
    fixture = load_frozen_consumer_fixture(fixture_root)
    controls = frozen_consumer_controls(fixture)
    plan = plan_frozen_consumer_calls(visual_control_rows=len(controls))
    return settings, model, plan, controls, fixture


def _accept_live_receipt(
    receipt: dict[str, Any],
    *,
    out_dir: Path | None,
    wheel_sha256: str | None,
    model: str,
) -> ConsumerProofResult:
    accepted, failures = evaluate_consumer_receipt_acceptance(receipt)
    if not accepted:
        receipt["failure_attempt"] = {
            "checks_failed": receipt.get("failure_attempt", {}).get("checks_failed", [])
            + failures,
            "timestamp_unix": time.time(),
        }
    exit_code = 0 if accepted else _EXIT_ACCEPTANCE_FAIL
    receipt_path: Path | None = None
    if out_dir is not None:
        receipt_path = resolve_receipt_write_path(
            out_dir,
            protocol_revision=PROTOCOL_REVISION,
            wheel_sha256=wheel_sha256,
            model_id=model,
        )
        write_receipt_exclusive(receipt_path, receipt)
    return ConsumerProofResult(
        exit_code=exit_code, receipt=receipt, receipt_path=receipt_path
    )


def _dispatch_failure_receipt(
    *,
    plan: ConsumerCallCounts,
    model: str,
    ledger: ConsumerDispatchLedger,
    matrix: ConsumerLiveMatrixResult | None,
    exc: Exception,
    fixture_root: Path,
) -> dict[str, Any]:
    failure_receipt: dict[str, Any] = {
        "consumer_live_protocol_revision": PROTOCOL_REVISION,
        "require_live": True,
        "model": model,
        "judgment_call_count": plan.judgment_calls,
        "scoring_request_count": plan.scoring_requests,
        "scoring_requests_observed": ledger.scoring_requests,
        "auxiliary_http_count": ledger.auxiliary_http_total,
        "failed_attempts": ledger.failed_attempts,
        "matrix_rows": matrix.matrix_rows if matrix else [],
        "failure_attempt": {
            "checks_failed": [str(exc)],
            "timestamp_unix": time.time(),
        },
    }
    failure_receipt.update(ledger.accounting())
    failure_receipt.update(consumer_fixture_identity_pins(fixture_root))
    return failure_receipt


def run_live_consumer_proof(
    *,
    fixture_root: Path,
    out_dir: Path | None = None,
    wheel_sha256: str | None = None,
    typevet_install_path: str | None = None,
    evidence_kind: str = "isolated_wheel_live",
) -> ConsumerProofResult:
    """Run the frozen matrix and retain successes or known dispatch failures.

    Args:
        fixture_root: Committed ``vision_smoke`` directory.
        out_dir: When set, write a versioned receipt JSON here (exclusive).
        wheel_sha256: Wheel digest recorded on the receipt.
        typevet_install_path: Resolved ``typevet.__file__`` when known.
        evidence_kind: Label for checkout vs isolated wheel evidence.

    Returns:
        ``ConsumerProofResult`` with exit code and receipt payload. Receipts
        merge pre-dispatch identity, ledger counts, and fail-closed acceptance.

    Raises:
        ValueError: Live gate blocked or template/model invalid.
        ConsumerCallBudgetError: Auxiliary HTTP exceeded budget.
        ReceiptAlreadyExistsError: Target receipt path already exists.
    """
    if not require_live_enabled():
        msg = f"{TYPEVET_REQUIRE_LIVE_ENV} must be set for live consumer proof"
        raise ValueError(msg)
    settings, model, plan, controls, fixture = _prepare_live_run(fixture_root)
    ledger = ConsumerDispatchLedger()
    matrix: ConsumerLiveMatrixResult | None = None
    try:
        matrix = run_consumer_live_matrix(
            settings=settings,
            model=model,
            fixture_root=fixture_root,
            ledger=ledger,
        )
    except _DISPATCH_ERRORS as exc:
        failure_receipt = _dispatch_failure_receipt(
            plan=plan,
            model=model,
            ledger=ledger,
            matrix=matrix,
            exc=exc,
            fixture_root=fixture_root,
        )
        return _accept_live_receipt(
            failure_receipt,
            out_dir=out_dir,
            wheel_sha256=wheel_sha256,
            model=model,
        )
    sample_png = (fixture_root / fixture.examples[0].screenshot.file_name).read_bytes()
    negative = run_negative_template_probe(model, sample_png)
    pairs = paired_image_ordering(controls, matrix.probabilities)
    install_path = typevet_install_path or _typevet_install_path()
    receipt = build_live_receipt_payload(
        LiveReceiptContext(
            fixture_root=fixture_root,
            plan=plan,
            matrix=matrix,
            model=model,
            wheel_sha256=wheel_sha256,
            typevet_install_path=install_path,
            evidence_kind=evidence_kind,
            negative=negative,
            pairs=pairs,
            git_head=_git_head(),
        )
    )
    return _accept_live_receipt(
        receipt, out_dir=out_dir, wheel_sha256=wheel_sha256, model=model
    )


def live_consumer_proof_main(argv: Sequence[str] | None = None) -> int:
    """CLI for live consumer proof (requires ``TYPEVET_REQUIRE_LIVE=1``).

    Returns:
        Exit code 0 on acceptance pass, 1 on fail, 2 on invalid input.
    """
    parser = argparse.ArgumentParser(description="PSAI consumer live proof (#177)")
    parser.add_argument(
        "--fixture-root",
        type=Path,
        default=Path("tests/fixtures/psai/vision_smoke"),
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--wheel-sha256", default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        result = run_live_consumer_proof(
            fixture_root=args.fixture_root.resolve(),
            out_dir=args.out_dir.resolve() if args.out_dir else None,
            wheel_sha256=args.wheel_sha256,
        )
    except (
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
        print(
            consumer_receipt_basename(
                protocol_revision=PROTOCOL_REVISION,
                wheel_sha256=args.wheel_sha256,
                model_id=str(result.receipt["model"]),
            )
        )
    return result.exit_code
