"""Live PSAI consumer matrix under public adapters ([#177][i177]).

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
    - [typevet.evaluation.psai_vision_consumer_live_router][]: router HTTP leg
    - [scripts.psai_vision_consumer_wheel_proof][]: isolated wheel entry

[i177]: https://github.com/Alberto-Codes/typevet/issues/177
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import os
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from typevet.adapters.inbound.settings import LlamaSettings, load_llama_settings
from typevet.evaluation.datasets.psai_vision import VisionSmokeFixture
from typevet.evaluation.datasets.psai_vision_controls import (
    PairedOrdering,
    VisualControl,
    paired_image_ordering,
)
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
from typevet.evaluation.psai_vision_consumer_harness import (
    _EXIT_ACCEPTANCE_FAIL,
    _EXIT_INVALID,
    PROTOCOL_REVISION,
    ConsumerProofResult,
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
    live_gate_action,
    live_skip_reason,
    require_live_enabled,
)

_MODEL_ENV = ("TYPEVET_GEMMA_MODEL", "TYPEVET_LLAMA__DEFAULT_MODEL")
_REPO_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True, slots=True)
class _LiveReceiptAssembly:
    """Inputs for live receipt JSON (keeps function arity small).

    Attributes:
        fixture_root (Path): Committed fixture directory.
        plan (ConsumerCallCounts): Scheduled call counts.
        matrix (ConsumerLiveMatrixResult): Live matrix outputs.
        model (str): Model id under test.
        wheel_sha256 (str | None): Wheel digest when known.
        typevet_install_path (str): Resolved ``typevet.__file__``.
        evidence_kind (str): Evidence label for the receipt.
        negative (dict[str, Any]): Unsupported-template probe outcome.
        pairs (Sequence[PairedOrdering]): Paired ordering summary rows.

    Examples:
        ```python
        from pathlib import Path

        from typevet.evaluation.psai_vision_consumer_live import _LiveReceiptAssembly

        assert _LiveReceiptAssembly.__dataclass_fields__
        ```
    """

    fixture_root: Path
    plan: ConsumerCallCounts
    matrix: ConsumerLiveMatrixResult
    model: str
    wheel_sha256: str | None
    typevet_install_path: str
    evidence_kind: str
    negative: dict[str, Any]
    pairs: Sequence[PairedOrdering]


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


def _require_live_gate(settings: LlamaSettings) -> None:
    reason = live_skip_reason(settings)
    action = live_gate_action(reason)
    if action.name != "RUN":
        msg = reason or "live gate blocked"
        raise ValueError(f"{TYPEVET_REQUIRE_LIVE_ENV}: {msg}")


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


def _live_receipt_payload(assembly: _LiveReceiptAssembly) -> dict[str, Any]:
    scoring_observed = sum(len(row["answers"]) for row in assembly.matrix.matrix_rows)
    try:
        package_version = importlib.metadata.version("typevet")
    except importlib.metadata.PackageNotFoundError:
        package_version = "unknown"
    payload: dict[str, Any] = {
        "consumer_live_protocol_revision": PROTOCOL_REVISION,
        "evidence_kind": assembly.evidence_kind,
        "git_head": _git_head(),
        "wheel_sha256": assembly.wheel_sha256,
        "typevet_version": package_version,
        "typevet_install_path": assembly.typevet_install_path,
        "model": assembly.model,
        "require_live": True,
        "questions_per_judge_call_annotation": ANNOTATION_QUESTIONS_PER_JUDGE_CALL,
        "judgment_call_count": assembly.plan.judgment_calls,
        "scoring_request_count": assembly.plan.scoring_requests,
        "scoring_requests_observed": scoring_observed,
        "auxiliary_http_count": assembly.matrix.auxiliary_http,
        "failed_attempts": 0,
        "matrix_rows": assembly.matrix.matrix_rows,
        "health": assembly.matrix.health,
        "capability": {
            "vision": assembly.matrix.capability.vision,
            "marker": assembly.matrix.capability.marker,
        },
        "served_template": assembly.matrix.served.name,
        "unsupported_capability_negative": assembly.negative,
        "paired_ordering": [
            {
                "unique_data_id": p.unique_data_id,
                "ordered": p.ordered,
                "margin": p.margin,
            }
            for p in assembly.pairs
        ],
        "elapsed_s": assembly.matrix.elapsed_s,
        "fixture_root": str(assembly.fixture_root.resolve()),
    }
    payload.update(consumer_fixture_identity_pins(assembly.fixture_root))
    return payload


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
    _require_live_gate(settings)
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
    plan = plan_frozen_consumer_calls(
        visual_control_rows=len(controls),
        auxiliary_http=3,
    )
    enforce_consumer_call_budget(plan, plan)
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
            "checks_failed": failures,
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


def run_live_consumer_proof(
    *,
    fixture_root: Path,
    out_dir: Path | None = None,
    wheel_sha256: str | None = None,
    typevet_install_path: str | None = None,
    evidence_kind: str = "isolated_wheel_live",
) -> ConsumerProofResult:
    """Run the frozen live consumer matrix and optionally write a receipt.

    Args:
        fixture_root: Committed ``vision_smoke`` directory.
        out_dir: When set, write a versioned receipt JSON here (exclusive).
        wheel_sha256: Wheel digest recorded on the receipt.
        typevet_install_path: Resolved ``typevet.__file__`` when known.
        evidence_kind: Label for checkout vs isolated wheel evidence.

    Returns:
        ``ConsumerProofResult`` with exit code and receipt payload.

    Raises:
        ValueError: Live gate blocked or template/model invalid.
        ConsumerCallBudgetError: Auxiliary HTTP exceeded budget.
        ReceiptAlreadyExistsError: Target receipt path already exists.
    """
    if not require_live_enabled():
        msg = f"{TYPEVET_REQUIRE_LIVE_ENV} must be set for live consumer proof"
        raise ValueError(msg)
    settings, model, plan, controls, fixture = _prepare_live_run(fixture_root)
    matrix = run_consumer_live_matrix(
        settings=settings,
        model=model,
        fixture_root=fixture_root,
        auxiliary_budget=plan.auxiliary_http,
    )
    sample_png = (fixture_root / fixture.examples[0].screenshot.file_name).read_bytes()
    negative = run_negative_template_probe(model, sample_png)
    pairs = paired_image_ordering(controls, matrix.probabilities)
    install_path = typevet_install_path or _typevet_install_path()
    receipt = _live_receipt_payload(
        _LiveReceiptAssembly(
            fixture_root=fixture_root,
            plan=plan,
            matrix=matrix,
            model=model,
            wheel_sha256=wheel_sha256,
            typevet_install_path=install_path,
            evidence_kind=evidence_kind,
            negative=negative,
            pairs=pairs,
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
