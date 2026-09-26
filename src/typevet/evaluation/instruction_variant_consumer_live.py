"""Live instruction-variant consumer proof ([#177][i177], [#174][i174]).

Examples:
    ```python
    from pathlib import Path

    from typevet.evaluation.instruction_variant_consumer_live import (
        run_live_instruction_variant_proof,
    )

    result = run_live_instruction_variant_proof(
        fixture_root=Path("tests/fixtures/psai/vision_smoke"),
    )
    assert result.exit_code in {0, 1, 2}
    ```

See Also:
    - [typevet.evaluation.instruction_variant_consumer_live_router][]: router matrix
    - [typevet.evaluation.instruction_variant_consumer_offline][]: offline orchestration

When ``out_dir`` is set, ``finalize_variant_proof`` writes an exclusive receipt
with ``evidence_kind`` ``instruction_variant_live`` and fail-closed collision
handling.

[i174]: https://github.com/Alberto-Codes/typevet/issues/174
[i177]: https://github.com/Alberto-Codes/typevet/issues/177
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from typevet.adapters.inbound.settings import load_llama_settings
from typevet.evaluation.experiment_identity import ReceiptAlreadyExistsError
from typevet.evaluation.instruction_variant_consumer_live_router import (
    resolve_variant_live_model,
    run_live_variant_matrix,
)
from typevet.evaluation.instruction_variant_consumer_offline import (
    InstructionVariantProofResult,
    finalize_variant_proof,
)
from typevet.evaluation.instruction_variant_consumer_protocol import (
    DEFAULT_CANDIDATE_INSTRUCTION,
    DEFAULT_SEED_INSTRUCTION,
    plan_instruction_variant_calls,
)
from typevet.evaluation.instruction_variant_consumer_receipt import (
    VariantProofRunContext,
    variant_receipt_basename,
)
from typevet.evaluation.runner.live_gate import (
    TYPEVET_REQUIRE_LIVE_ENV,
    live_skip_reason,
    require_live_enabled,
)

_EXIT_INVALID = 2


def run_live_instruction_variant_proof(
    *,
    fixture_root: Path,
    seed_instruction: str = DEFAULT_SEED_INSTRUCTION,
    candidate_instruction: str = DEFAULT_CANDIDATE_INSTRUCTION,
    wheel_sha256: str | None = None,
    typevet_install_path: str | None = None,
    out_dir: Path | None = None,
) -> InstructionVariantProofResult:
    """Run live variant matrix when ``TYPEVET_REQUIRE_LIVE`` and router allow.

    Args:
        fixture_root: Committed ``vision_smoke`` directory.
        seed_instruction: Baseline instruction text.
        candidate_instruction: Candidate instruction text.
        wheel_sha256: Wheel digest recorded on the receipt.
        typevet_install_path: Resolved ``typevet.__file__`` when known.
        out_dir: When set, write an exclusive receipt JSON here.

    Returns:
        Proof result, or exit code ``2`` when live is required but blocked.
    """
    if not require_live_enabled():
        return InstructionVariantProofResult(
            exit_code=_EXIT_INVALID,
            receipt={"acceptance_failures": [f"{TYPEVET_REQUIRE_LIVE_ENV} not set"]},
            replay_report={},
        )
    plan = plan_instruction_variant_calls()
    run = run_live_variant_matrix(
        fixture_root=fixture_root,
        seed_instruction=seed_instruction,
        candidate_instruction=candidate_instruction,
    )
    if run is None:
        reason = live_skip_reason(load_llama_settings()) or "live gate blocked"
        return InstructionVariantProofResult(
            exit_code=_EXIT_INVALID,
            receipt={"acceptance_failures": [reason]},
            replay_report={},
        )
    model_id = resolve_variant_live_model()
    return finalize_variant_proof(
        plan,
        run,
        seed_instruction=seed_instruction,
        candidate_instruction=candidate_instruction,
        context=VariantProofRunContext(
            model_id=model_id,
            wheel_sha256=wheel_sha256,
            typevet_install_path=typevet_install_path,
            out_dir=out_dir,
            evidence_kind="instruction_variant_live",
            require_live=True,
        ),
    )


def live_instruction_variant_proof_main(argv: Sequence[str] | None = None) -> int:
    """CLI for live instruction-variant proof (requires ``TYPEVET_REQUIRE_LIVE=1``).

    Returns:
        Exit code ``0`` on acceptance pass, ``1`` on fail, ``2`` on invalid input.
    """
    parser = argparse.ArgumentParser(
        description="Instruction-variant consumer live proof (#177 slice 4)",
    )
    parser.add_argument(
        "--fixture-root",
        type=Path,
        default=Path("tests/fixtures/psai/vision_smoke"),
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--wheel-sha256", default=None)
    parser.add_argument("--typevet-install-path", default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        result = run_live_instruction_variant_proof(
            fixture_root=args.fixture_root.resolve(),
            wheel_sha256=args.wheel_sha256,
            typevet_install_path=args.typevet_install_path,
            out_dir=args.out_dir.resolve() if args.out_dir else None,
        )
    except ReceiptAlreadyExistsError as exc:
        print(f"FAIL_CLOSED: {exc}", file=sys.stderr)
        return _EXIT_INVALID
    if result.exit_code != 0:
        for msg in result.receipt.get("acceptance_failures") or []:
            print(f"FAIL: {msg}", file=sys.stderr)
    elif result.receipt_path is not None:
        print(f"wrote {result.receipt_path}")
        print(
            variant_receipt_basename(
                protocol_revision=int(
                    result.receipt["instruction_variant_protocol_revision"]
                ),
                wheel_sha256=args.wheel_sha256,
                model_id=str(result.receipt["model"]),
            )
        )
    return result.exit_code
