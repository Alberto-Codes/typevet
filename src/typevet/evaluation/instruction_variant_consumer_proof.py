"""CLI shim for instruction-variant consumer proof ([#177][i177]).

Examples:
    ```bash
    uv run python -m typevet.evaluation.instruction_variant_consumer_proof \
        --fixture-root tests/fixtures/psai/vision_smoke
    ```

See Also:
    - [typevet.evaluation.instruction_variant_consumer_offline][]: offline runner

``--out-dir`` requests an exclusive receipt path; ``--out`` remains a deprecated
exact-path write for local debugging.

[i177]: https://github.com/Alberto-Codes/typevet/issues/177
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from typevet.evaluation.experiment_identity import ReceiptAlreadyExistsError
from typevet.evaluation.instruction_variant_consumer_offline import (
    InstructionVariantProofResult,
    run_offline_instruction_variant_proof,
)
from typevet.evaluation.instruction_variant_consumer_receipt import (
    VariantProofRunContext,
)

__all__ = [
    "InstructionVariantProofResult",
    "proof_main",
    "run_offline_instruction_variant_proof",
]


def proof_main(argv: Sequence[str] | None = None) -> int:
    """CLI for the instruction-variant consumer proof.

    Returns:
        Exit code ``0`` when acceptance passes, ``1`` on acceptance failure,
        and ``2`` when exclusive receipt write hits ``ReceiptAlreadyExistsError``.
        Prints a JSON summary with ``replay_descriptive_label`` on success paths.
    """
    parser = argparse.ArgumentParser(
        description="Instruction-variant consumer proof (#177 slice 4)",
    )
    parser.add_argument(
        "--fixture-root",
        type=Path,
        default=Path("tests/fixtures/psai/vision_smoke"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Write a versioned exclusive receipt JSON under this directory",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Deprecated: write receipt JSON to this exact path (non-exclusive)",
    )
    parser.add_argument("--wheel-sha256", default=None)
    parser.add_argument("--typevet-install-path", default=None)
    args = parser.parse_args(list(argv if argv is not None else sys.argv[1:]))
    try:
        result = run_offline_instruction_variant_proof(
            fixture_root=args.fixture_root,
            context=VariantProofRunContext(
                model_id="offline-instruction-variant-fake",
                wheel_sha256=args.wheel_sha256,
                typevet_install_path=args.typevet_install_path,
                out_dir=args.out_dir.resolve() if args.out_dir else None,
            ),
        )
    except ReceiptAlreadyExistsError as exc:
        print(f"FAIL_CLOSED: {exc}", file=sys.stderr)
        return 2
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result.receipt, indent=2), encoding="utf-8")
    if result.receipt_path is not None:
        print(f"wrote {result.receipt_path}")
    summary = {
        "exit_code": result.exit_code,
        "replay_descriptive_label": result.receipt.get("replay_descriptive_label"),
    }
    print(json.dumps(summary))
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(proof_main())
