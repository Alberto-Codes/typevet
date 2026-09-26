"""CLI shim for instruction-variant consumer proof ([#177][i177]).

Examples:
    ```bash
    uv run python -m typevet.evaluation.instruction_variant_consumer_proof \
        --fixture-root tests/fixtures/psai/vision_smoke
    ```

See Also:
    - [typevet.evaluation.instruction_variant_consumer_offline][]: offline runner
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from typevet.evaluation.instruction_variant_consumer_offline import (
    InstructionVariantProofResult,
    run_offline_instruction_variant_proof,
)

__all__ = [
    "InstructionVariantProofResult",
    "proof_main",
    "run_offline_instruction_variant_proof",
]


def proof_main(argv: Sequence[str] | None = None) -> int:
    """CLI for the instruction-variant consumer proof.

    Returns:
        Exit code ``0`` when acceptance passes.
    """
    parser = argparse.ArgumentParser(
        description="Instruction-variant consumer proof (#177 slice 4)",
    )
    parser.add_argument(
        "--fixture-root",
        type=Path,
        default=Path("tests/fixtures/psai/vision_smoke"),
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--wheel-sha256", default=None)
    parser.add_argument("--typevet-install-path", default=None)
    args = parser.parse_args(list(argv if argv is not None else sys.argv[1:]))
    result = run_offline_instruction_variant_proof(
        fixture_root=args.fixture_root,
        wheel_sha256=args.wheel_sha256,
        typevet_install_path=args.typevet_install_path,
    )
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result.receipt, indent=2), encoding="utf-8")
    summary = {
        "exit_code": result.exit_code,
        "verdict": result.receipt.get("comparison_verdict"),
    }
    print(json.dumps(summary))
    return result.exit_code


if __name__ == "__main__":
    raise SystemExit(proof_main())
