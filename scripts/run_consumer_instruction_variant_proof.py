r"""Run instruction-variant consumer proof ([#177][i177]).

Default path builds a wheel, installs it in an isolated cwd outside the
checkout, and runs the public proof API. Pass ``--dev`` for in-tree iteration.

Examples:
    ```console
    $ uv run python scripts/run_consumer_instruction_variant_proof.py
    $ uv run python scripts/run_consumer_instruction_variant_proof.py --dev
    ```

See Also:
    - [scripts.consumer_instruction_variant_proof][]: wheel-isolated runner
    - [typevet_evals.instruction_variant.proof][]: in-tree CLI
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from typevet_evals.instruction_variant.proof import proof_main

_WHEEL_PROOF = (
    Path(__file__).resolve().with_name("consumer_instruction_variant_proof.py")
)


def _wheel_main(argv: list[str]) -> int:
    spec = importlib.util.spec_from_file_location(
        "consumer_instruction_variant_proof",
        _WHEEL_PROOF,
    )
    if spec is None or spec.loader is None:
        msg = f"could not load {_WHEEL_PROOF}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return int(module.main(argv))


def main(argv: list[str] | None = None) -> int:
    """Dispatch wheel-isolated proof unless ``--dev`` is present.

    Returns:
        Harness exit code.
    """
    args = list(argv if argv is not None else sys.argv[1:])
    if args and args[0] == "--dev":
        return proof_main(args[1:])
    return _wheel_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
