r"""Run the PSAI consumer proof harness ([#177][i177]).

Default path builds a wheel, installs it in an isolated cwd outside the
checkout, and runs the public consumer proof API. Pass ``--dev`` to run the
in-tree harness only (tests and local iteration).

Examples:
    ```console
    $ uv run python scripts/run_psai_vision_consumer_proof.py
    $ uv run python scripts/run_psai_vision_consumer_proof.py --dev --out-dir scratchpad/receipts
    ```

See Also:
    - [scripts.psai_vision_consumer_wheel_proof][]: wheel-isolated runner
    - [typevet.evaluation.psai_vision_consumer_harness][]: in-tree harness
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from typevet.evaluation.psai_vision_consumer_harness import consumer_proof_main

_WHEEL_PROOF = Path(__file__).resolve().with_name("psai_vision_consumer_wheel_proof.py")


def _wheel_main(argv: list[str]) -> int:
    spec = importlib.util.spec_from_file_location(
        "psai_vision_consumer_wheel_proof",
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
        return consumer_proof_main(args[1:])
    return _wheel_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
