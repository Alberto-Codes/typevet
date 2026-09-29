"""Instruction-variant consumer proof harness ([#177][i177], moved in #256 E4).

The harness runs two frozen instruction arms over the PSAI vision smoke
fixture, through the offline scripted port or a live llama.cpp server. It
writes a versioned receipt and a replay comparison. This package re-exports
the names that callers outside the package use. The ``proof`` module is the
``python -m`` entry and is not imported here, so that ``runpy`` does not load
it twice.

Attributes:
    __all__ (list[str]): Public harness names re-exported from the submodules.

Examples:
    ```python
    from typevet_evals.instruction_variant import run_variant_matrix
    ```

    ```bash
    uv run python -m typevet_evals.instruction_variant.proof \
        --fixture-root tests/fixtures/psai/vision_smoke
    ```

See Also:
    - [typevet_evals.instruction_variant.proof][]: command-line entry
    - [typevet_evals.instruction_variant.offline][]: offline orchestration
    - [typevet_evals.instruction_variant.live][]: live orchestration
    - [typevet_evals.outcome_replay_metrics][]: replay comparison metrics

[i177]: https://github.com/Alberto-Codes/typevet/issues/177
"""

from __future__ import annotations

from typevet_evals.instruction_variant.live import (
    live_instruction_variant_proof_main,
    run_live_instruction_variant_proof,
)
from typevet_evals.instruction_variant.live_router import run_live_variant_matrix
from typevet_evals.instruction_variant.matrix import gold_labels, probe_invalid_model
from typevet_evals.instruction_variant.offline import (
    InstructionVariantProofResult,
    run_offline_instruction_variant_proof,
)
from typevet_evals.instruction_variant.protocol import VariantDispatchLedger
from typevet_evals.instruction_variant.receipt import (
    acceptance_failures,
    descriptive_replay_label,
)
from typevet_evals.instruction_variant.run import VariantMatrixRun, run_variant_matrix

__all__ = [
    "InstructionVariantProofResult",
    "VariantDispatchLedger",
    "VariantMatrixRun",
    "acceptance_failures",
    "descriptive_replay_label",
    "gold_labels",
    "live_instruction_variant_proof_main",
    "probe_invalid_model",
    "run_live_instruction_variant_proof",
    "run_live_variant_matrix",
    "run_offline_instruction_variant_proof",
    "run_variant_matrix",
]
