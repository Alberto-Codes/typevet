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

[i174]: https://github.com/Alberto-Codes/typevet/issues/174
[i177]: https://github.com/Alberto-Codes/typevet/issues/177
"""

from __future__ import annotations

from pathlib import Path

from typevet.adapters.inbound.settings import load_llama_settings
from typevet.evaluation.instruction_variant_consumer_live_router import (
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
) -> InstructionVariantProofResult:
    """Run live variant matrix when ``TYPEVET_REQUIRE_LIVE`` and router allow.

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
    result = finalize_variant_proof(
        plan,
        run,
        seed_instruction=seed_instruction,
        candidate_instruction=candidate_instruction,
        wheel_sha256=wheel_sha256,
        typevet_install_path=typevet_install_path,
    )
    receipt = dict(result.receipt)
    receipt["evidence_kind"] = "instruction_variant_live"
    return InstructionVariantProofResult(
        exit_code=result.exit_code,
        receipt=receipt,
        replay_report=result.replay_report,
    )
