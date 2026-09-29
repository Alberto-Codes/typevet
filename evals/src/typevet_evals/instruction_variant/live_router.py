"""Budgeted router dispatch for instruction-variant live matrix ([#177][i177]).

Examples:
    ```python
    from pathlib import Path

    from typevet_evals.instruction_variant.live_router import (
        run_live_variant_matrix,
    )

    run = run_live_variant_matrix(
        fixture_root=Path("tests/fixtures/psai/vision_smoke"),
        seed_instruction="seed",
        candidate_instruction="candidate",
    )
    assert run is None or run.ledger.judgment_calls == 4
    ```

See Also:
    - [typevet_evals.instruction_variant.live][]: receipt orchestration
    - [typevet.adapters.outbound.llama_cpp.gemma_native_vision_factory][]: native vision factory

``run_live_variant_matrix`` opens a scoring-backed judgment port when the live
gate and native Gemma template class allow; otherwise it returns ``None``.

[i177]: https://github.com/Alberto-Codes/typevet/issues/177
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx

from typevet.adapters.inbound.settings import load_llama_settings
from typevet.adapters.outbound.llama_cpp.gemma_native_vision_factory import (
    open_gemma_native_vision_judgment,
)
from typevet.evaluation.psai_vision_consumer_dispatch import wrap_scoring_port
from typevet.evaluation.psai_vision_consumer_offline import load_frozen_consumer_fixture
from typevet.evaluation.runner.live_gate import require_live_enabled
from typevet.ports.judgment import JudgmentPort
from typevet_evals.instruction_variant.matrix import (
    _LedgerJudgmentPort as LedgerJudgmentPort,
)
from typevet_evals.instruction_variant.matrix import (
    probe_invalid_model,
    run_negative_probe,
    run_variant_arm,
    slice_present_controls,
)
from typevet_evals.instruction_variant.protocol import (
    VariantDispatchLedger,
)
from typevet_evals.instruction_variant.run import VariantMatrixRun

_MODEL_ENV = ("TYPEVET_GEMMA_MODEL", "TYPEVET_LLAMA__DEFAULT_MODEL")


def resolve_variant_live_model() -> str:
    """Return the model id for one live instruction-variant matrix.

    Returns:
        Model id from env or llama settings.
    """
    for key in _MODEL_ENV:
        val = os.environ.get(key)
        if val:
            return val.strip()
    settings = load_llama_settings()
    return settings.multimodal_model


def _resolve_model() -> str:
    return resolve_variant_live_model()


def _live_variant_arms(
    port: JudgmentPort,
    *,
    controls: Any,
    fixture: Any,
    loader: Callable[[str], bytes],
    model_id: str,
    seed_instruction: str,
    candidate_instruction: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    seed_rows, seed_outcomes = run_variant_arm(
        port,
        controls=controls,
        fixture=fixture,
        loader=loader,
        model_id=model_id,
        instruction=seed_instruction,
        arm="seed",
    )
    candidate_rows, candidate_outcomes = run_variant_arm(
        port,
        controls=controls,
        fixture=fixture,
        loader=loader,
        model_id=model_id,
        instruction=candidate_instruction,
        arm="candidate",
    )
    return seed_rows, seed_outcomes, candidate_rows, candidate_outcomes


def run_live_variant_matrix(
    *,
    fixture_root: Path,
    seed_instruction: str,
    candidate_instruction: str,
    ledger: VariantDispatchLedger | None = None,
) -> VariantMatrixRun | None:
    """Execute the opted-in matrix with counted factory metadata and scoring.

    Args:
        fixture_root: Frozen fixture image directory.
        seed_instruction: Caller instruction for the seed arm.
        candidate_instruction: Caller instruction for the candidate arm.
        ledger: Optional external ledger that retains attempts when setup fails.

    Opens ``open_gemma_native_vision_judgment`` for the configured router and
    reuses ``session.port`` for all variant arms.

    Returns:
        Matrix run payload, or ``None`` when the live gate skips dispatch.
    """
    settings = load_llama_settings()
    if not require_live_enabled():
        return None
    model_id = _resolve_model()
    fixture = load_frozen_consumer_fixture(fixture_root)
    controls = slice_present_controls(fixture)
    loader = lambda name: (fixture_root / name).read_bytes()
    ledger = ledger or VariantDispatchLedger()
    with httpx.Client(base_url=settings.base_url, timeout=settings.timeout) as client:
        client.event_hooks["request"].append(ledger.before_http)
        with open_gemma_native_vision_judgment(
            settings=settings,
            model=model_id,
            http_client=client,
            scoring_port_wrapper=lambda scoring: wrap_scoring_port(scoring, ledger),
        ) as session:
            bare = session.port
            invalid = probe_invalid_model(bare)
            if invalid["ok"]:
                ledger.record_failure()
            port = LedgerJudgmentPort(bare, ledger)
            negative = run_negative_probe(model_id, fixture_root, loader, fixture)
            seed_rows, seed_outcomes, candidate_rows, candidate_outcomes = (
                _live_variant_arms(
                    port,
                    controls=controls,
                    fixture=fixture,
                    loader=loader,
                    model_id=model_id,
                    seed_instruction=seed_instruction,
                    candidate_instruction=candidate_instruction,
                )
            )
    return VariantMatrixRun(
        fixture_root=fixture_root,
        controls=controls,
        ledger=ledger,
        invalid=invalid,
        negative=negative,
        seed_rows=seed_rows,
        candidate_rows=candidate_rows,
        seed_outcomes=seed_outcomes,
        candidate_outcomes=candidate_outcomes,
    )
