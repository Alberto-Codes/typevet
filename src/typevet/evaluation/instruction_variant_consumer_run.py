"""Execute instruction-variant matrix legs ([#177][i177]).

Examples:
    ```python
    from pathlib import Path

    from typevet.evaluation.instruction_variant_consumer_run import (
        run_variant_matrix,
    )

    run = run_variant_matrix(
        fixture_root=Path("tests/fixtures/psai/vision_smoke"),
        seed_instruction="seed",
        candidate_instruction="candidate",
        model_id="offline-fake",
    )
    assert run.ledger.judgment_calls == 4
    ```

See Also:
    - [typevet.evaluation.instruction_variant_consumer_matrix][]: matrix helpers
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from typevet.adapters.outbound.gemma import ServedTemplateClass
from typevet.adapters.outbound.judgment_scoring import ScoringJudgmentAdapter
from typevet.evaluation.datasets.psai_vision_controls import VisualControl
from typevet.evaluation.instruction_variant_consumer_matrix import (
    build_offline_variant_port,
    probe_invalid_model,
    run_negative_probe,
    run_variant_arm,
    slice_present_controls,
)
from typevet.evaluation.instruction_variant_consumer_protocol import (
    VariantDispatchLedger,
)
from typevet.evaluation.outcome_replay_metrics import SavedPromptOutcome
from typevet.evaluation.psai_vision_consumer_offline import load_frozen_consumer_fixture
from typevet.testing import ScriptedScoringFake


@dataclass(frozen=True, slots=True)
class VariantMatrixRun:
    """Matrix outputs retained for receipt assembly.

    Attributes:
        fixture_root (Path): Committed fixture directory.
        controls (tuple[VisualControl, ...]): Slice control rows.
        ledger (VariantDispatchLedger): Observed dispatch counters.
        invalid (dict[str, Any]): Invalid-model probe payload.
        negative (dict[str, Any]): Unsupported-template probe payload.
        seed_rows (list[dict[str, Any]]): Seed arm judgment rows.
        candidate_rows (list[dict[str, Any]]): Candidate arm rows.
        seed_outcomes (dict[str, SavedPromptOutcome]): Seed distributions.
        candidate_outcomes (dict[str, SavedPromptOutcome]): Candidate distributions.

    Examples:
        ```python
        assert VariantMatrixRun.__dataclass_fields__
        ```
    """

    fixture_root: Path
    controls: tuple[VisualControl, ...]
    ledger: VariantDispatchLedger
    invalid: dict[str, Any]
    negative: dict[str, Any]
    seed_rows: list[dict[str, Any]]
    candidate_rows: list[dict[str, Any]]
    seed_outcomes: dict[str, SavedPromptOutcome]
    candidate_outcomes: dict[str, SavedPromptOutcome]


def run_variant_matrix(
    *,
    fixture_root: Path,
    seed_instruction: str,
    candidate_instruction: str,
    model_id: str,
    read_image: Callable[[str], bytes] | None = None,
) -> VariantMatrixRun:
    """Execute offline variant matrix legs and negative probes.

    Returns:
        Retained rows, outcomes, and dispatch ledger for receipt assembly.
    """
    fixture = load_frozen_consumer_fixture(fixture_root)
    controls = slice_present_controls(fixture)
    loader = read_image or (lambda name: (fixture_root / name).read_bytes())
    ledger = VariantDispatchLedger()
    probe_port = ScoringJudgmentAdapter(
        ScriptedScoringFake(logprobs={"True": -0.2, "False": -1.0}),
        tokenize_content=lambda text: (ord(text[0]),) if text else (),
        served_template=ServedTemplateClass.NATIVE_GEMMA4_TURN,
    )
    invalid = probe_invalid_model(probe_port)
    if invalid["ok"]:
        ledger.record_failure()
    port = build_offline_variant_port(controls, ledger)
    negative = run_negative_probe(model_id, fixture_root, loader, fixture)
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
