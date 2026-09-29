"""Ledger ports and matrix legs for instruction-variant proof ([#177][i177]).

Examples:
    ```python
    from pathlib import Path

    from typevet_evals.instruction_variant.matrix import (
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
    - [typevet_evals.instruction_variant.offline][]: orchestration
    - [typevet_evals.psai_vision_consumer.offline][]: consumer fixture helpers
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from typevet.adapters.outbound.gemma import ServedTemplateClass
from typevet.adapters.outbound.judgment_scoring import ScoringJudgmentAdapter
from typevet.domain.candidate_scoring_request import CandidateScoringRequest
from typevet.domain.candidate_scoring_response import CandidateScoringResult
from typevet.domain.errors import JudgmentValidationError
from typevet.domain.judgment_questions import Noul
from typevet.domain.judgment_response import JudgmentResponse
from typevet.domain.media import ImageInput
from typevet.evaluation.datasets.psai_vision import (
    VisionSmokeFixture,
    example_image_input,
)
from typevet.evaluation.datasets.psai_vision_controls import (
    CONDITION_PRESENT,
    VISUAL_QUESTION_NAME,
    VisualControl,
    control_matrix,
    visual_question,
)
from typevet.ports.judgment import JudgmentPort
from typevet.ports.scoring import CandidateScoringPort
from typevet_evals.instruction_variant.protocol import (
    FROZEN_VARIANT_CASE_UIDS,
    VariantDispatchLedger,
)
from typevet_evals.outcome_replay_metrics import SavedPromptOutcome
from typevet_evals.psai_vision_consumer.offline import (
    SequentialConsumerScoringFake,
    run_negative_template_probe,
)
from typevet_evals.psai_vision_consumer.receipt import serialize_answer


class _LedgerScoringPort:
    """Count scoring requests against the variant ledger.

    Attributes:
        _inner (CandidateScoringPort): Wrapped scoring port.
        _ledger (VariantDispatchLedger): Budget ledger.

    Examples:
        ```python
        from typevet.testing import ScriptedScoringFake

        _LedgerScoringPort(ScriptedScoringFake(logprobs={"True": -0.1}), ledger)
        ```
    """

    def __init__(
        self,
        inner: CandidateScoringPort,
        ledger: VariantDispatchLedger,
    ) -> None:
        self._inner = inner
        self._ledger = ledger

    def score_candidates(
        self, request: CandidateScoringRequest
    ) -> CandidateScoringResult:
        """Reserve scoring attempts and count successes separately.

        Raises:
            Exception: Re-raise dispatch failures after retaining their attempt.

        Returns:
            Scoring result from the inner port.
        """
        self._ledger.before_scoring()
        try:
            result = self._inner.score_candidates(request)
        except Exception:
            self._ledger.record_failure()
            raise
        self._ledger.record_scoring_success()
        return result


class _LedgerJudgmentPort:
    """Count judgment calls against the variant ledger.

    Attributes:
        _inner (JudgmentPort): Wrapped judgment port.
        _ledger (VariantDispatchLedger): Budget ledger.

    Examples:
        ```python
        assert _LedgerJudgmentPort.__name__
        ```
    """

    def __init__(self, inner: JudgmentPort, ledger: VariantDispatchLedger) -> None:
        self._inner = inner
        self._ledger = ledger

    def judge(
        self,
        state: str | dict[str, Any] | list[Any],
        questions: Mapping[str, Any],
        model: str,
        *,
        media: tuple[ImageInput, ...] | None = None,
    ) -> JudgmentResponse:
        """Reserve judgment attempts and count returned responses separately.

        Returns:
            Judgment response from the inner port.

        Raises:
            Exception: Re-raises validation or transport errors after counting.
        """
        self._ledger.before_judgment()
        try:
            response = self._inner.judge(state, questions, model, media=media)
        except Exception:
            self._ledger.record_failure()
            raise
        self._ledger.record_judgment_success()
        return response


def slice_present_controls(fixture: VisionSmokeFixture) -> tuple[VisualControl, ...]:
    """Return present-image controls for the two frozen slice cases.

    Returns:
        Two ``present`` controls sorted by ``unique_data_id``.

    Raises:
        ValueError: When a pinned case is missing from the manifest.
    """
    controls = [
        row
        for row in control_matrix(fixture.examples)
        if row.unique_data_id in FROZEN_VARIANT_CASE_UIDS
        and row.condition == CONDITION_PRESENT
    ]
    controls.sort(key=lambda row: row.unique_data_id)
    if len(controls) != len(FROZEN_VARIANT_CASE_UIDS):
        msg = "slice fixture missing one or more present controls"
        raise ValueError(msg)
    return tuple(controls)


def gold_labels(controls: Sequence[VisualControl]) -> dict[str, str]:
    """Map case ids to ``true``/``false`` gold strings.

    Returns:
        Gold label per ``unique_data_id``.
    """
    return {
        control.unique_data_id: "true" if control.expected else "false"
        for control in controls
    }


def _noul_with_instruction(instructions: str) -> Noul:
    base = visual_question()
    return Noul(instructions=instructions, criteria=base.criteria)


def _saved_outcome(probability_true: float) -> SavedPromptOutcome:
    p_true = float(probability_true)
    return SavedPromptOutcome(
        probabilities={"false": 1.0 - p_true, "true": p_true},
    )


def _variant_logprob_maps(
    controls: Sequence[VisualControl],
    *,
    arm: str,
) -> list[dict[str, float]]:
    maps: list[dict[str, float]] = []
    for control in controls:
        p_true = (
            (0.65 if control.expected else 0.45)
            if arm == "seed"
            else (0.80 if control.expected else 0.55)
        )
        maps.append(
            {"True": math.log(p_true), "False": math.log(1.0 - p_true)},
        )
    return maps


def _media_for_control(
    control: VisualControl,
    fixture: VisionSmokeFixture,
    loader: Callable[[str], bytes],
) -> tuple[ImageInput, ...]:
    if control.image_unique_data_id is None or control.image_file is None:
        return ()
    donor = next(
        ex
        for ex in fixture.examples
        if ex.unique_data_id == control.image_unique_data_id
    )
    return (example_image_input(donor, loader),)


def run_variant_arm(
    port: JudgmentPort,
    *,
    controls: Sequence[VisualControl],
    fixture: VisionSmokeFixture,
    loader: Callable[[str], bytes],
    model_id: str,
    instruction: str,
    arm: str,
) -> tuple[list[dict[str, Any]], dict[str, SavedPromptOutcome]]:
    """Run one instruction arm across slice controls.

    Returns:
        Serialized rows and saved outcomes keyed by case id.
    """
    question = _noul_with_instruction(instruction)
    rows: list[dict[str, Any]] = []
    outcomes: dict[str, SavedPromptOutcome] = {}
    for control in controls:
        t0 = time.perf_counter()
        response = port.judge(
            control.state,
            {VISUAL_QUESTION_NAME: question},
            model_id,
            media=_media_for_control(control, fixture, loader),
        )
        prob_true = response.nouls[VISUAL_QUESTION_NAME].noul
        case_id = control.unique_data_id
        outcomes[case_id] = _saved_outcome(prob_true)
        rows.append(
            {
                "arm": arm,
                "instruction": instruction,
                "unique_data_id": case_id,
                "condition": control.condition,
                "answers": {
                    VISUAL_QUESTION_NAME: serialize_answer(
                        response.answers[VISUAL_QUESTION_NAME]
                    ),
                },
                "probabilities": dict(outcomes[case_id].probabilities),
                "latency_s": round(time.perf_counter() - t0, 3),
            }
        )
    return rows, outcomes


def probe_invalid_model(port: JudgmentPort) -> dict[str, Any]:
    """Exercise invalid model input and retain the validation failure.

    Returns:
        Probe payload with ``ok`` true when validation failed closed.
    """
    question = visual_question()
    try:
        port.judge(
            "invalid-state-probe",
            {VISUAL_QUESTION_NAME: question},
            "",
            media=None,
        )
    except JudgmentValidationError as exc:
        return {
            "ok": True,
            "exception": type(exc).__name__,
            "message": str(exc),
        }
    return {
        "ok": False,
        "exception": None,
        "message": "expected JudgmentValidationError for empty model",
    }


def build_offline_variant_port(
    controls: Sequence[VisualControl],
    ledger: VariantDispatchLedger,
) -> JudgmentPort:
    """Build a ledger-wrapped offline port for both instruction arms.

    Returns:
        Judgment port with scoring sequence for seed then candidate arms.
    """
    maps = _variant_logprob_maps(controls, arm="seed")
    maps.extend(_variant_logprob_maps(controls, arm="candidate"))
    fake = SequentialConsumerScoringFake(maps)
    scoring = _LedgerScoringPort(fake, ledger)
    adapter = ScoringJudgmentAdapter(
        scoring,
        tokenize_content=lambda text: (ord(text[0]),) if text else (),
        served_template=ServedTemplateClass.NATIVE_GEMMA4_TURN,
    )
    return _LedgerJudgmentPort(adapter, ledger)


def run_negative_probe(
    model_id: str,
    fixture_root: Path,
    loader: Callable[[str], bytes],
    fixture: VisionSmokeFixture,
) -> dict[str, Any]:
    """Run unsupported-template probe using fixture PNG bytes.

    Returns:
        Negative validation payload from ``run_negative_template_probe``.
    """
    sample_png = loader(fixture.examples[0].screenshot.file_name)
    return run_negative_template_probe(model_id, sample_png)
