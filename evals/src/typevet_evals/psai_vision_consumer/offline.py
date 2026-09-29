"""Offline matrix execution for the PSAI consumer harness ([#177][i177]).

Examples:
    ```python
    from pathlib import Path

    from typevet_evals.psai_vision_consumer.offline import (
        build_offline_consumer_port,
        run_offline_consumer_matrix,
    )

    port, controls = build_offline_consumer_port(
        Path("tests/fixtures/psai/vision_smoke")
    )
    rows, probs, negative = run_offline_consumer_matrix(
        port, fixture_root=Path("tests/fixtures/psai/vision_smoke")
    )
    assert negative["ok"]
    ```

See Also:
    - [typevet_evals.psai_vision_consumer.harness][]: receipt orchestration

[i177]: https://github.com/Alberto-Codes/typevet/issues/177
"""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from typevet.adapters.outbound.gemma import ServedTemplateClass
from typevet.adapters.outbound.judgment_scoring import ScoringJudgmentAdapter
from typevet.domain import ImageInput, JudgmentValidationError
from typevet.domain.candidate_scoring_request import CandidateScoringRequest
from typevet.domain.candidate_scoring_response import CandidateScoringResult
from typevet.domain.judgment_response import JudgmentResponse
from typevet.evaluation.datasets.psai_vision import (
    VisionSmokeFixture,
    example_image_input,
    load_vision_smoke,
    vision_smoke_manifest_path,
)
from typevet.evaluation.datasets.psai_vision_controls import (
    VISUAL_QUESTION_NAME,
    VisualControl,
    annotation_questions,
    annotation_state,
    control_matrix,
    visual_question,
)
from typevet.ports.judgment import JudgmentPort
from typevet.testing import ScriptedScoringFake
from typevet_evals.psai_vision_consumer.accounting import (
    FROZEN_CONSUMER_CASE_UIDS,
    TEXT_ANNOTATION_JUDGE_UIDS,
)
from typevet_evals.psai_vision_consumer.receipt import serialize_answer


def load_frozen_consumer_fixture(fixture_root: Path) -> VisionSmokeFixture:
    """Load the committed vision smoke manifest from ``fixture_root``.

    Args:
        fixture_root: Directory containing ``manifest.json``.

    Returns:
        Parsed fixture set.

    Raises:
        FileNotFoundError: When the manifest is missing.
    """
    manifest_path = vision_smoke_manifest_path(fixture_root)
    if not manifest_path.is_file():
        msg = f"missing vision smoke manifest: {manifest_path}"
        raise FileNotFoundError(msg)
    return load_vision_smoke(manifest_path.read_text(encoding="utf-8"))


def frozen_consumer_controls(fixture: VisionSmokeFixture) -> tuple[VisualControl, ...]:
    """Return present/omit/swap controls for frozen protocol rows only.

    Args:
        fixture: Loaded vision smoke fixture.

    Returns:
        Control rows for protocol rev 1 pins.

    Raises:
        ValueError: When a frozen uid is absent from the manifest.
    """
    controls = [
        c
        for c in control_matrix(fixture.examples)
        if c.unique_data_id in FROZEN_CONSUMER_CASE_UIDS
    ]
    if len({c.unique_data_id for c in controls}) != len(FROZEN_CONSUMER_CASE_UIDS):
        msg = "manifest missing one or more frozen consumer case rows"
        raise ValueError(msg)
    return tuple(controls)


def _read_image(fixture_root: Path, file_name: str) -> bytes:
    return (fixture_root / file_name).read_bytes()


def _media_for_control(
    control: VisualControl,
    fixture: VisionSmokeFixture,
    read_image: Callable[[str], bytes],
) -> tuple[ImageInput, ...]:
    if control.image_unique_data_id is None or control.image_file is None:
        return ()
    donor = next(
        ex
        for ex in fixture.examples
        if ex.unique_data_id == control.image_unique_data_id
    )
    return (example_image_input(donor, read_image),)


def _matrix_row(
    response: JudgmentResponse,
    question_names: Sequence[str],
    extra: Mapping[str, Any],
) -> dict[str, Any]:
    answers = {
        name: serialize_answer(response.answers[name]) for name in question_names
    }
    usage = response.usage
    row: dict[str, Any] = {
        "answers": answers,
        "usage": {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
        },
    }
    row.update(extra)
    return row


def run_negative_template_probe(
    model_id: str,
    png_bytes: bytes,
) -> dict[str, Any]:
    """Run unsupported-template negative validation without live HTTP.

    Args:
        model_id: Model string forwarded to ``judge``.
        png_bytes: PNG bytes attached to the probe call.

    Returns:
        Dict with ``ok``, ``exception``, and ``message`` keys.
    """
    bad_port = ScoringJudgmentAdapter(
        ScriptedScoringFake(logprobs={"True": -0.2, "False": -1.0}),
        tokenize_content=lambda text: (ord(text[0]),) if text else (),
        served_template=None,
    )
    media = (ImageInput(data=png_bytes, mime_type="image/png"),)
    try:
        bad_port.judge(
            "probe",
            {VISUAL_QUESTION_NAME: visual_question()},
            model_id,
            media=media,
        )
    except JudgmentValidationError as exc:
        return {
            "ok": True,
            "exception": type(exc).__name__,
            "message": str(exc),
        }
    else:
        return {
            "ok": False,
            "exception": None,
            "message": "expected JudgmentValidationError",
        }


def _run_visual_leg(
    port: JudgmentPort,
    *,
    controls: Sequence[VisualControl],
    fixture: VisionSmokeFixture,
    loader: Callable[[str], bytes],
    model_id: str,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], float]]:
    matrix_rows: list[dict[str, Any]] = []
    probabilities: dict[tuple[str, str], float] = {}
    for control in controls:
        t0 = time.perf_counter()
        response = port.judge(
            control.state,
            {VISUAL_QUESTION_NAME: visual_question()},
            model_id,
            media=_media_for_control(control, fixture, loader),
        )
        prob = response.nouls[VISUAL_QUESTION_NAME].noul
        probabilities[(control.unique_data_id, control.condition)] = prob
        matrix_rows.append(
            _matrix_row(
                response,
                (VISUAL_QUESTION_NAME,),
                {
                    "latency_s": round(time.perf_counter() - t0, 3),
                    "unique_data_id": control.unique_data_id,
                    "condition": control.condition,
                    "leg": "visual",
                },
            )
        )
    return matrix_rows, probabilities


def _run_annotation_leg(
    port: JudgmentPort,
    *,
    fixture: VisionSmokeFixture,
    model_id: str,
) -> list[dict[str, Any]]:
    questions = annotation_questions()
    by_uid = {ex.unique_data_id: ex for ex in fixture.examples}
    rows: list[dict[str, Any]] = []
    for uid in TEXT_ANNOTATION_JUDGE_UIDS:
        ex = by_uid[uid]
        t0 = time.perf_counter()
        response = port.judge(
            annotation_state(ex),
            questions,
            model_id,
            media=None,
        )
        rows.append(
            _matrix_row(
                response,
                tuple(questions.keys()),
                {
                    "latency_s": round(time.perf_counter() - t0, 3),
                    "unique_data_id": uid,
                    "condition": "text_annotation",
                    "leg": "annotation",
                },
            )
        )
    return rows


def run_offline_consumer_matrix(
    port: JudgmentPort,
    *,
    fixture_root: Path,
    model_id: str = "offline-consumer-fake",
    read_image: Callable[[str], bytes] | None = None,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], float], dict[str, Any]]:
    """Execute the frozen matrix through ``port`` without live HTTP.

    Args:
        port: Judgment adapter under test.
        fixture_root: ``tests/fixtures/psai/vision_smoke`` root.
        model_id: Model string forwarded to ``judge``.
        read_image: Optional image loader.

    Returns:
        ``(matrix_rows, visual_probabilities, negative_validation)``.
    """
    fixture = load_frozen_consumer_fixture(fixture_root)
    loader = read_image or (lambda name: _read_image(fixture_root, name))
    controls = frozen_consumer_controls(fixture)
    visual_rows, probabilities = _run_visual_leg(
        port,
        controls=controls,
        fixture=fixture,
        loader=loader,
        model_id=model_id,
    )
    matrix_rows = visual_rows + _run_annotation_leg(
        port, fixture=fixture, model_id=model_id
    )
    sample_png = loader(fixture.examples[0].screenshot.file_name)
    negative = run_negative_template_probe(model_id, sample_png)
    return matrix_rows, probabilities, negative


def _visual_logprobs(control: VisualControl) -> dict[str, float]:
    """Return logprobs that preserve fox-vs-non-fox ordering offline.

    Args:
        control: One visual control row.

    Returns:
        ``True``/``False`` logprob map for ``ScriptedScoringFake``.
    """
    if control.condition == "omitted":
        p_true = 0.27
    elif control.expected is True:
        p_true = 0.99
    else:
        p_true = 0.01
    return {"True": math.log(p_true), "False": math.log(1.0 - p_true)}


def _annotation_logprob_sequence() -> list[dict[str, float]]:
    choice = {"BROWSER_TASK": math.log(0.7), "COMPUTER_TASK": math.log(0.3)}
    noul = {"True": math.log(0.2), "False": math.log(0.8)}
    return [choice, noul, choice, noul]


class SequentialConsumerScoringFake(ScriptedScoringFake):
    """``ScriptedScoringFake`` with one logprob map per matrix scoring request.

    Attributes:
        calls (list): Inherited request log from ``ScriptedScoringFake``.

    Examples:
        ```python
        from typevet_evals.psai_vision_consumer.offline import (
            SequentialConsumerScoringFake,
        )

        SequentialConsumerScoringFake([{"True": -0.2, "False": -1.0}])
        ```
    """

    def __init__(self, maps: Sequence[Mapping[str, float]]) -> None:
        """Store one logprob map per upcoming ``score_candidates`` call.

        Args:
            maps: Ordered logprob dicts returned in sequence.
        """
        super().__init__()
        self._maps = list(maps)
        self._index = 0

    def score_candidates(
        self, request: CandidateScoringRequest
    ) -> CandidateScoringResult:
        """Return the next scripted logprob map.

        Args:
            request: Incoming scoring request (recorded on ``self.calls``).

        Returns:
            Validation-wrapped scoring result from the scripted map.

        Raises:
            ValueError: When the scripted sequence is exhausted.
        """
        if self._index >= len(self._maps):
            msg = "sequential consumer fake exhausted scripted maps"
            raise ValueError(msg)
        self._scripted = dict(self._maps[self._index])
        self._index += 1
        return super().score_candidates(request)


def consumer_fixture_identity_pins(fixture_root: Path) -> dict[str, Any]:
    """Return frozen manifest and image digests for consumer receipts ([#177][i177]).

    Args:
        fixture_root: Committed ``vision_smoke`` directory.

    Returns:
        JSON-serializable pin block; unknown fields stay absent upstream.
    """
    fixture = load_frozen_consumer_fixture(fixture_root)
    manifest_path = vision_smoke_manifest_path(fixture_root)
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    image_pins: list[dict[str, str]] = []
    for example in fixture.examples:
        if example.unique_data_id not in FROZEN_CONSUMER_CASE_UIDS:
            continue
        image_pins.append(
            {
                "unique_data_id": example.unique_data_id,
                "file_name": example.screenshot.file_name,
                "sha256": example.screenshot.sha256,
            }
        )
    image_pins.sort(key=lambda row: row["unique_data_id"])
    return {
        "manifest_sha256": manifest_sha256,
        "frozen_case_image_digests": image_pins,
        "harness_entrypoint": "scripts/run_psai_vision_consumer_proof.py",
    }


def build_offline_consumer_port(
    fixture_root: Path,
) -> tuple[
    ScoringJudgmentAdapter,
    tuple[VisualControl, ...],
    SequentialConsumerScoringFake,
]:
    """Build an offline adapter and frozen controls for the consumer matrix.

    Args:
        fixture_root: Committed ``vision_smoke`` directory.

    Returns:
        Adapter, frozen control rows, and the sequential scoring fake.
    """
    fixture = load_frozen_consumer_fixture(fixture_root)
    controls = frozen_consumer_controls(fixture)
    maps = [_visual_logprobs(c) for c in controls]
    maps.extend(_annotation_logprob_sequence())
    fake = SequentialConsumerScoringFake(maps)
    port = ScoringJudgmentAdapter(
        fake,
        tokenize_content=lambda text: (ord(text[0]),) if text else (),
        served_template=ServedTemplateClass.NATIVE_GEMMA4_TURN,
    )
    return port, controls, fake
