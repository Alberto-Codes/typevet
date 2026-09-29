"""Offline contract: PSAI vision smoke through ScoringJudgmentAdapter (#154).

The fake scorer replaces llama.cpp, so these tests prove wiring, prompt hygiene
and control bookkeeping. They say nothing about model quality.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from tests.fixtures.judgment_scoring_contract import SequentialScoringFake
from typevet.adapters.outbound.gemma import (
    CHATML_IM_START,
    GEMMA3_MODEL_TURN_HEADER,
    GEMMA3_START_OF_TURN,
    ServedTemplateClass,
)
from typevet.adapters.outbound.judgment_scoring import ScoringJudgmentAdapter
from typevet.domain.media import MEDIA_MARKER, count_media_markers
from typevet_evals.datasets.psai_vision import (
    FOX_FAMILY,
    GOLD_FIELDS,
    VISUAL_QUESTION_NAME,
    example_image_input,
    load_vision_smoke,
)
from typevet_evals.datasets.psai_vision_controls import (
    CONDITION_OMITTED,
    CONDITION_PRESENT,
    FAMILY_LEAK_TERMS,
    VISUAL_CONTROL_STATE,
    annotation_questions,
    annotation_state,
    control_matrix,
    semantic_hit,
    validate_no_family_leak,
    visual_question,
)

pytestmark = pytest.mark.contract

FIXTURE_DIR = (
    Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "psai" / "vision_smoke"
)
MODEL = "fake-vision-scoring"


def _read_image(file_name: str) -> bytes:
    return (FIXTURE_DIR / file_name).read_bytes()


def _tokenize(text: str) -> tuple[int, ...]:
    return (ord(text[0]),) if text else ()


@pytest.fixture
def fixture_set():
    text = (FIXTURE_DIR / "manifest.json").read_text(encoding="utf-8")
    return load_vision_smoke(text)


def _noul(probability: float) -> dict[str, float]:
    return {
        "True": math.log(probability),
        "False": math.log(1.0 - probability),
    }


def _adapter(logprobs: list[dict[str, float]]):
    fake = SequentialScoringFake(logprobs)
    adapter = ScoringJudgmentAdapter(
        fake,
        tokenize_content=_tokenize,
        served_template=ServedTemplateClass.NATIVE_GEMMA3_TURN,
    )
    return adapter, fake


def _judge_control(control, fixture_set, probability: float):
    """Score one control row with a scripted Noul probability."""
    adapter, fake = _adapter([_noul(probability)])
    media = ()
    if control.image_file is not None:
        donor = next(
            example
            for example in fixture_set.examples
            if example.unique_data_id == control.image_unique_data_id
        )
        media = (example_image_input(donor, _read_image),)
    response = adapter.judge(
        control.state,
        {VISUAL_QUESTION_NAME: visual_question()},
        MODEL,
        media=media,
    )
    return response, fake.calls[0]


def test_annotation_leg_returns_typed_answers_for_both_hub_fields(
    fixture_set,
) -> None:
    example = fixture_set.examples[0]
    adapter, fake = _adapter(
        [
            {"BROWSER_TASK": math.log(0.8), "COMPUTER_TASK": math.log(0.2)},
            _noul(0.2),
        ]
    )
    response = adapter.judge(annotation_state(example), annotation_questions(), MODEL)
    assert response.choices["category"].choice == "BROWSER_TASK"
    assert response.nouls["requires_login"].noul == pytest.approx(0.2)
    for call in fake.calls:
        assert call.media == ()
        assert count_media_markers(call.prefix) == 0


def test_annotation_prompt_shows_every_label_and_no_gold_marker(
    fixture_set,
) -> None:
    example = fixture_set.examples[0]
    adapter, fake = _adapter(
        [
            {"BROWSER_TASK": math.log(0.8), "COMPUTER_TASK": math.log(0.2)},
            _noul(0.2),
        ]
    )
    adapter.judge(annotation_state(example), annotation_questions(), MODEL)
    category_prefix = fake.calls[0].prefix
    # Both labels are offered, so the prompt cannot single out the gold one.
    assert "BROWSER_TASK" in category_prefix
    assert "COMPUTER_TASK" in category_prefix
    for prefix in (call.prefix for call in fake.calls):
        lowered = prefix.lower()
        assert "expected" not in lowered
        assert "ground_truth" not in lowered
        assert example.unique_data_id not in prefix


def test_visual_prompt_names_no_visual_family(fixture_set) -> None:
    for control in control_matrix(fixture_set.examples):
        _response, call = _judge_control(control, fixture_set, 0.5)
        # The question may name Fox News; the state must not say which row this is.
        assert VISUAL_CONTROL_STATE in call.prefix
        validate_no_family_leak(VISUAL_CONTROL_STATE)
        state_slice = call.prefix.split(VISUAL_CONTROL_STATE)[0].lower()
        for term in FAMILY_LEAK_TERMS:
            assert term not in state_slice


def test_present_and_swapped_attach_one_marked_image_and_omitted_attaches_none(
    fixture_set,
) -> None:
    for control in control_matrix(fixture_set.examples):
        _response, call = _judge_control(control, fixture_set, 0.5)
        if control.condition == CONDITION_OMITTED:
            assert call.media == ()
            assert count_media_markers(call.prefix) == 0
            continue
        assert control.image_file is not None
        assert len(call.media) == 1
        assert count_media_markers(call.prefix) == 1
        assert call.media[0].mime_type == "image/png"
        assert call.media[0].data == _read_image(control.image_file)


def test_present_and_omitted_share_the_native_wrapper_and_field_block(
    fixture_set,
) -> None:
    """Omitting the image drops only the marker and payload (#171)."""
    rows: dict[str, list] = {}
    for control in control_matrix(fixture_set.examples):
        _response, call = _judge_control(control, fixture_set, 0.5)
        rows.setdefault(control.unique_data_id, []).append((control.condition, call))
    for calls in rows.values():
        by_condition = dict(calls)
        present = by_condition[CONDITION_PRESENT]
        omitted = by_condition[CONDITION_OMITTED]
        for call in (present, omitted):
            assert call.prefix.startswith(f"{GEMMA3_START_OF_TURN}user\n")
            assert call.prefix.endswith(GEMMA3_MODEL_TURN_HEADER)
            assert CHATML_IM_START not in call.prefix
        assert present.prefix.replace(f"{MEDIA_MARKER}\n", "", 1) == omitted.prefix
        assert present.candidates == omitted.candidates
        assert omitted.media == ()


def test_a_perfectly_image_conditioned_scorer_hits_every_counted_control(
    fixture_set,
) -> None:
    """Green path: the scorer answers from the attached image."""
    counted = 0
    for control in control_matrix(fixture_set.examples):
        # A confident 0.9 on the omitted row must still earn nothing.
        probability = 0.9 if control.expected is None else 0.95
        if control.expected is False:
            probability = 0.05
        response, _call = _judge_control(control, fixture_set, probability)
        noul = response.nouls[VISUAL_QUESTION_NAME].noul
        hit = semantic_hit(control, noul)
        if control.counts_as_semantic_hit:
            assert hit is True
            counted += 1
        else:
            assert hit is False
    assert counted == 2 * len(fixture_set.examples)


def test_a_blind_scorer_fails_the_swap_control(fixture_set) -> None:
    """Red path: a scorer that ignores pixels cannot pass the matrix."""
    misses = []
    for control in control_matrix(fixture_set.examples):
        # A blind model answers "true" from its prior, whatever the image.
        response, _call = _judge_control(control, fixture_set, 0.9)
        noul = response.nouls[VISUAL_QUESTION_NAME].noul
        if control.counts_as_semantic_hit and not semantic_hit(control, noul):
            misses.append((control.unique_data_id, control.condition))
    # Every Fox swap and every non-Fox present row must miss.
    assert misses, "a blind scorer must not pass the control matrix"
    fox_ids = {example.unique_data_id for example in fixture_set.by_family(FOX_FAMILY)}
    assert any(uid in fox_ids and cond != CONDITION_PRESENT for uid, cond in misses)


def test_gold_fields_never_reach_a_scoring_prefix(fixture_set) -> None:
    example = fixture_set.examples[0]
    adapter, fake = _adapter(
        [
            {"BROWSER_TASK": math.log(0.8), "COMPUTER_TASK": math.log(0.2)},
            _noul(0.2),
        ]
    )
    adapter.judge(annotation_state(example), annotation_questions(), MODEL)
    serialized = "\n".join(call.prefix for call in fake.calls)
    for field in GOLD_FIELDS:
        assert f"{field}=" not in serialized
        assert f'"{field}":' not in serialized
