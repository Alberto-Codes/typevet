"""Question map, leakage bans and control matrix for the PSAI vision smoke (#154)."""

from __future__ import annotations

from pathlib import Path

import pytest

from typevet.domain.judgment_questions import Choice, Noul
from typevet.evaluation.datasets.psai_schema import BANNED_STATE_KEYS
from typevet.evaluation.datasets.psai_vision import (
    ANNOTATION_PROVENANCE,
    FOX_FAMILY,
    GOLD_FIELDS,
    MANUAL_VISUAL_PROVENANCE,
    NON_FOX_FAMILY,
    VISUAL_QUESTION_NAME,
    load_vision_smoke,
)
from typevet.evaluation.datasets.psai_vision_controls import (
    CONDITION_OMITTED,
    CONDITION_PRESENT,
    CONDITION_SWAPPED,
    CONDITIONS,
    FAMILY_LEAK_TERMS,
    NOUL_THRESHOLD,
    PAIRED_MARGIN_FLOOR,
    VISUAL_CONTROL_STATE,
    annotation_questions,
    annotation_state,
    control_matrix,
    control_summary,
    noul_polarity,
    paired_image_ordering,
    question_provenance,
    semantic_hit,
    validate_no_family_leak,
    vision_questions,
    visual_question,
)

pytestmark = pytest.mark.unit

FIXTURE_DIR = Path(__file__).resolve().parents[1] / "fixtures" / "psai" / "vision_smoke"


@pytest.fixture
def fixture_set():
    text = (FIXTURE_DIR / "manifest.json").read_text(encoding="utf-8")
    return load_vision_smoke(text)


def test_annotation_state_is_task_name_only(fixture_set) -> None:
    for example in fixture_set.examples:
        state = annotation_state(example)
        assert set(state) == {"task_name"}
        assert state["task_name"] == example.task_name
        assert not BANNED_STATE_KEYS.intersection(state)


def test_annotation_state_never_carries_a_gold_value(fixture_set) -> None:
    for example in fixture_set.examples:
        text = annotation_state(example)["task_name"]
        assert "BROWSER_TASK" not in text
        assert VISUAL_QUESTION_NAME not in text


def test_visual_control_state_names_neither_visual_family() -> None:
    validate_no_family_leak(VISUAL_CONTROL_STATE)
    lowered = VISUAL_CONTROL_STATE.lower()
    for term in FAMILY_LEAK_TERMS:
        assert term not in lowered


def test_validate_no_family_leak_rejects_a_branded_state() -> None:
    with pytest.raises(ValueError, match="remove: fox"):
        validate_no_family_leak("Only use http://foxnews.com to achieve this task.")


def test_every_task_name_names_its_site_so_the_visual_leg_cannot_use_it(
    fixture_set,
) -> None:
    """The recorded reason the visual controls drop ``task_name``.

    Each PSAI ``task_name`` pins the site with ``Only use http://<site>``, which
    hands the model the ``shows_fox_news_chrome`` answer as text. A text-derivable
    answer would void the image-conditioning controls, so the visual leg swaps in
    a family-neutral state.
    """
    for example in fixture_set.examples:
        with pytest.raises(ValueError, match="leak"):
            validate_no_family_leak(example.task_name)


def test_question_map_covers_the_three_gold_fields() -> None:
    questions = vision_questions()
    assert set(questions) == set(GOLD_FIELDS)
    assert isinstance(questions["category"], Choice)
    assert isinstance(questions["requires_login"], Noul)
    assert isinstance(questions[VISUAL_QUESTION_NAME], Noul)
    assert set(annotation_questions()) == {"category", "requires_login"}
    assert VISUAL_QUESTION_NAME not in annotation_questions()


def test_question_provenance_separates_annotation_from_manual_visual() -> None:
    provenance = question_provenance()
    assert provenance["category"] == ANNOTATION_PROVENANCE
    assert provenance["requires_login"] == ANNOTATION_PROVENANCE
    assert provenance[VISUAL_QUESTION_NAME] == MANUAL_VISUAL_PROVENANCE


def test_visual_question_states_both_polarities_without_naming_a_row() -> None:
    question = visual_question()
    assert question.criteria is not None
    assert set(question.criteria) == {"true", "false"}


def test_control_matrix_runs_three_conditions_for_every_row(fixture_set) -> None:
    controls = control_matrix(fixture_set.examples)
    assert len(controls) == len(fixture_set.examples) * len(CONDITIONS)
    for example in fixture_set.examples:
        rows = [c for c in controls if c.unique_data_id == example.unique_data_id]
        assert [row.condition for row in rows] == list(CONDITIONS)


def test_present_control_uses_the_row_screenshot_and_its_own_gold(
    fixture_set,
) -> None:
    by_id = {e.unique_data_id: e for e in fixture_set.examples}
    for control in control_matrix(fixture_set.examples):
        if control.condition != CONDITION_PRESENT:
            continue
        example = by_id[control.unique_data_id]
        assert control.image_file == example.screenshot.file_name
        assert control.image_unique_data_id == example.unique_data_id
        assert control.expected is example.shows_fox_news_chrome
        assert control.counts_as_semantic_hit is True


def test_omitted_control_drops_the_image_and_never_counts_as_a_hit(
    fixture_set,
) -> None:
    for control in control_matrix(fixture_set.examples):
        if control.condition != CONDITION_OMITTED:
            continue
        assert control.image_file is None
        assert control.image_unique_data_id is None
        assert control.expected is None
        assert control.counts_as_semantic_hit is False


def test_swapped_control_borrows_the_opposite_family_and_flips_the_expectation(
    fixture_set,
) -> None:
    by_id = {e.unique_data_id: e for e in fixture_set.examples}
    swapped = [
        control
        for control in control_matrix(fixture_set.examples)
        if control.condition == CONDITION_SWAPPED
    ]
    assert len(swapped) == len(fixture_set.examples)
    for control in swapped:
        own = by_id[control.unique_data_id]
        assert control.image_unique_data_id is not None
        donor = by_id[control.image_unique_data_id]
        assert donor.visual_family != own.visual_family
        assert control.image_file == donor.screenshot.file_name
        assert control.expected is donor.shows_fox_news_chrome
        assert control.expected is not own.shows_fox_news_chrome
        assert control.counts_as_semantic_hit is True


def test_every_control_carries_the_family_neutral_state(fixture_set) -> None:
    for control in control_matrix(fixture_set.examples):
        assert control.state == VISUAL_CONTROL_STATE
        validate_no_family_leak(control.state)


def test_control_matrix_needs_both_families(fixture_set) -> None:
    fox_only = [e for e in fixture_set.examples if e.visual_family == FOX_FAMILY]
    with pytest.raises(ValueError, match=NON_FOX_FAMILY):
        control_matrix(fox_only)


@pytest.mark.parametrize(
    ("probability", "polarity"),
    [(0.0, False), (0.49, False), (0.5, True), (0.99, True)],
)
def test_noul_polarity_splits_on_the_threshold(
    probability: float, polarity: bool
) -> None:
    assert NOUL_THRESHOLD == 0.5
    assert noul_polarity(probability) is polarity


def test_semantic_hit_scores_present_and_swapped_but_never_omitted(
    fixture_set,
) -> None:
    first = fixture_set.by_family(FOX_FAMILY)[0]
    controls = {
        control.condition: control
        for control in control_matrix(fixture_set.examples)
        if control.unique_data_id == first.unique_data_id
    }
    present = controls[CONDITION_PRESENT]
    assert present.expected is True
    assert semantic_hit(present, 0.9) is True
    assert semantic_hit(present, 0.1) is False

    swapped = controls[CONDITION_SWAPPED]
    assert swapped.expected is False
    assert semantic_hit(swapped, 0.1) is True
    assert semantic_hit(swapped, 0.9) is False

    omitted = controls[CONDITION_OMITTED]
    assert semantic_hit(omitted, 0.9) is False
    assert semantic_hit(omitted, 0.1) is False


def test_control_summary_counts_only_the_counted_conditions(fixture_set) -> None:
    controls = control_matrix(fixture_set.examples)
    perfect = {
        (control.unique_data_id, control.condition): (
            0.95 if control.expected else 0.05
        )
        for control in controls
    }
    summary = control_summary(controls, perfect)
    rows = len(fixture_set.examples)
    assert summary == {"counted": 2 * rows, "hits": 2 * rows, "uncounted": rows}


def test_control_summary_credits_no_hit_to_a_blind_scorer(fixture_set) -> None:
    controls = control_matrix(fixture_set.examples)
    blind = {(c.unique_data_id, c.condition): 0.9 for c in controls}
    summary = control_summary(controls, blind)
    assert summary["counted"] == 2 * len(fixture_set.examples)
    assert summary["hits"] < summary["counted"]


def test_control_summary_needs_a_probability_for_every_control(fixture_set) -> None:
    controls = control_matrix(fixture_set.examples)
    with pytest.raises(KeyError):
        control_summary(controls, {})


def _probabilities(controls, fox: float, non_fox: float):
    """Score every imaged control by the family of the image it carries."""
    return {
        (control.unique_data_id, control.condition): (
            fox if control.expected else non_fox
        )
        for control in controls
    }


def test_paired_ordering_pairs_each_row_across_the_two_families(
    fixture_set,
) -> None:
    controls = control_matrix(fixture_set.examples)
    pairs = paired_image_ordering(controls, _probabilities(controls, 0.92, 0.09))
    assert len(pairs) == len(fixture_set.examples)
    families = {e.unique_data_id: e.visual_family for e in fixture_set.examples}
    for pair in pairs:
        assert families[pair.fox_image_id] == FOX_FAMILY
        assert families[pair.non_fox_image_id] == NON_FOX_FAMILY
        assert pair.margin == pytest.approx(0.92 - 0.09)
        assert pair.ordered is True


def test_paired_ordering_fails_when_the_pixels_do_not_move_the_answer(
    fixture_set,
) -> None:
    """A blind scorer gives both images the same probability, so margin is zero."""
    controls = control_matrix(fixture_set.examples)
    pairs = paired_image_ordering(controls, _probabilities(controls, 0.9, 0.9))
    assert all(pair.margin == pytest.approx(0.0) for pair in pairs)
    assert not any(pair.ordered for pair in pairs)


def test_paired_ordering_fails_when_the_families_are_inverted(fixture_set) -> None:
    controls = control_matrix(fixture_set.examples)
    pairs = paired_image_ordering(controls, _probabilities(controls, 0.09, 0.92))
    assert all(pair.margin < 0 for pair in pairs)
    assert not any(pair.ordered for pair in pairs)


def test_paired_ordering_honours_the_margin_floor(fixture_set) -> None:
    controls = control_matrix(fixture_set.examples)
    probabilities = _probabilities(controls, 0.55, 0.45)
    assert PAIRED_MARGIN_FLOOR == 0.10
    # 0.55 - 0.45 lands exactly on the default floor, so the default accepts it.
    assert all(pair.ordered for pair in paired_image_ordering(controls, probabilities))
    assert not any(
        pair.ordered for pair in paired_image_ordering(controls, probabilities, 0.50)
    )


def test_paired_ordering_needs_both_families_in_the_row(fixture_set) -> None:
    controls = [
        control
        for control in control_matrix(fixture_set.examples)
        if control.condition != CONDITION_SWAPPED
    ]
    with pytest.raises(ValueError, match="one Fox and one non-Fox"):
        paired_image_ordering(controls, _probabilities(controls, 0.9, 0.1))


def test_paired_ordering_needs_a_probability_for_every_imaged_control(
    fixture_set,
) -> None:
    controls = control_matrix(fixture_set.examples)
    with pytest.raises(KeyError):
        paired_image_ordering(controls, {})
