"""Unit checks for the Choice and Score wording part names (#369 slice 1).

Examples:
    ```bash
    uv run pytest -q evals/tests/unit/test_wording_choice_parts.py
    ```

See Also:
    - [typevet_evals.wording.parts][]: the part names of each question type
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from typing import Any

import pytest
from judgevet.domain.questions import Choice, Noul, Score

from typevet_evals.datasets.difraud import DIFrauDRecord, map_row, record_id
from typevet_evals.wording import WordingRunConfig, WordingTransport
from typevet_evals.wording.held_out import HeldOutRows, score_held_out
from typevet_evals.wording.parts import (
    WordingParts,
    part_roles,
    part_table,
    question_from_parts,
    question_mapping,
)
from typevet_evals.wording.runner import evolve_wording, reflection_prompt

pytestmark = pytest.mark.unit

SENTINEL = "SENTINEL-WORDING-TEXT"
ASK = "Which expense type is it?"
NOUL = Noul(instructions=ASK, criteria={"true": "Yes", "false": "No"})
CHOICE = Choice(
    instructions=ASK,
    criteria={"food": "A meal", "travel": "A trip", "other": "Anything else"},
)
SPACED = Choice(
    instructions=ASK,
    criteria={"food": "A meal", "air travel": "A flight", "other": None},
)
SCORE = Score(instructions="Rate it.", criteria=["Poor", "Fair", "Good"])
NO_PORT: Any = None


def _record(text: str, split: str) -> DIFrauDRecord:
    """Return one record whose example names ``split``."""
    return DIFrauDRecord(record_id(text), replace(map_row(text, 1), split=split))


def test_identifier_labels_map_to_criteria_names() -> None:
    assert question_mapping(CHOICE) == {
        "instructions": ASK,
        "criteria_food": "A meal",
        "criteria_travel": "A trip",
        "criteria_other": "Anything else",
    }
    assert part_table(CHOICE) == {
        "criteria_food": "food",
        "criteria_travel": "travel",
        "criteria_other": "other",
    }


def test_one_non_identifier_label_makes_every_option_positional() -> None:
    assert question_mapping(SPACED) == {
        "instructions": ASK,
        "option_0": "A meal",
        "option_1": "A flight",
    }
    assert part_table(SPACED) == {"option_0": "food", "option_1": "air travel"}


def test_a_label_that_is_not_text_makes_every_option_positional() -> None:
    criteria: dict[Any, Any] = {"food": "A meal", 7: "Seven"}
    seed = Choice(instructions=ASK, criteria=criteria)

    assert question_mapping(seed) == {
        "instructions": ASK,
        "option_0": "A meal",
        "option_1": "Seven",
    }
    assert part_table(seed) == {"option_0": "food", "option_1": 7}
    assert question_from_parts(seed, question_mapping(seed)).criteria == criteria


def test_a_score_level_without_a_description_has_no_part() -> None:
    levels: list[Any] = ["Poor", None, "Good"]
    seed = Score(instructions="Rate it.", criteria=levels)

    assert question_mapping(seed) == {
        "instructions": "Rate it.",
        "level_0": "Poor",
        "level_2": "Good",
    }
    assert part_table(seed) == {"level_0": 0, "level_2": 2}
    mapping = question_mapping(seed) | {"level_2": "Great"}
    assert question_from_parts(seed, mapping).criteria == ["Poor", None, "Great"]


def test_a_score_maps_each_described_level_by_index() -> None:
    assert question_mapping(SCORE) == {
        "instructions": "Rate it.",
        "level_0": "Poor",
        "level_1": "Fair",
        "level_2": "Good",
    }
    assert part_table(SCORE) == {"level_0": 0, "level_1": 1, "level_2": 2}


def test_a_noul_keeps_its_names_and_table() -> None:
    assert question_mapping(NOUL) == {
        "instructions": ASK,
        "criteria_true": "Yes",
        "criteria_false": "No",
    }
    assert part_table(NOUL) == {"criteria_true": "true", "criteria_false": "false"}
    choice = Choice(instructions=ASK, criteria={"true": "Yes", "false": "No"})
    assert part_table(choice) == part_table(NOUL)


@pytest.mark.parametrize(
    ("seed", "names"),
    [
        (Choice(instructions=ASK, criteria={"food": {"x": SENTINEL}}), "'food'"),
        (Choice(instructions=ASK, criteria={"a b": [SENTINEL]}), "'a b'"),
        (Score(instructions=ASK, criteria=["Poor", {"x": SENTINEL}]), "level 1"),
    ],
    ids=["choice-dict", "choice-list", "score-dict"],
)
def test_a_structured_description_refuses_the_seed_without_its_text(
    seed: Any, names: str
) -> None:
    with pytest.raises(TypeError, match=names) as caught:
        question_mapping(seed)
    assert SENTINEL not in str(caught.value)


@dataclass(frozen=True)
class Other:
    """A seed of a type the wording parts do not know.

    Attributes:
        instructions (object): The wording.
        criteria (object): The criteria.
    """

    instructions: object = ASK
    criteria: object = None


def test_an_unknown_seed_or_non_text_instructions_are_refused() -> None:
    with pytest.raises(ValueError, match="Other"):
        question_mapping(Other())
    with pytest.raises(TypeError, match="instructions") as caught:
        question_mapping(Choice(instructions={"x": SENTINEL}, criteria={"a": "A"}))
    assert SENTINEL not in str(caught.value)


def test_a_none_description_has_no_part() -> None:
    seed = Choice(instructions=ASK, criteria={"food": None, "other": "Else"})

    assert question_mapping(seed) == {"instructions": ASK, "criteria_other": "Else"}
    assert part_table(seed) == {"criteria_other": "other"}


@pytest.mark.parametrize(
    "seed",
    [NOUL, Noul(instructions=ASK), CHOICE, SPACED, SCORE],
    ids=["noul", "bare-noul", "choice", "spaced-choice", "score"],
)
def test_the_seed_mapping_renders_back_to_the_seed(seed: Any) -> None:
    question = question_from_parts(seed, question_mapping(seed))

    assert type(question) is type(seed)
    assert question.instructions == seed.instructions
    assert question.criteria == seed.criteria
    if isinstance(seed, Choice):
        assert list(question.criteria) == list(seed.criteria)


def test_an_evolved_choice_part_renders_in_label_order() -> None:
    mapping = question_mapping(SPACED) | {"option_1": "A plane ticket"}

    question = question_from_parts(SPACED, mapping)

    assert question.criteria == {
        "food": "A meal",
        "air travel": "A plane ticket",
        "other": None,
    }
    assert list(question.criteria) == ["food", "air travel", "other"]


def test_part_roles_name_each_part_of_the_seed() -> None:
    for seed in (NOUL, CHOICE, SPACED, SCORE):
        roles = part_roles(seed)
        assert set(roles) == set(question_mapping(seed))
        assert all(SENTINEL not in role for role in roles.values())
    assert "'air travel'" in part_roles(SPACED)["option_1"]
    assert "level 2" in part_roles(SCORE)["level_2"]


def test_the_reflection_prompt_takes_the_seed_roles() -> None:
    prompt = reflection_prompt(
        question_mapping(SPACED), ("option_1",), roles=part_roles(SPACED)
    )

    assert "- option_1 (what the option 'air travel' means): at most 12" in prompt
    assert "A flight" not in prompt


@pytest.mark.parametrize("seed", [SCORE], ids=["score"])
def test_a_run_still_refuses_a_score_seed(seed: Any) -> None:
    name = type(seed).__name__
    mapping = question_mapping(seed)

    with pytest.raises(ValueError, match=name):
        WordingTransport(
            port=NO_PORT, mapping=mapping, question_name="q", seed=seed, judge_model="m"
        )
    parts = WordingParts(("instructions",), mapping, mapping)
    with pytest.raises(ValueError, match=name):
        score_held_out(
            NO_PORT,
            seed,
            "q",
            evolved=parts,
            rows=HeldOutRows((_record("Hi", "test"),), frozenset()),
            judge_model="m",
            failures=(RuntimeError,),
        )
    config = WordingRunConfig(reflector="r", judge_model="m")
    with pytest.raises(ValueError, match=name):
        asyncio.run(
            evolve_wording(
                port=NO_PORT,
                seed=seed,
                question_name="q",
                train=(_record("Win", "train"),),
                validation=(_record("Lunch", "validation"),),
                config=config,
            )
        )
