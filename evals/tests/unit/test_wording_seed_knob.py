"""Unit checks for the ``TYPEVET_WORDING_SEED`` knob of the live test (#369).

The knob selects the DIFrauD ``Noul`` seed (default) or the PubMedQA
``Choice`` seed. These checks call the live test's pure helpers on
in-memory JSONL, so no call leaves the process.

Examples:
    ```bash
    uv run pytest -q evals/tests/unit/test_wording_seed_knob.py
    ```

See Also:
    - [typevet_evals.wording.rows][]: the PubMedQA splits
"""

from __future__ import annotations

import json
from collections import Counter

import pytest
from judgevet.domain.questions import Choice, Noul

from evals.tests.live import test_wording_evolution_live as live
from typevet_evals.wording.parts import question_mapping

pytestmark = pytest.mark.unit

PUBMEDQA = {"TYPEVET_WORDING_SEED": "pubmedqa"}


def _jsonl(per_label: int) -> str:
    """Return ``per_label`` PubMedQA rows of each label as JSONL."""
    rows = [
        {
            "pubid": index * 10 + offset,
            "question": f"Question {index} {label}?",
            "context": {"contexts": ["Some text."], "labels": ["RESULTS"]},
            "final_decision": label,
        }
        for offset, label in enumerate(("yes", "no", "maybe"))
        for index in range(per_label)
    ]
    return "".join(json.dumps(row) + "\n" for row in rows)


def test_the_default_seed_knob_keeps_the_difraud_noul() -> None:
    for environ in ({}, {"TYPEVET_WORDING_SEED": "difraud"}):
        seed = live.wording_seed(environ)
        assert isinstance(seed, Noul)
        assert seed.instructions == live.SEED_TEXT
        assert live.wording_question_name(environ) == live.PRIMARY_NOUL_NAME
    assert live.wording_split_sizes({}) == (1000, 200, 158)


def test_the_pubmedqa_seed_is_a_choice_with_one_criterion_per_label() -> None:
    seed = live.wording_seed(PUBMEDQA)

    assert isinstance(seed, Choice)
    assert list(seed.criteria) == ["yes", "no", "maybe"]
    assert set(question_mapping(seed)) == {
        "instructions",
        "criteria_yes",
        "criteria_no",
        "criteria_maybe",
    }
    assert live.wording_question_name(PUBMEDQA) == "answer"
    environ = PUBMEDQA | {"TYPEVET_WORDING_COMPONENTS": "criteria_maybe"}
    assert live.evolution_inputs(environ)[1] == ("criteria_maybe",)


@pytest.mark.parametrize(
    "environ",
    [
        {"TYPEVET_WORDING_SEED": "cord"},
        PUBMEDQA | {"TYPEVET_WORDING_SEED_CRITERIA": "1"},
        PUBMEDQA | {"TYPEVET_WORDING_COMPONENTS": "criteria_true"},
    ],
    ids=["unknown-seed", "noul-criteria", "noul-part"],
)
def test_the_seed_knob_refuses_an_unknown_or_mixed_value(
    environ: dict[str, str],
) -> None:
    with pytest.raises(ValueError, match="TYPEVET_WORDING"):
        live.evolution_inputs(environ)


def test_the_pubmedqa_splits_are_balanced_and_disjoint() -> None:
    environ = PUBMEDQA | {
        "TYPEVET_WORDING_TRAIN_ROWS": "6",
        "TYPEVET_WORDING_VALIDATION_ROWS": "3",
    }

    splits = live.pubmedqa_wording_splits(environ, _jsonl(8), held_out=6)

    for rows, split, size in (
        (splits.train, "train", 6),
        (splits.validation, "validation", 3),
        (splits.held_out, "test", 6),
    ):
        assert len(rows) == size
        assert {row.split for row in rows} == {split}
        assert set(Counter(row.gold for row in rows).values()) == {size // 3}
        assert all(isinstance(row.state, dict) for row in rows)
    ids = [r.record_id for r in (*splits.train, *splits.validation, *splits.held_out)]
    assert len(set(ids)) == len(ids)
    again = live.pubmedqa_wording_splits(environ, _jsonl(8), held_out=6)
    assert again == splits


def test_a_default_pubmedqa_run_takes_60_15_35_rows_per_label() -> None:
    assert live.wording_split_sizes(PUBMEDQA) == (180, 45, 105)
    environ = PUBMEDQA | {
        "TYPEVET_WORDING_TRAIN_ROWS": "9",
        "TYPEVET_WORDING_VALIDATION_ROWS": "6",
    }
    assert live.wording_split_sizes(environ) == (9, 6, 105)

    # 110 rows per label, as the PubMedQA card gives for ``maybe``.
    splits = live.pubmedqa_wording_splits(PUBMEDQA, _jsonl(110))

    for rows, size, per_label in (
        (splits.train, 180, 60),
        (splits.validation, 45, 15),
        (splits.held_out, 105, 35),
    ):
        assert len(rows) == size
        assert Counter(row.gold for row in rows) == dict.fromkeys(
            ("yes", "no", "maybe"), per_label
        )
    ids = [r.record_id for r in (*splits.train, *splits.validation, *splits.held_out)]
    assert len(set(ids)) == len(ids) == 330


def test_the_pubmedqa_splits_refuse_a_pool_too_small() -> None:
    with pytest.raises(ValueError, match="per label") as caught:
        live.pubmedqa_wording_splits(PUBMEDQA, _jsonl(8), held_out=158)

    assert "Question" not in str(caught.value)
    assert "Some text." not in str(caught.value)
