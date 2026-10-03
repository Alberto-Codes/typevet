"""Unit tests for duel scoring, binning, parsing and the duel receipt (#412)."""

from __future__ import annotations

import json

import pytest

from typevet_evals.doodle_duel import (
    BASELINE_BRIER,
    CONFIDENCE_MAX,
    CONFIDENCE_MIN,
    DuelAnswer,
    build_duel_receipt,
    multiclass_brier,
    parse_answers,
    parse_confidence,
    parse_label,
    pick_brier,
    reliability_rows,
    score_round,
)

pytestmark = pytest.mark.unit

CATEGORIES = ("cat", "moon", "sun")
BIN_NAMES = ["0-50", "50-60", "60-70", "70-80", "80-90", "90-100"]
ROW_KEYS = {
    "round",
    "key_id",
    "true_label",
    "player_label",
    "player_confidence",
    "player_correct",
    "player_brier",
    "model_label",
    "model_confidence",
    "model_correct",
    "model_brier",
}


def _source_row(key_id: str, true_label: str, chosen: str, p: float) -> dict:
    rest = (1.0 - p) / (len(CATEGORIES) - 1)
    probabilities = {name: (p if name == chosen else rest) for name in CATEGORIES}
    return {
        "key_id": key_id,
        "true_label": true_label,
        "chosen_label": chosen,
        "probabilities": probabilities,
        "correct": chosen == true_label,
        "latency_seconds": 0.1,
    }


def _bins(rows: list[dict]) -> dict[str, dict]:
    return {row["bin"]: row for row in rows}


def test_pick_brier_squares_the_miss_for_right_and_wrong_picks() -> None:
    assert pick_brier(0.8, correct=True) == pytest.approx(0.04)
    assert pick_brier(0.8, correct=False) == pytest.approx(0.64)
    assert pick_brier(0.5, correct=True) == pytest.approx(0.25)
    assert pick_brier(0.5, correct=False) == pytest.approx(0.25)
    assert pick_brier(0.0, correct=False) == 0.0
    assert pick_brier(1.0, correct=False) == 1.0


def test_baseline_is_always_fifty_percent_brier() -> None:
    assert BASELINE_BRIER == 0.25
    assert pick_brier(0.5, correct=True) == BASELINE_BRIER
    assert pick_brier(0.5, correct=False) == BASELINE_BRIER
    assert (CONFIDENCE_MIN, CONFIDENCE_MAX) == (50, 100)


def test_multiclass_brier_sums_over_every_category() -> None:
    probabilities = {"cat": 0.7, "moon": 0.2, "sun": 0.1}
    expected = (0.7 - 1) ** 2 + 0.2**2 + 0.1**2
    assert multiclass_brier(probabilities, "cat", CATEGORIES) == pytest.approx(expected)
    wrong = 0.7**2 + (0.2 - 1) ** 2 + 0.1**2
    assert multiclass_brier(probabilities, "moon", CATEGORIES) == pytest.approx(wrong)
    with pytest.raises(ValueError, match="dog"):
        multiclass_brier(probabilities, "dog", CATEGORIES)


def test_reliability_rows_hold_count_confidence_accuracy_and_gap() -> None:
    rows = reliability_rows([0.82, 0.88, 0.55], [True, False, True])

    assert [row["bin"] for row in rows] == BIN_NAMES
    by_bin = _bins(rows)
    high = by_bin["80-90"]
    assert high["count"] == 2
    assert high["lower"] == pytest.approx(0.8)
    assert high["upper"] == pytest.approx(0.9)
    assert high["mean_confidence"] == pytest.approx(0.85)
    assert high["accuracy"] == pytest.approx(0.5)
    assert high["gap"] == pytest.approx(0.35)
    assert by_bin["50-60"]["gap"] == pytest.approx(0.55 - 1.0)
    empty = by_bin["60-70"]
    assert empty["count"] == 0
    assert empty["mean_confidence"] is None
    assert empty["accuracy"] is None
    assert empty["gap"] is None
    assert by_bin["0-50"]["count"] == 0


def test_reliability_edges_go_up_and_one_goes_to_top_bin() -> None:
    edges = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    rows = reliability_rows(edges, [True] * len(edges))

    counts = {row["bin"]: row["count"] for row in rows}
    assert counts == {
        "0-50": 0,
        "50-60": 1,
        "60-70": 1,
        "70-80": 1,
        "80-90": 1,
        "90-100": 2,
    }


def test_model_confidence_below_fifty_goes_to_low_bin() -> None:
    rows = reliability_rows([0.1, 0.3, 0.49, 0.0], [True, False, False, True])

    low = _bins(rows)["0-50"]
    assert low["count"] == 4
    assert low["lower"] == 0.0
    assert low["upper"] == 0.5
    assert low["mean_confidence"] == pytest.approx((0.1 + 0.3 + 0.49) / 4)
    assert low["accuracy"] == pytest.approx(0.5)
    assert sum(row["count"] for row in rows) == 4


def test_parse_label_accepts_number_or_name() -> None:
    assert parse_label("1", CATEGORIES) == "cat"
    assert parse_label(" 3 ", CATEGORIES) == "sun"
    assert parse_label("Moon", CATEGORIES) == "moon"
    assert parse_label("  SUN ", CATEGORIES) == "sun"
    for bad in ("0", "4", "dog", "", "-1"):
        with pytest.raises(ValueError, match="pick"):
            parse_label(bad, CATEGORIES)


@pytest.mark.parametrize(
    ("text", "expected"),
    [("50", 50), ("77", 77), ("100", 100), (" 70% ", 70), ("85%", 85)],
)
def test_parse_confidence_accepts_50_to_100_with_optional_percent(
    text: str, expected: int
) -> None:
    assert parse_confidence(text) == expected


@pytest.mark.parametrize(
    "text", ["0", "49", "101", "-1", "30%", "abc", "85.5", "", "%"]
)
def test_parse_confidence_refuses_outside_50_to_100(text: str) -> None:
    with pytest.raises(ValueError, match="50 to 100"):
        parse_confidence(text)


def test_parse_answers_accepts_schema() -> None:
    raw = [
        {"key_id": "1", "label": "cat", "confidence": 85},
        {"key_id": "2", "label": "sun", "confidence": 50},
        {"key_id": "3", "label": "cat", "confidence": 77},
        {"key_id": "4", "label": "sun", "confidence": 100},
    ]

    answers = parse_answers(raw, ["1", "2", "3", "4"], CATEGORIES)

    assert answers == (
        DuelAnswer("1", "cat", 85),
        DuelAnswer("2", "sun", 50),
        DuelAnswer("3", "cat", 77),
        DuelAnswer("4", "sun", 100),
    )


GOOD = {"key_id": "1", "label": "cat", "confidence": 85}


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ({"key_id": "1"}, "list"),
        ([], "1 round"),
        ([GOOD, GOOD], "1 round"),
        (["cat"], "object"),
        ([{"key_id": "1", "label": "cat"}], "confidence"),
        ([{**GOOD, "key_id": "9"}], "key_id"),
        ([{**GOOD, "label": "dog"}], "label"),
        ([{**GOOD, "label": 1}], "label"),
        ([{**GOOD, "confidence": 101}], "50 to 100"),
        ([{**GOOD, "confidence": 0}], "50 to 100"),
        ([{**GOOD, "confidence": 49}], "50 to 100"),
        ([{**GOOD, "confidence": -1}], "50 to 100"),
        ([{**GOOD, "confidence": 85.0}], "50 to 100"),
        ([{**GOOD, "confidence": True}], "50 to 100"),
        ([{**GOOD, "confidence": "85"}], "50 to 100"),
    ],
)
def test_parse_answers_refuses_bad_shape_label_confidence_or_order(
    raw: object, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        parse_answers(raw, ["1"], CATEGORIES)


def test_duel_receipt_has_scores_reliability_rows_and_no_image_bytes() -> None:
    source_rows = [
        _source_row("1", "cat", "cat", 0.9),
        _source_row("2", "moon", "sun", 0.4),
    ]
    answers = [DuelAnswer("1", "moon", 70), DuelAnswer("2", "moon", 60)]
    records = [
        score_round(i, row, answer, CATEGORIES)
        for i, (row, answer) in enumerate(zip(source_rows, answers, strict=True), 1)
    ]

    receipt = build_duel_receipt(
        records,
        source_rows=source_rows,
        categories=CATEGORIES,
        input_mode="answers_file",
        source_receipt={"path": "s.json", "sha256": "ab", "backend": "b", "model": "m"},
        per_category=2,
    )

    assert receipt["issue"] == 412
    assert receipt["kind"] == "doodle_duel_play"
    assert receipt["input_mode"] == "answers_file"
    assert receipt["rounds"] == 2
    assert receipt["pins"] == {
        "categories": list(CATEGORIES),
        "per_category": 2,
        "dataset_license": "CC BY 4.0",
        "credit": "Google Quick, Draw!",
    }
    rows = receipt["rows"]
    assert [set(row) for row in rows] == [ROW_KEYS, ROW_KEYS]
    assert rows[0]["player_correct"] is False
    assert rows[0]["player_confidence"] == pytest.approx(0.7)
    assert rows[0]["player_brier"] == pytest.approx(0.49)
    assert rows[0]["model_brier"] == pytest.approx(0.01)
    assert rows[1]["model_confidence"] == pytest.approx(0.4)
    assert rows[1]["model_correct"] is False
    assert rows[1]["model_brier"] == pytest.approx(0.16)
    scores = receipt["scores"]
    assert scores["baseline_brier"] == 0.25
    assert scores["player"] == pytest.approx(
        {"accuracy": 0.5, "mean_brier": (0.49 + 0.16) / 2, "mean_confidence": 0.65}
    )
    model = scores["model"]
    assert model["accuracy"] == 0.5
    assert model["mean_brier"] == pytest.approx((0.01 + 0.16) / 2)
    assert model["mean_confidence"] == pytest.approx(0.65)
    expected_mc = (
        multiclass_brier(source_rows[0]["probabilities"], "cat", CATEGORIES)
        + multiclass_brier(source_rows[1]["probabilities"], "moon", CATEGORIES)
    ) / 2
    assert model["multiclass_brier"] == pytest.approx(expected_mc)
    assert [r["bin"] for r in receipt["reliability"]["player"]] == BIN_NAMES
    assert _bins(receipt["reliability"]["model"])["0-50"]["count"] == 1
    text = json.dumps(receipt)
    assert "base64" not in text
    assert "PNG" not in text
    assert ".png" not in text
