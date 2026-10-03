"""Unit tests for the doodle Choice run, its metrics and its receipt (#412)."""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Mapping
from typing import Any

import pytest

from typevet.domain import ImageInput, JudgmentResponse, Question
from typevet.testing import ScriptedJudgmentFake
from typevet_evals.datasets.quickdraw import Doodle
from typevet_evals.doodle_duel import (
    DOODLE_CATEGORIES,
    DOODLE_QUESTION,
    DoodleOutcome,
    build_doodle_receipt,
    build_doodle_request,
    doodle_metrics,
    run_doodle_duel,
    shuffle_rows,
)

pytestmark = pytest.mark.unit

PNG = b"\x89PNG\r\n\x1a\nsynthetic-doodle-bytes"


class _ByWordPort:
    """Port that answers each image with the distribution for its doodle."""

    def __init__(self, answers: Mapping[bytes, Mapping[str, float]]) -> None:
        self._answers = answers

    def judge(
        self,
        state: str | dict[str, Any] | list[Any],
        questions: Mapping[str, Question | Mapping[str, Any]],
        model: str,
        *,
        media: tuple[ImageInput, ...] | None = None,
        off_option_threshold: float | None = None,
    ) -> JudgmentResponse:
        assert off_option_threshold is None
        assert media is not None
        assert len(media) == 1
        fake = ScriptedJudgmentFake({DOODLE_QUESTION: self._answers[media[0].data]})
        return fake.judge(state, questions, model, media=media)


def _doodle(key_id: str, word: str) -> Doodle:
    return Doodle(key_id=key_id, word=word, countrycode="US", strokes=(((0, 0),),))


def _clock() -> Any:
    ticks = iter(float(i) for i in range(100))
    return lambda: next(ticks)


def _run() -> Any:
    png_a, png_b = PNG + b"a", PNG + b"b"
    requests = [
        build_doodle_request(_doodle("k1", "cat"), png=png_a),
        build_doodle_request(_doodle("k2", "pizza"), png=png_b),
    ]
    port = _ByWordPort({png_a: {"pizza": 0.7, "cat": 0.3}, png_b: {"pizza": 1.0}})
    return run_doodle_duel(port, requests, "fake", clock=_clock())


def test_row_carries_true_chosen_and_every_probability() -> None:
    run = _run()
    assert run.stopped is None
    row = run.outcomes[0].to_receipt()
    assert set(row) == {
        "key_id",
        "true_label",
        "chosen_label",
        "probabilities",
        "correct",
        "latency_seconds",
    }
    assert row["key_id"] == "k1"
    assert row["true_label"] == "cat"
    assert row["chosen_label"] == "pizza"
    assert row["correct"] is False
    assert row["latency_seconds"] == pytest.approx(1.0)
    probabilities = row["probabilities"]
    assert list(probabilities) == list(DOODLE_CATEGORIES)
    assert probabilities["pizza"] == pytest.approx(0.7)
    assert probabilities["cat"] == pytest.approx(0.3)
    assert probabilities["spider"] == 0.0
    assert run.outcomes[1].to_receipt()["correct"] is True


def _outcome(word: str, chosen: str, p: float) -> DoodleOutcome:
    probabilities = dict.fromkeys(DOODLE_CATEGORIES, 0.0)
    probabilities[chosen] = p
    return DoodleOutcome(f"{word}-{chosen}", word, chosen, probabilities, 0.5)


def test_metrics_report_accuracy_per_category() -> None:
    outcomes = [
        _outcome("cat", "cat", 0.9),
        _outcome("cat", "pizza", 0.6),
        _outcome("pizza", "pizza", 0.5),
    ]
    metrics = doodle_metrics(outcomes)
    assert metrics["count"] == 3
    assert metrics["accuracy"] == pytest.approx(2 / 3)
    assert metrics["per_category_accuracy"] == {"cat": 0.5, "pizza": 1.0}
    assert metrics["mean_chosen_probability"] == pytest.approx(2.0 / 3)
    empty = doodle_metrics([])
    assert empty["count"] == 0
    assert empty["accuracy"] is None
    assert empty["mean_chosen_probability"] is None


def test_row_order_is_seeded() -> None:
    rows = [_doodle(f"k{i:02d}", "cat") for i in range(30)]
    first = shuffle_rows(rows, 0, key=lambda d: d.key_id)
    expected = sorted(
        rows, key=lambda d: hashlib.sha256(f"0:{d.key_id}".encode()).digest()
    )
    assert first == expected
    assert first != rows
    assert shuffle_rows(rows, 0, key=lambda d: d.key_id) == first
    other = shuffle_rows(rows, 1, key=lambda d: d.key_id)
    assert other != first
    assert sorted(other, key=lambda d: d.key_id) == rows
    assert shuffle_rows(list(reversed(rows)), 0, key=lambda d: d.key_id) == first


def test_receipt_has_credit_and_no_image_bytes() -> None:
    run = _run()
    receipt = build_doodle_receipt(
        run,
        backend="fake",
        model="fake",
        pins={"per_category": 1},
        identity={"run_id": "r"},
    )
    assert receipt["issue"] == 412
    assert receipt["pins"]["dataset_license"] == "CC BY 4.0"
    assert receipt["pins"]["credit"] == "Google Quick, Draw!"
    assert receipt["pins"]["per_category"] == 1
    assert receipt["throughput"]["images"] == 2
    assert receipt["stopped"] is None
    assert set(receipt["metrics"]) == {
        "count",
        "accuracy",
        "per_category_accuracy",
        "mean_chosen_probability",
    }
    assert [row["key_id"] for row in receipt["rows"]] == ["k1", "k2"]
    text = json.dumps(receipt)
    assert "PNG" not in text
    assert base64.b64encode(PNG).decode()[:12] not in text
