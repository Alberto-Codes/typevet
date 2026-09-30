"""Offline tests for face-match metrics, runner and receipt (#301).

Expected values are hand computed in the comments. The runner tests use a
fake judgment port and synthetic image bytes. No test reads the network, the
LFW cache or a face image.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any

import pytest

from typevet.domain import (
    ChoiceAnswer,
    ImageInput,
    JudgmentResponse,
    NoulAnswer,
    Question,
    ScoreAnswer,
    TokenUsage,
)
from typevet.domain.errors import BackendHttpError, TransportError
from typevet_evals.datasets.lfw import LfwFace, LfwPair
from typevet_evals.face_match import (
    FACE_VISIBILITY,
    SAME_PERSON,
    VERDICT,
    FaceMatchOutcome,
    FaceMatchRun,
    build_face_match_receipt,
    build_face_match_request,
    cannot_tell_rate,
    ensure_key_free,
    expected_calibration_error,
    face_match_metrics,
    reliability_table,
    roc_auc,
    run_face_match,
    score_distribution,
    verdict_accuracy,
)

pytestmark = pytest.mark.unit

_IMAGE = b"synthetic image bytes"


# --- verdict accuracy and cannot_tell rate ---------------------------------


def test_accuracy_counts_cannot_tell_as_wrong() -> None:
    # same/T right, different/F right, cannot_tell/T wrong, same/F wrong: 2/4.
    verdicts = ["same_person", "different_person", "cannot_tell", "same_person"]
    gold = [True, False, True, False]
    assert verdict_accuracy(verdicts, gold) == 0.5


def test_accuracy_all_cannot_tell_is_zero() -> None:
    assert verdict_accuracy(["cannot_tell", "cannot_tell"], [True, False]) == 0.0


def test_cannot_tell_rate() -> None:
    verdicts = ["same_person", "cannot_tell", "different_person", "cannot_tell"]
    assert cannot_tell_rate(verdicts) == 0.5


@pytest.mark.parametrize("metric", [verdict_accuracy, roc_auc])
def test_pairwise_metrics_reject_unequal_lengths(metric) -> None:
    with pytest.raises(ValueError, match="equal length"):
        metric(["same_person"] if metric is verdict_accuracy else [0.5], [])


def test_rate_metrics_reject_empty_input() -> None:
    with pytest.raises(ValueError, match="empty"):
        verdict_accuracy([], [])
    with pytest.raises(ValueError, match="empty"):
        cannot_tell_rate([])


def test_accuracy_rejects_unknown_label() -> None:
    with pytest.raises(ValueError, match="maybe"):
        verdict_accuracy(["maybe"], [True])


# --- ROC-AUC ---------------------------------------------------------------


def test_roc_auc_hand_computed_with_a_tie() -> None:
    # Positives 0.9, 0.8; negatives 0.8, 0.3. Pairs: 1 + 1 + 0.5 + 1 = 3.5 / 4.
    assert roc_auc([0.9, 0.8, 0.8, 0.3], [True, False, True, False]) == 0.875


def test_roc_auc_perfect_and_inverted() -> None:
    assert roc_auc([0.9, 0.1], [True, False]) == 1.0
    assert roc_auc([0.1, 0.9], [True, False]) == 0.0


def test_roc_auc_all_ties_is_one_half() -> None:
    assert roc_auc([0.5, 0.5, 0.5, 0.5], [True, False, True, False]) == 0.5


def test_roc_auc_three_way_tie_uses_average_ranks() -> None:
    # Positives 0.7, 0.7; negatives 0.7, 0.2, 0.9.
    # 0.7 vs 0.7 = 0.5, vs 0.2 = 1, vs 0.9 = 0 -> 1.5 each -> 3 / 6.
    scores = [0.7, 0.7, 0.7, 0.2, 0.9]
    gold = [True, True, False, False, False]
    assert roc_auc(scores, gold) == pytest.approx(0.5)


def test_roc_auc_is_none_for_one_class() -> None:
    assert roc_auc([0.2, 0.9], [True, True]) is None


# --- reliability and ECE -------------------------------------------------


def test_ece_hand_computed() -> None:
    # Bin 0: 0.05 vs 0 -> 0.05. Bin 1: 0.15 vs 1 -> 0.85.
    # Bin 9: mean 0.95 vs 0.5 -> 0.45 x 2 = 0.9. (0.05 + 0.85 + 0.9) / 4 = 0.45.
    confidences = [0.05, 0.15, 0.95, 0.95]
    gold = [False, True, True, False]
    assert expected_calibration_error(confidences, gold) == pytest.approx(0.45)


def test_reliability_table_has_ten_bins_and_edges() -> None:
    table = reliability_table([0.0, 0.1, 1.0], [False, True, True])
    assert len(table) == 10
    assert (table[0].lower, table[0].upper) == (0.0, 0.1)
    assert table[0].count == 1
    assert table[1].count == 1
    assert table[1].fraction_same_person == 1.0
    assert table[9].count == 1
    assert table[9].mean_confidence == 1.0
    assert table[5].count == 0
    assert table[5].mean_confidence is None
    assert table[5].fraction_same_person is None


def test_reliability_rejects_out_of_range_confidence() -> None:
    with pytest.raises(ValueError, match=r"1\.5"):
        reliability_table([1.5], [True])


def test_ece_is_zero_when_calibrated() -> None:
    assert expected_calibration_error([1.0, 0.0], [True, False]) == 0.0


# --- Score distribution ----------------------------------------------------


def test_score_distribution_by_label() -> None:
    levels = [4, 4, 3, 0, 4]
    gold = [True, False, True, False, True]
    assert score_distribution(levels, gold, n_levels=5) == {
        "same_person": {"0": 0, "1": 0, "2": 0, "3": 1, "4": 2},
        "different_person": {"0": 1, "1": 0, "2": 0, "3": 0, "4": 1},
    }


def test_score_distribution_rejects_out_of_range_level() -> None:
    with pytest.raises(ValueError, match="level 5"):
        score_distribution([5], [True], n_levels=5)


# --- runner ----------------------------------------------------------------


def _pair(index: int, *, same: bool) -> LfwPair:
    right = "Alpha_Example" if same else "Bravo_Example"
    return LfwPair(
        fold=1,
        left=LfwFace("Alpha_Example", index),
        right=LfwFace(right, index + 1),
        same_person=same,
    )


def _response(noul: float, verdict: str, level: int) -> JudgmentResponse:
    labels = ("same_person", "different_person", "cannot_tell")
    probabilities = {label: (0.8 if label == verdict else 0.1) for label in labels}
    levels = dict.fromkeys(range(5), 0.0)
    levels[level] = 1.0
    return JudgmentResponse(
        model="fake-model",
        usage=TokenUsage(input_tokens=300),
        answers={
            SAME_PERSON: NoulAnswer(noul=noul),
            VERDICT: ChoiceAnswer(
                choice=verdict, confidence=0.8, probabilities=probabilities
            ),
            FACE_VISIBILITY: ScoreAnswer(
                score=float(level),
                confidence=1.0,
                legend={i: f"level {i}" for i in range(5)},
                probabilities=levels,
            ),
        },
    )


class _FakePort:
    """Judgment port that replays responses or raises at one call."""

    def __init__(
        self, responses: list[JudgmentResponse | Exception], media_counts: list[int]
    ) -> None:
        self._responses = responses
        self.media_counts = media_counts

    def judge(
        self,
        state: str | dict[str, Any] | list[Any],
        questions: Mapping[str, Question | Mapping[str, Any]],
        model: str,
        *,
        media: tuple[ImageInput, ...] | None = None,
    ) -> JudgmentResponse:
        del state, questions, model
        self.media_counts.append(len(media or ()))
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        self.now += 0.5
        return self.now


def _requests(count: int):
    return [
        build_face_match_request(
            _pair(i + 1, same=i % 2 == 0), left_image=_IMAGE, right_image=_IMAGE
        )
        for i in range(count)
    ]


def test_run_records_typed_answers_and_latency() -> None:
    media: list[int] = []
    port = _FakePort(
        [
            _response(0.9, "same_person", 4),
            _response(0.2, "different_person", 3),
        ],
        media,
    )
    run = run_face_match(port, _requests(2), "fake-model", clock=_Clock())
    assert media == [2, 2]
    assert run.stopped is None
    first = run.outcomes[0]
    assert first.pair_id == "1:Alpha_Example_0001:Alpha_Example_0002"
    assert first.gold_same_person is True
    assert first.same_person_confidence == 0.9
    assert first.verdict == "same_person"
    assert first.visibility_level == 4
    assert first.latency_seconds == 0.5
    assert first.input_tokens == 300
    assert run.wall_seconds > 0.0


def test_run_stops_on_first_transport_failure() -> None:
    port = _FakePort(
        [
            _response(0.9, "same_person", 4),
            TransportError("connection refused"),
            _response(0.1, "different_person", 4),
        ],
        [],
    )
    run = run_face_match(port, _requests(3), "fake-model", clock=_Clock())
    assert len(run.outcomes) == 1
    assert run.stopped == {
        "index": 1,
        "pair_id": "1:Alpha_Example_0002:Bravo_Example_0003",
        "error_class": "TransportError",
        "message": "connection refused",
    }


def test_run_stops_on_backend_http_error() -> None:
    error = BackendHttpError("HTTP 500", status_code=500, body_snippet="boom")
    run = run_face_match(_FakePort([error], []), _requests(1), "m", clock=_Clock())
    assert run.outcomes == ()
    assert run.stopped is not None
    assert run.stopped["error_class"] == "BackendHttpError"


def _outcome(
    confidence: float, verdict: str, gold: bool, level: int = 4
) -> FaceMatchOutcome:
    return FaceMatchOutcome(
        pair_id=f"1:A_{confidence}:B_{verdict}",
        gold_same_person=gold,
        same_person_confidence=confidence,
        verdict=verdict,
        verdict_probabilities={verdict: 1.0},
        visibility_score=float(level),
        visibility_level=level,
        visibility_probabilities={level: 1.0},
        latency_seconds=2.0,
        input_tokens=300,
    )


def test_face_match_metrics_combines_every_measure() -> None:
    outcomes = [
        _outcome(0.9, "same_person", True),
        _outcome(0.8, "cannot_tell", False, level=1),
        _outcome(0.8, "same_person", True),
        _outcome(0.3, "different_person", False),
    ]
    metrics = face_match_metrics(outcomes)
    assert metrics["pairs"] == 4
    assert metrics["accuracy"] == 0.75
    assert metrics["roc_auc"] == 0.875
    assert metrics["cannot_tell_rate"] == 0.25
    assert metrics["mean_latency_seconds"] == 2.0
    assert len(metrics["reliability"]) == 10
    assert metrics["score_distribution"]["different_person"]["1"] == 1
    assert metrics["confidence_note"].startswith("same_person_confidence is")
    # Bin 3: 0.3 vs 0 -> 0.3. Bin 8: 0.8, 0.8 vs 0.5 -> 0.6. Bin 9: 0.9 vs 1 -> 0.1.
    assert metrics["ece"] == pytest.approx(0.25)


def test_face_match_metrics_with_no_outcomes() -> None:
    metrics = face_match_metrics([])
    assert metrics["pairs"] == 0
    assert metrics["accuracy"] is None
    assert metrics["roc_auc"] is None
    assert metrics["ece"] is None


# --- receipt ---------------------------------------------------------------


def test_receipt_is_key_free_json_without_image_bytes() -> None:
    run = FaceMatchRun(
        outcomes=(_outcome(0.9, "same_person", True),), stopped=None, wall_seconds=3.0
    )
    receipt = build_face_match_receipt(
        run,
        backend="llama_cpp",
        model="fake-model",
        pins={"pairs_sha256": "abc"},
        identity={"run_id": "r1"},
    )
    text = json.dumps(receipt)
    assert receipt["issue"] == 301
    assert receipt["pairs"][0]["pair_id"] == "1:A_0.9:B_same_person"
    assert receipt["metrics"]["accuracy"] == 1.0
    assert receipt["wall_seconds"] == 3.0
    assert "synthetic image bytes" not in text
    assert "match percentage" in receipt["metrics"]["confidence_note"]
    assert math.isfinite(receipt["pairs"][0]["same_person_confidence"])
    ensure_key_free(text, secrets=("sk-secret",))


@pytest.mark.parametrize(
    "text",
    ['{"key": "sk-secret"}', '{"Authorization": "x"}', '{"h": "Bearer abc"}'],
)
def test_ensure_key_free_rejects_secrets_and_auth_headers(text: str) -> None:
    with pytest.raises(ValueError, match="key-free"):
        ensure_key_free(text, secrets=("sk-secret",))


def test_ensure_key_free_ignores_empty_secret() -> None:
    ensure_key_free('{"a": 1}', secrets=("", None))
