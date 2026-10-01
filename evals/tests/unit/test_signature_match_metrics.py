"""Offline tests for signature-match metrics, runner and receipt (#319).

Expected values are hand computed in the comments. The runner tests use a
fake judgment port and synthetic image bytes. No test reads the network, the
CEDAR cache or a signature image.
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
from typevet_evals.datasets.cedar import CedarPair, CedarSignature, PairKind
from typevet_evals.signature_match import (
    IMAGE_QUALITY,
    SAME_WRITER,
    VERDICT,
    VERDICT_LABELS,
    SignatureMatchOutcome,
    SignatureMatchRun,
    build_signature_match_receipt,
    build_signature_match_request,
    kind_accuracy,
    noul_accept_rate,
    noul_choice_agreement,
    run_signature_match,
    signature_match_metrics,
    skilled_false_accept,
    verdict_accept_rate,
    verdict_accuracy,
    verdict_says_same_writer,
)

pytestmark = pytest.mark.unit

_IMAGE = b"synthetic signature bytes"
GG = PairKind.GENUINE_GENUINE
GS = PairKind.GENUINE_SKILLED
GR = PairKind.GENUINE_RANDOM


# --- verdict mapping and accuracy ------------------------------------------


def test_verdict_mapping_treats_skilled_and_different_as_different() -> None:
    assert verdict_says_same_writer("same_writer") is True
    assert verdict_says_same_writer("different_writer") is False
    assert verdict_says_same_writer("skilled_forgery_suspected") is False
    assert verdict_says_same_writer("cannot_tell") is None


def test_verdict_mapping_rejects_unknown_label() -> None:
    with pytest.raises(ValueError, match="unknown verdict"):
        verdict_says_same_writer("same_person")


def test_accuracy_is_same_writer_accuracy_with_cannot_tell_wrong() -> None:
    # same/T right, skilled/F right, different/F right, cannot_tell/T wrong,
    # same/F wrong: 3/5.
    verdicts = [
        "same_writer",
        "skilled_forgery_suspected",
        "different_writer",
        "cannot_tell",
        "same_writer",
    ]
    gold = [True, False, False, True, False]
    assert verdict_accuracy(verdicts, gold) == pytest.approx(0.6)


def test_accuracy_counts_cannot_tell_wrong_on_different_writer_pairs() -> None:
    # cannot_tell/F wrong, different/F right: 1/2.
    assert verdict_accuracy(["cannot_tell", "different_writer"], [False, False]) == 0.5


def test_kind_accuracy_needs_the_label_of_the_pair_kind() -> None:
    # gg/same right, gs/different wrong, gr/different right, gs/skilled right.
    verdicts = [
        "same_writer",
        "different_writer",
        "different_writer",
        "skilled_forgery_suspected",
    ]
    assert kind_accuracy(verdicts, [GG, GS, GR, GS]) == 0.75


def test_accuracy_functions_reject_bad_input() -> None:
    with pytest.raises(ValueError, match="equal length"):
        verdict_accuracy(["same_writer"], [True, False])
    with pytest.raises(ValueError, match="empty"):
        kind_accuracy([], [])
    with pytest.raises(ValueError, match="unknown verdict"):
        kind_accuracy(["same_person"], [GG])


# --- accept rates and the skilled false-accept rate ------------------------


def test_noul_accept_rate_counts_one_half_as_accept() -> None:
    # 0.5 and 0.9 are at or above 0.5; 0.49 and 0.1 are not: 2/4.
    assert noul_accept_rate([0.5, 0.49, 0.9, 0.1]) == 0.5


def test_verdict_accept_rate_counts_only_same_writer() -> None:
    verdicts = ["same_writer", "cannot_tell", "skilled_forgery_suspected"]
    assert verdict_accept_rate(verdicts) == pytest.approx(1 / 3)


def test_accept_rates_reject_empty_input() -> None:
    with pytest.raises(ValueError, match="empty"):
        noul_accept_rate([])
    with pytest.raises(ValueError, match="empty"):
        verdict_accept_rate([])


def test_skilled_false_accept_reports_both_definitions_on_skilled_pairs() -> None:
    # Skilled pairs: 0.6 and 0.5 accept by the Noul (2/3); only the third
    # verdict is same_writer (1/3). The gg and gr pairs do not count.
    confidences = [0.6, 0.5, 0.2, 0.9, 0.7]
    verdicts = [
        "skilled_forgery_suspected",
        "skilled_forgery_suspected",
        "same_writer",
        "same_writer",
        "same_writer",
    ]
    result = skilled_false_accept(confidences, verdicts, [GS, GS, GS, GG, GR])
    assert result.pairs == 3
    assert result.noul_rate == pytest.approx(2 / 3)
    assert result.verdict_rate == pytest.approx(1 / 3)


def test_skilled_false_accept_without_skilled_pairs_is_none() -> None:
    result = skilled_false_accept([0.9], ["same_writer"], [GG])
    assert result.pairs == 0
    assert result.noul_rate is None
    assert result.verdict_rate is None


def test_skilled_false_accept_rejects_unequal_lengths() -> None:
    with pytest.raises(ValueError, match="equal length"):
        skilled_false_accept([0.9], ["same_writer", "same_writer"], [GS])


# --- Noul and Choice agreement ---------------------------------------------


def test_agreement_counts_decided_pairs_and_reports_cannot_tell_apart() -> None:
    # 0.9/same agree, 0.2/skilled agree, 0.7/different disagree,
    # 0.5/same agree, 0.3/same disagree, two cannot_tell: 3 of 5 decided.
    confidences = [0.9, 0.2, 0.7, 0.5, 0.3, 0.8, 0.1]
    verdicts = [
        "same_writer",
        "skilled_forgery_suspected",
        "different_writer",
        "same_writer",
        "same_writer",
        "cannot_tell",
        "cannot_tell",
    ]
    result = noul_choice_agreement(confidences, verdicts)
    assert result.pairs == 7
    assert result.decided == 5
    assert result.agree == 3
    assert result.rate == pytest.approx(0.6)
    assert result.cannot_tell == 2
    assert result.cannot_tell_rate == pytest.approx(2 / 7)


def test_agreement_with_only_cannot_tell_has_no_rate() -> None:
    result = noul_choice_agreement([0.9, 0.1], ["cannot_tell", "cannot_tell"])
    assert result.decided == 0
    assert result.rate is None
    assert result.cannot_tell_rate == 1.0


def test_agreement_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="empty"):
        noul_choice_agreement([], [])
    with pytest.raises(ValueError, match="equal length"):
        noul_choice_agreement([0.9], [])


# --- runner ----------------------------------------------------------------


def _pair(writer: int, kind: PairKind) -> CedarPair:
    reference = CedarSignature(writer, 1, forged=False)
    questioned = {
        GG: CedarSignature(writer, 2, forged=False),
        GS: CedarSignature(writer, 1, forged=True),
        GR: CedarSignature(writer + 1, 1, forged=False),
    }[kind]
    return CedarPair(kind, reference, questioned)


def _response(noul: float, verdict: str, level: int) -> JudgmentResponse:
    probabilities = {
        label: (0.7 if label == verdict else 0.1) for label in VERDICT_LABELS
    }
    levels = dict.fromkeys(range(5), 0.0)
    levels[level] = 1.0
    return JudgmentResponse(
        model="fake-model",
        usage=TokenUsage(input_tokens=400),
        answers={
            SAME_WRITER: NoulAnswer(noul=noul),
            VERDICT: ChoiceAnswer(
                choice=verdict, confidence=0.7, probabilities=probabilities
            ),
            IMAGE_QUALITY: ScoreAnswer(
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
        off_option_threshold: float | None = None,
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


def _requests(kinds: list[PairKind]):
    return [
        build_signature_match_request(
            _pair(i + 1, kind), reference_image=_IMAGE, questioned_image=_IMAGE
        )
        for i, kind in enumerate(kinds)
    ]


def test_run_records_typed_answers_and_latency() -> None:
    media: list[int] = []
    port = _FakePort(
        [
            _response(0.9, "same_writer", 4),
            _response(0.2, "skilled_forgery_suspected", 3),
        ],
        media,
    )
    run = run_signature_match(port, _requests([GG, GS]), "fake-model", clock=_Clock())
    assert media == [2, 2]
    assert run.stopped is None
    first, second = run.outcomes
    assert first.pair_id == "genuine_genuine:original_1_1:original_1_2"
    assert first.kind is GG
    assert first.gold_same_writer is True
    assert first.same_writer_confidence == 0.9
    assert first.verdict == "same_writer"
    assert first.quality_level == 4
    assert first.latency_seconds == 0.5
    assert first.input_tokens == 400
    assert second.kind is GS
    assert second.gold_same_writer is False
    assert run.wall_seconds > 0.0


def test_run_stops_on_first_transport_failure() -> None:
    port = _FakePort(
        [
            _response(0.9, "same_writer", 4),
            TransportError("connection refused"),
            _response(0.1, "different_writer", 4),
        ],
        [],
    )
    run = run_signature_match(
        port, _requests([GG, GR, GS]), "fake-model", clock=_Clock()
    )
    assert len(run.outcomes) == 1
    assert run.stopped == {
        "index": 1,
        "pair_id": "genuine_random:original_2_1:original_3_1",
        "error_class": "TransportError",
        "message": "connection refused",
        "discarded": 0,
    }


def test_run_stops_on_backend_http_error() -> None:
    error = BackendHttpError("HTTP 500", status_code=500, body_snippet="boom")
    run = run_signature_match(
        _FakePort([error], []), _requests([GG]), "m", clock=_Clock()
    )
    assert run.outcomes == ()
    assert run.stopped is not None
    assert run.stopped["error_class"] == "BackendHttpError"


def _outcome(
    kind: PairKind, confidence: float, verdict: str, level: int = 4
) -> SignatureMatchOutcome:
    return SignatureMatchOutcome(
        pair_id=f"{kind}:a_{confidence}:b_{verdict}",
        kind=kind,
        gold_same_writer=kind is GG,
        same_writer_confidence=confidence,
        verdict=verdict,
        verdict_probabilities={verdict: 1.0},
        quality_score=float(level),
        quality_level=level,
        quality_probabilities={level: 1.0},
        latency_seconds=2.0,
        input_tokens=400,
    )


def _six_outcomes() -> list[SignatureMatchOutcome]:
    return [
        _outcome(GG, 0.9, "same_writer"),
        _outcome(GG, 0.6, "cannot_tell", level=1),
        _outcome(GS, 0.7, "same_writer"),
        _outcome(GS, 0.2, "skilled_forgery_suspected"),
        _outcome(GR, 0.1, "different_writer"),
        _outcome(GR, 0.4, "skilled_forgery_suspected"),
    ]


def test_metrics_combine_every_measure() -> None:
    metrics = signature_match_metrics(_six_outcomes())
    assert metrics["pairs"] == 6
    assert metrics["same_writer_pairs"] == 2
    # Right same-writer side: gg same, gs skilled, gr different, gr skilled.
    assert metrics["accuracy"] == pytest.approx(4 / 6)
    # Right kind label: gg same, gs skilled, gr different.
    assert metrics["kind_accuracy"] == 0.5
    # Positives 0.9, 0.6 against 0.7, 0.2, 0.1, 0.4: 4 + 3 of 8.
    assert metrics["roc_auc"] == 0.875
    assert metrics["roc_auc_by_negative_kind"] == {
        "genuine_skilled": 0.75,
        "genuine_random": 1.0,
    }
    # Gaps 0.1, 0.4, 0.7, 0.2, 0.1, 0.4 in six separate bins: 1.9 / 6.
    assert metrics["ece"] == pytest.approx(1.9 / 6)
    assert len(metrics["reliability"]) == 10
    assert metrics["cannot_tell_rate"] == pytest.approx(1 / 6)
    assert metrics["skilled_false_accept"] == {
        "pairs": 2,
        "threshold": 0.5,
        "noul_rate": 0.5,
        "verdict_rate": 0.5,
    }
    agreement = metrics["noul_choice_agreement"]
    assert (agreement["decided"], agreement["agree"], agreement["rate"]) == (5, 5, 1.0)
    assert agreement["cannot_tell"] == 1
    assert metrics["mean_latency_seconds"] == 2.0
    assert "match percentage" in metrics["confidence_note"]


def test_metrics_by_kind() -> None:
    by_kind = signature_match_metrics(_six_outcomes())["by_kind"]
    assert list(by_kind) == ["genuine_genuine", "genuine_skilled", "genuine_random"]
    gg, gs, gr = by_kind.values()
    assert (gg["pairs"], gg["accuracy"], gg["cannot_tell_rate"]) == (2, 0.5, 0.5)
    assert (gg["noul_accept_rate"], gg["verdict_accept_rate"]) == (1.0, 0.5)
    assert gg["quality_levels"] == {"0": 0, "1": 1, "2": 0, "3": 0, "4": 1}
    assert (gs["noul_accept_rate"], gs["verdict_accept_rate"]) == (0.5, 0.5)
    assert (gr["accuracy"], gr["kind_accuracy"]) == (1.0, 0.5)
    assert (gr["noul_accept_rate"], gr["verdict_accept_rate"]) == (0.0, 0.0)
    assert gr["verdict_counts"] == {
        "same_writer": 0,
        "different_writer": 1,
        "skilled_forgery_suspected": 1,
        "cannot_tell": 0,
    }
    assert gr["mean_same_writer_confidence"] == pytest.approx(0.25)


def test_metrics_with_no_outcomes() -> None:
    metrics = signature_match_metrics([])
    assert metrics["pairs"] == 0
    for key in ("accuracy", "kind_accuracy", "roc_auc", "ece", "cannot_tell_rate"):
        assert metrics[key] is None
    assert metrics["skilled_false_accept"]["noul_rate"] is None
    assert metrics["noul_choice_agreement"] is None
    assert metrics["by_kind"]["genuine_skilled"]["pairs"] == 0
    assert metrics["by_kind"]["genuine_skilled"]["accuracy"] is None


# --- receipt ---------------------------------------------------------------


def test_receipt_is_key_free_json_without_image_bytes() -> None:
    run = SignatureMatchRun(
        outcomes=(_outcome(GS, 0.3, "skilled_forgery_suspected"),),
        stopped=None,
        wall_seconds=3.0,
    )
    receipt = build_signature_match_receipt(
        run,
        backend="llama_cpp",
        model="fake-model",
        pins={"archive_sha256": "abc"},
        identity={"run_id": "r1"},
    )
    text = json.dumps(receipt)
    assert receipt["issue"] == 319
    assert receipt["pins"] == {"archive_sha256": "abc"}
    row = receipt["pairs"][0]
    assert row["pair_id"] == "genuine_skilled:a_0.3:b_skilled_forgery_suspected"
    assert row["kind"] == "genuine_skilled"
    assert row["quality_probabilities"] == {"4": 1.0}
    assert math.isfinite(row["same_writer_confidence"])
    assert receipt["metrics"]["kind_accuracy"] == 1.0
    assert receipt["wall_seconds"] == 3.0
    assert "synthetic signature bytes" not in text
