"""Offline tests for check-match metrics, runner and receipt (#316).

Expected values are hand computed in the comments. The runner tests use a
fake judgment port and synthetic image bytes. No test reads the network or
renders a check.
"""

from __future__ import annotations

import hashlib
import json
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
from typevet_evals.check_match import (
    AMOUNTS_MATCH,
    EXPECTED_LABELS,
    LEGIBILITY,
    PAYEE_MATCHES,
    VERDICT,
    VERDICT_LABELS,
    CheckMatchOutcome,
    CheckVariant,
    accuracy_by_group,
    agreement_rate,
    build_check_match_receipt,
    build_check_match_request,
    check_cases,
    check_match_metrics,
    false_clear_rate,
    legibility_gap,
    noul_choice_agreement,
    run_check_match,
    score_summary,
    verdict_correct,
)

pytestmark = pytest.mark.unit

_IMAGE = b"synthetic check bytes"
_V = CheckVariant
_E = EXPECTED_LABELS


# --- verdict correctness and accuracy ---------------------------------------


def test_verdict_correct_uses_the_accepted_set() -> None:
    assert verdict_correct("consistent", _E[_V.CLEAN])
    assert not verdict_correct("cannot_tell", _E[_V.CLEAN])
    assert verdict_correct("amount_mismatch", _E[_V.WRITTEN_AMOUNT_CHANGED])
    assert not verdict_correct("payee_mismatch", _E[_V.WRITTEN_AMOUNT_CHANGED])


def test_low_legibility_accepts_consistent_and_cannot_tell() -> None:
    low = _E[_V.LOW_LEGIBILITY]
    assert verdict_correct("consistent", low)
    assert verdict_correct("cannot_tell", low)
    assert not verdict_correct("unsigned", low)


def test_verdict_correct_rejects_unknown_label() -> None:
    with pytest.raises(ValueError, match="unknown verdict label: maybe"):
        verdict_correct("maybe", _E[_V.CLEAN])


def test_accuracy_by_group_hand_computed() -> None:
    # a: T, F -> 0.5; b: T -> 1.0; c: F, F, T -> 1/3.
    keys = ["a", "b", "a", "c", "c", "c"]
    correct = [True, True, False, False, False, True]
    assert accuracy_by_group(keys, correct) == {"a": 0.5, "b": 1.0, "c": 1 / 3}


def test_accuracy_by_group_rejects_unequal_lengths() -> None:
    with pytest.raises(ValueError, match="equal length"):
        accuracy_by_group(["a"], [])


# --- false-clear rate -------------------------------------------------------


def test_false_clear_counts_consistent_on_mismatch_variants() -> None:
    # Counted: payee, written, both, date, unsigned (5). Cleared: payee and
    # unsigned -> 2/5. Clean and low legibility are not counted.
    verdicts = [
        "consistent",
        "consistent",
        "amount_mismatch",
        "cannot_tell",
        "date_mismatch",
        "consistent",
        "consistent",
    ]
    variants = [
        _V.PAYEE_CHANGED,
        _V.CLEAN,
        _V.WRITTEN_AMOUNT_CHANGED,
        _V.BOTH_AMOUNTS_CHANGED,
        _V.WRONG_DATE,
        _V.UNSIGNED,
        _V.LOW_LEGIBILITY,
    ]
    assert false_clear_rate(verdicts, [_E[v] for v in variants]) == 2 / 5


def test_false_clear_excludes_low_legibility() -> None:
    assert false_clear_rate(["consistent"], [_E[_V.LOW_LEGIBILITY]]) is None


def test_false_clear_is_none_without_mismatch_cases() -> None:
    assert false_clear_rate(["consistent"], [_E[_V.CLEAN]]) is None


# --- Noul-Choice agreement --------------------------------------------------


@pytest.mark.parametrize(
    ("verdict", "payee", "amounts", "expected"),
    [
        ("payee_mismatch", 0.2, 0.9, True),
        ("payee_mismatch", 0.7, 0.9, False),
        ("payee_mismatch", 0.2, 0.1, True),
        ("amount_mismatch", 0.9, 0.3, True),
        ("amount_mismatch", 0.9, 0.5, False),
        ("consistent", 0.5, 0.5, True),
        ("consistent", 0.49, 0.9, False),
        ("consistent", 0.9, 0.2, False),
        ("date_mismatch", 0.9, 0.9, True),
        ("date_mismatch", 0.1, 0.9, False),
        ("unsigned", 0.9, 0.9, True),
        ("unsigned", 0.9, 0.1, False),
        ("cannot_tell", 0.1, 0.1, None),
    ],
)
def test_noul_choice_agreement_rule(
    verdict: str, payee: float, amounts: float, expected: bool | None
) -> None:
    assert noul_choice_agreement(verdict, payee, amounts) is expected


def test_noul_choice_agreement_rejects_unknown_label() -> None:
    with pytest.raises(ValueError, match="unknown verdict label"):
        noul_choice_agreement("maybe", 0.5, 0.5)


def test_agreement_rate_skips_none() -> None:
    assert agreement_rate([True, None, False, True]) == 2 / 3
    assert agreement_rate([None]) is None


# --- Score by variant and the legibility gap --------------------------------


def test_score_summary_mean_and_distribution() -> None:
    summary = score_summary([4.0, 3.0, 2.5], [4, 3, 3], n_levels=5)
    assert summary == {
        "count": 3,
        "mean": 3.1666666666666665,
        "distribution": {"0": 0, "1": 0, "2": 0, "3": 2, "4": 1},
    }


def test_score_summary_rejects_out_of_range_level() -> None:
    with pytest.raises(ValueError, match=r"not in 0\.\.4"):
        score_summary([1.0], [5], n_levels=5)


def test_legibility_gap_is_clean_minus_low() -> None:
    by_variant = {"clean": {"mean": 3.75}, "low_legibility": {"mean": 1.25}}
    assert legibility_gap(by_variant) == 2.5
    assert legibility_gap({"clean": {"mean": 3.0}}) is None


# --- runner -----------------------------------------------------------------


def _response(verdict: str, payee: float, amounts: float, level: int):
    probabilities = {v: (0.75 if v == verdict else 0.05) for v in VERDICT_LABELS}
    levels = dict.fromkeys(range(5), 0.0)
    levels[level] = 1.0
    return JudgmentResponse(
        model="fake-model",
        usage=TokenUsage(input_tokens=500),
        answers={
            PAYEE_MATCHES: NoulAnswer(noul=payee),
            AMOUNTS_MATCH: NoulAnswer(noul=amounts),
            VERDICT: ChoiceAnswer(
                choice=verdict, confidence=0.75, probabilities=probabilities
            ),
            LEGIBILITY: ScoreAnswer(
                score=float(level),
                confidence=1.0,
                legend={i: f"level {i}" for i in range(5)},
                probabilities=levels,
            ),
        },
    )


class _FakePort:
    """Judgment port that replays responses or raises at one call."""

    def __init__(self, responses: list[JudgmentResponse | Exception]) -> None:
        self._responses = responses
        self.media_counts: list[int] = []

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


def _requests():
    return [build_check_match_request(c, image=_IMAGE) for c in check_cases(count=1)]


# One right answer per variant, in enum order, with clean legibility 4 and low
# legibility 1.
_RIGHT = [
    ("consistent", 0.9, 0.9, 4),
    ("payee_mismatch", 0.1, 0.9, 4),
    ("amount_mismatch", 0.9, 0.2, 4),
    ("amount_mismatch", 0.9, 0.1, 4),
    ("date_mismatch", 0.9, 0.9, 4),
    ("unsigned", 0.9, 0.9, 3),
    ("cannot_tell", 0.8, 0.8, 1),
]


def test_run_records_typed_answers_and_render_digest() -> None:
    port = _FakePort([_response(*r) for r in _RIGHT])
    run = run_check_match(port, _requests(), "fake-model", clock=_Clock())
    assert port.media_counts == [1] * 7
    assert run.stopped is None
    first = run.outcomes[0]
    assert first.case_id == "r00:clean"
    assert first.variant == "clean"
    assert first.verdict == "consistent"
    assert first.payee_confidence == 0.9
    assert first.legibility_level == 4
    assert first.render_sha256 == hashlib.sha256(_IMAGE).hexdigest()
    assert first.latency_seconds == 0.5
    assert first.input_tokens == 500
    assert first.correct is True
    assert run.outcomes[1].agreement is True


def test_run_stops_on_first_transport_failure() -> None:
    port = _FakePort([_response(*_RIGHT[0]), TransportError("refused")])
    run = run_check_match(port, _requests(), "fake-model", clock=_Clock())
    assert len(run.outcomes) == 1
    assert run.stopped == {
        "index": 1,
        "case_id": "r00:payee_changed",
        "error_class": "TransportError",
        "message": "refused",
    }


def test_run_stops_on_backend_http_error() -> None:
    error = BackendHttpError("HTTP 500", status_code=500, body_snippet="boom")
    run = run_check_match(_FakePort([error]), _requests(), "m", clock=_Clock())
    assert run.outcomes == ()
    assert run.stopped is not None
    assert run.stopped["error_class"] == "BackendHttpError"


def _run(answers):
    port = _FakePort(
        [a if isinstance(a, Exception) else _response(*a) for a in answers]
    )
    return run_check_match(port, _requests(), "fake-model", clock=_Clock())


def test_metrics_all_right() -> None:
    metrics = check_match_metrics(_run(_RIGHT).outcomes)
    assert metrics["cases"] == 7
    assert metrics["accuracy"] == 1.0
    assert set(metrics["accuracy_by_variant"].values()) == {1.0}
    assert metrics["accuracy_by_class"]["cannot_tell|consistent"] == 1.0
    assert metrics["false_clear_rate"] == 0.0
    assert metrics["false_clear_cases"] == 5
    assert metrics["cannot_tell_rate"] == 1 / 7
    assert metrics["nouls"][PAYEE_MATCHES]["roc_auc"] == 1.0
    assert metrics["nouls"][AMOUNTS_MATCH]["roc_auc"] == 1.0
    assert len(metrics["nouls"][PAYEE_MATCHES]["reliability"]) == 10
    assert metrics["score_by_variant"]["clean"]["mean"] == 4.0
    assert metrics["legibility_gap_clean_minus_low"] == 3.0
    agreement = metrics["noul_choice_agreement"]
    assert (agreement["rate"], agreement["counted"]) == (1.0, 6)


def test_metrics_noul_ece_hand_computed() -> None:
    # Payee: bin 1 (0.1, false) gap 0.1; bin 8 (0.8, true) gap 0.2; bin 9
    # (five at 0.9, true) gap 5 x 0.1. ECE = 0.8 / 7.
    # Amounts: bin 1 (0.1, false) 0.1; bin 2 (0.2, false) 0.2; bin 8 (0.8,
    # true) 0.2; bin 9 (four at 0.9, true) 0.4. ECE = 0.9 / 7.
    nouls = check_match_metrics(_run(_RIGHT).outcomes)["nouls"]
    assert nouls[PAYEE_MATCHES]["ece"] == pytest.approx(0.8 / 7)
    assert nouls[AMOUNTS_MATCH]["ece"] == pytest.approx(0.9 / 7)


def test_metrics_cannot_tell_by_variant_uses_each_variant() -> None:
    # Only the low-legibility case answers cannot_tell: 1/1 there, 0/1 clean.
    by_variant = check_match_metrics(_run(_RIGHT).outcomes)["cannot_tell_by_variant"]
    assert by_variant["low_legibility"] == 1.0
    assert by_variant["clean"] == 0.0


def test_metrics_false_clear_by_variant_keys_are_mismatch_variants() -> None:
    by_variant = check_match_metrics(_run(_RIGHT).outcomes)["false_clear_by_variant"]
    assert set(by_variant) == {
        "payee_changed",
        "written_amount_changed",
        "both_amounts_changed",
        "wrong_date",
        "unsigned",
    }


def test_metrics_false_clear_and_wrong_verdicts() -> None:
    # Payee and date cases answered consistent: 2 of 5 cleared; 5/7 right.
    answers = list(_RIGHT)
    answers[1] = ("consistent", 0.9, 0.9, 4)
    answers[4] = ("consistent", 0.9, 0.9, 4)
    metrics = check_match_metrics(_run(answers).outcomes)
    assert metrics["false_clear_rate"] == 2 / 5
    assert metrics["false_clear_by_variant"]["payee_changed"] == 1.0
    assert metrics["false_clear_by_variant"]["unsigned"] == 0.0
    assert metrics["accuracy"] == 5 / 7
    assert metrics["accuracy_by_variant"]["payee_changed"] == 0.0
    assert metrics["accuracy_by_class"]["payee_mismatch"] == 0.0
    # Payee: truth False only on the payee case, now 0.9 among 0.9 and 0.8.
    assert metrics["nouls"][PAYEE_MATCHES]["roc_auc"] < 0.5


def test_metrics_disagreement_counts() -> None:
    answers = list(_RIGHT)
    answers[1] = ("payee_mismatch", 0.9, 0.9, 4)
    metrics = check_match_metrics(_run(answers).outcomes)
    assert metrics["noul_choice_agreement"]["rate"] == 5 / 6


def test_metrics_with_no_outcomes() -> None:
    metrics = check_match_metrics(())
    assert metrics["cases"] == 0
    assert metrics["accuracy"] is None
    assert metrics["false_clear_rate"] is None
    assert metrics["nouls"][PAYEE_MATCHES]["roc_auc"] is None
    assert metrics["nouls"][PAYEE_MATCHES]["ece"] is None
    assert metrics["legibility_gap_clean_minus_low"] is None


def test_outcome_receipt_holds_expected_labels() -> None:
    outcome = _run([*_RIGHT[:1], TransportError("x")]).outcomes[0]
    assert isinstance(outcome, CheckMatchOutcome)
    row = outcome.to_receipt()
    assert row["expected_verdicts"] == ["consistent"]
    assert row["payee_truth"] is True
    assert row["legibility_probabilities"]["4"] == 1.0


def test_receipt_is_json_without_image_bytes() -> None:
    run = _run(_RIGHT)
    receipt = build_check_match_receipt(
        run,
        backend="llama_cpp",
        model="fake-model",
        pins={"generator_seed": 0},
        identity={"schema": "x"},
    )
    text = json.dumps(receipt)
    assert receipt["issue"] == 316
    assert [c["case_id"] for c in receipt["cases"]][:2] == [
        "r00:clean",
        "r00:payee_changed",
    ]
    assert receipt["pins"] == {"generator_seed": 0}
    assert "synthetic check bytes" not in text
    assert "c3ludGhldGlj" not in text  # base64 prefix of the image bytes
