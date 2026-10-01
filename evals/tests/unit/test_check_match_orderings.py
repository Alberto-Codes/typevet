"""Unit tests for the balanced option-order study of the verdict (#105).

The contract pre-registers the design, the averaging rule and the decision
rule before any live run. These tests prove each rule offline: the K=6
design puts every option once in every position, ordering 0 is today's
order, the mean is the arithmetic mean of post-softmax probabilities
(TypeLLM's rule), and the statistics function gives the pre-registered
numbers and decision label on small fixture receipts.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from typevet.domain import (
    Choice,
    ChoiceAnswer,
    GenerationError,
    ImageInput,
    JudgmentResponse,
    Question,
)
from typevet_evals.check_match import (
    VERDICT,
    VERDICT_LABELS,
    build_check_match_request,
    check_cases,
    check_match_questions,
)
from typevet_evals.check_match.orderings import (
    ADOPT_OPT_IN,
    DO_NOT_ADOPT,
    INCONCLUSIVE,
    POSITION_SPREAD_LIMIT,
    balanced_orders,
    build_orderings_receipt,
    decision_label,
    mean_probabilities,
    orderings_statistics,
    position_probabilities,
    remap_positions,
    reordered_choice,
    run_orderings,
    single_order_cases,
)

pytestmark = pytest.mark.unit

_PNG = b"\x89PNG\r\n\x1a\nfixture"


def _softmax(logits: Sequence[float]) -> list[float]:
    top = max(logits)
    exps = [math.exp(v - top) for v in logits]
    total = math.fsum(exps)
    return [v / total for v in exps]


def test_k6_design_puts_each_option_once_in_each_position() -> None:
    orders = balanced_orders(6)

    assert len(orders) == 6
    for order in orders:
        assert sorted(order) == list(range(6))
    for position in range(6):
        assert sorted(order[position] for order in orders) == list(range(6))


def test_k6_design_balances_each_ordered_neighbour_pair() -> None:
    orders = balanced_orders(6)
    pairs = [(o[i], o[i + 1]) for o in orders for i in range(5)]

    assert len(set(pairs)) == 30 == len(pairs)


def test_odd_k_design_doubles_and_stays_balanced_by_position() -> None:
    orders = balanced_orders(3)

    assert len(orders) == 6
    for position in range(3):
        column = [order[position] for order in orders]
        assert all(column.count(option) == 2 for option in range(3))


@pytest.mark.parametrize("k", [0, -1])
def test_design_refuses_an_empty_option_set(k: int) -> None:
    with pytest.raises(ValueError, match="at least 1"):
        balanced_orders(k)


def test_ordering_zero_is_the_canonical_order_and_its_remap_is_identity() -> None:
    orders = balanced_orders(len(VERDICT_LABELS))
    probs = [0.4, 0.2, 0.15, 0.1, 0.1, 0.05]

    assert orders[0] == tuple(range(6))
    remapped = remap_positions(VERDICT_LABELS, orders[0], probs)
    assert remapped == dict(zip(VERDICT_LABELS, probs, strict=True))
    assert list(remapped) == list(VERDICT_LABELS)


def test_reordered_choice_lists_the_options_in_the_ordering() -> None:
    base = check_match_questions()[VERDICT]
    assert isinstance(base, Choice)
    orders = balanced_orders(6)

    same = reordered_choice(base, VERDICT_LABELS, orders[0])
    assert list(same.criteria.items()) == list(base.criteria.items())
    moved = reordered_choice(base, VERDICT_LABELS, orders[1])
    assert tuple(moved.criteria) == tuple(VERDICT_LABELS[i] for i in orders[1])
    assert moved.instructions == base.instructions
    assert all(moved.criteria[k] == base.criteria[k] for k in VERDICT_LABELS)


def test_remap_and_position_view_are_inverse() -> None:
    labels = ("a", "b", "c")
    order = (2, 0, 1)
    remapped = remap_positions(labels, order, [0.5, 0.3, 0.2])

    assert remapped == {"a": 0.3, "b": 0.2, "c": 0.5}
    assert position_probabilities(labels, order, remapped) == (0.5, 0.3, 0.2)


def test_average_probabilities_not_logits() -> None:
    labels = ("a", "b", "c")
    order0, order1 = (0, 1, 2), (2, 0, 1)
    logits0, logits1 = [2.0, 0.0, -1.0], [0.5, 1.5, -2.0]
    first = remap_positions(labels, order0, _softmax(logits0))
    second = remap_positions(labels, order1, _softmax(logits1))

    mean = mean_probabilities([first, second], labels)

    for label in labels:
        assert mean[label] == pytest.approx((first[label] + second[label]) / 2)
    by_label1 = remap_positions(labels, order1, logits1)
    pooled = _softmax([(logits0[i] + by_label1[k]) / 2 for i, k in enumerate(labels)])
    assert list(mean.values()) != pytest.approx(pooled)
    assert math.fsum(mean.values()) == pytest.approx(1.0)


def test_mean_refuses_no_orderings() -> None:
    with pytest.raises(ValueError, match="at least one ordering"):
        mean_probabilities([], ("a",))


def _case(
    case_id: str,
    expected: list[str],
    single: str,
    orderings: Sequence[Mapping[str, float]],
) -> dict[str, object]:
    orders = (("x", "y"), ("y", "x"))
    return {
        "case_id": case_id,
        "expected_verdicts": expected,
        "single_order": {
            "verdict": single,
            "verdict_probabilities": {"x": 0.5, "y": 0.5},
        },
        "orderings": [
            {"order": list(order), "probabilities": dict(probs)}
            for order, probs in zip(orders, orderings, strict=True)
        ],
    }


def _adopt_fixture(b_expected: str = "y") -> list[dict[str, object]]:
    return [
        _case("A", ["x"], "x", [{"x": 0.9, "y": 0.1}, {"x": 0.7, "y": 0.3}]),
        _case("B", [b_expected], "x", [{"x": 0.6, "y": 0.4}, {"x": 0.2, "y": 0.8}]),
    ]


def test_statistics_give_the_pre_registered_numbers_and_adopt() -> None:
    stats = orderings_statistics(_adopt_fixture())

    assert stats["cases"] == 2
    assert stats["position_means"] == pytest.approx([0.65, 0.35])
    assert stats["position_spread"] == pytest.approx(0.30)
    assert stats["argmax_stability"] == 0.5
    assert stats["averaged_accuracy"] == 1.0
    assert stats["single_order_accuracy"] == 0.5
    assert stats["ordering0_agreement"] == 1.0
    assert stats["ordering0_max_abs_difference"] == pytest.approx(0.4)
    assert stats["spread_limit"] == POSITION_SPREAD_LIMIT
    assert stats["decision"] == ADOPT_OPT_IN


def test_statistics_label_inconclusive_when_averaging_loses_accuracy() -> None:
    stats = orderings_statistics(_adopt_fixture(b_expected="x"))

    assert stats["position_spread"] == pytest.approx(0.30)
    assert stats["averaged_accuracy"] == 0.5
    assert stats["single_order_accuracy"] == 1.0
    assert stats["decision"] == INCONCLUSIVE


def test_statistics_label_do_not_adopt_for_a_small_spread() -> None:
    cases = [_case("A", ["x"], "x", [{"x": 0.5, "y": 0.5}, {"x": 0.52, "y": 0.48}])]

    stats = orderings_statistics(cases)

    assert stats["position_means"] == pytest.approx([0.49, 0.51])
    assert stats["position_spread"] == pytest.approx(0.02)
    assert stats["decision"] == DO_NOT_ADOPT


@pytest.mark.parametrize(
    ("spread", "averaged", "single", "label"),
    [
        (0.05, 0.1, 0.9, DO_NOT_ADOPT),
        (0.0501, 0.5, 0.5, ADOPT_OPT_IN),
        (0.2, 0.49, 0.5, INCONCLUSIVE),
    ],
)
def test_decision_rule_boundaries(
    spread: float, averaged: float, single: float, label: str
) -> None:
    assert decision_label(spread, averaged, single) == label


def test_statistics_refuse_no_cases() -> None:
    with pytest.raises(ValueError, match="at least one case"):
        orderings_statistics([])


class _FakePort:
    """Answer each verdict call from a fixed per-position distribution."""

    def __init__(self, position_probs: Sequence[float], fail_at: int = -1) -> None:
        self.position_probs = list(position_probs)
        self.fail_at = fail_at
        self.calls: list[tuple[str, ...]] = []
        self.seen: list[tuple[object, object]] = []

    def judge(
        self,
        state: str | dict[str, Any] | list[Any],
        questions: Mapping[str, Question | Mapping[str, Any]],
        model: str,
        *,
        media: tuple[ImageInput, ...] | None = None,
    ) -> JudgmentResponse:
        assert tuple(questions) == (VERDICT,)
        choice = questions[VERDICT]
        assert isinstance(choice, Choice)
        listed = tuple(choice.criteria)
        if len(self.calls) == self.fail_at:
            raise GenerationError("backend down")
        self.calls.append(listed)
        self.seen.append((state, media))
        probs = dict(zip(listed, self.position_probs, strict=True))
        best = max(probs, key=probs.__getitem__)
        answer = ChoiceAnswer(choice=best, confidence=probs[best], probabilities=probs)
        return JudgmentResponse(model=model, answers={VERDICT: answer})


def _requests(rows: int = 1) -> list:
    return [build_check_match_request(c, image=_PNG) for c in check_cases(1, rows)]


def _single(requests: Sequence) -> dict[str, dict[str, object]]:
    return {
        r.case_id: {
            "case_id": r.case_id,
            "verdict": "consistent",
            "verdict_probabilities": dict.fromkeys(VERDICT_LABELS, 1 / 6),
            "render_sha256": "unused",
        }
        for r in requests
    }


def test_run_scores_each_ordering_once_and_measures_the_position_effect() -> None:
    requests = _requests()
    orders = balanced_orders(6)
    port = _FakePort([0.5, 0.1, 0.1, 0.1, 0.1, 0.1])

    run = run_orderings(port, requests, "m", orders, _single(requests))

    assert run.stopped is None
    assert len(port.calls) == 6 * len(requests)
    assert port.calls[:6] == [tuple(VERDICT_LABELS[i] for i in o) for o in orders]
    record = run.records[0]
    assert record["mean_probabilities"] == pytest.approx(
        dict.fromkeys(VERDICT_LABELS, 0.5 / 6 + 0.5 / 6)
    )
    assert record["argmax_stable"] is False
    stats = orderings_statistics(run.records)
    assert stats["position_spread"] == pytest.approx(0.4)
    assert stats["decision"] in {ADOPT_OPT_IN, INCONCLUSIVE}
    receipt = build_orderings_receipt(
        run, backend="llama_cpp", model="m", pins={"p": 1}, identity={"i": 2}
    )
    assert receipt["issue"] == 105
    assert receipt["requests"] == 6 * len(requests)
    assert receipt["statistics"] == stats
    assert receipt["cases"][0]["single_order"]["verdict"] == "consistent"


def test_run_stops_at_the_first_backend_failure() -> None:
    requests = _requests()
    port = _FakePort([0.5, 0.1, 0.1, 0.1, 0.1, 0.1], fail_at=8)

    run = run_orderings(port, requests, "m", balanced_orders(6), _single(requests))

    assert len(run.records) == 1
    assert run.stopped is not None
    assert run.stopped["case_id"] == requests[1].case_id
    assert run.stopped["ordering"] == 2
    assert run.stopped["error_class"] == "GenerationError"
    receipt = build_orderings_receipt(run, backend="b", model="m", pins={}, identity={})
    assert receipt["statistics"]["cases"] == 1


def test_run_with_no_records_has_no_statistics() -> None:
    requests = _requests()
    port = _FakePort([0.5, 0.1, 0.1, 0.1, 0.1, 0.1], fail_at=0)

    run = run_orderings(port, requests, "m", balanced_orders(6), _single(requests))
    receipt = build_orderings_receipt(run, backend="b", model="m", pins={}, identity={})

    assert run.records == ()
    assert receipt["statistics"] is None


def test_single_order_cases_match_case_ids_and_render_digests() -> None:
    requests = _requests()
    digest = hashlib.sha256(_PNG).hexdigest()
    cases = [
        {"case_id": r.case_id, "render_sha256": digest, "verdict": "consistent"}
        for r in requests
    ]
    receipt = {"pins": {"generator_seed": 1}, "cases": [*cases, {"case_id": "r09:x"}]}

    found = single_order_cases(receipt, requests, seed=1)
    assert list(found) == [r.case_id for r in requests]

    with pytest.raises(ValueError, match="seed"):
        single_order_cases(receipt, requests, seed=2)
    cases[0]["render_sha256"] = "other"
    with pytest.raises(ValueError, match="render"):
        single_order_cases(receipt, requests, seed=1)
    with pytest.raises(ValueError, match="missing"):
        single_order_cases(
            {"pins": {"generator_seed": 1}, "cases": []}, requests, seed=1
        )


def test_media_and_state_are_forwarded_unchanged() -> None:
    request = _requests()[0]
    port = _FakePort([1 / 6] * 6)

    run_orderings(port, [request], "m", balanced_orders(6), _single([request]))

    assert port.seen == [(request.state, request.media)] * 6
    assert isinstance(request.media[0], ImageInput)


def test_run_refuses_a_verdict_that_is_not_a_choice() -> None:
    request = _requests()[0]
    other = dataclasses.replace(
        request, questions={VERDICT: check_match_questions()["legibility"]}
    )

    with pytest.raises(TypeError, match="not a Choice"):
        run_orderings(
            _FakePort([1.0]), [other], "m", balanced_orders(6), _single([request])
        )
