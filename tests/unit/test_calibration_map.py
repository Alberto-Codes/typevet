"""Unit tests for caller-supplied Noul calibration maps (#352 slice 1)."""

from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from typevet.adapters.inbound.calibration_map import load_calibration_map
from typevet.domain.calibration import (
    CALIBRATION_CLIP,
    CALIBRATION_MAP_SCHEMA,
    CalibrationMap,
    CalibrationRecord,
    calibration_map_from_mapping,
)
from typevet.domain.errors import (
    CalibrationDigestError,
    CalibrationMapError,
    CalibrationModelMismatchError,
    CalibrationTargetError,
    CalibrationTaskMismatchError,
    JudgmentError,
)
from typevet.domain.judgment_answers import ChoiceAnswer, NoulAnswer, ScoreAnswer
from typevet.domain.judgment_questions import Choice, Noul, Question, Score
from typevet.domain.judgment_response import JudgmentResponse
from typevet.runtime import CalibratedJudgment

pytestmark = pytest.mark.unit

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "calibration"
PLATT = FIXTURE / "platt_noul_map.json"
DIGEST = "a" * 64


def _base() -> dict[str, Any]:
    return json.loads(PLATT.read_text(encoding="utf-8"))


def _map(**overrides: Any) -> CalibrationMap:
    data = _base()
    data.update(overrides)
    return calibration_map_from_mapping(data, sha256=DIGEST)


def _isotonic() -> CalibrationMap:
    return _map(
        method="isotonic",
        parameters={"knots": [0.2, 0.5, 0.8], "values": [0.1, 0.4, 0.9]},
    )


class _StubJudgment:
    """Inner judgment port that returns one fixed response."""

    def __init__(self, response: JudgmentResponse) -> None:
        self.response = response
        self.calls = 0

    def judge(
        self,
        state: str | dict[str, Any] | list[Any],
        questions: Mapping[str, Question | Mapping[str, Any]],
        model: str,
        *,
        media: Any = None,
    ) -> JudgmentResponse:
        self.calls += 1
        return self.response


def _wrapper(
    answer: Any = None, *, model: str = "fake-judgment", **kwargs: Any
) -> tuple[CalibratedJudgment, _StubJudgment]:
    response = JudgmentResponse(
        model=model, answers={"fraud": answer or NoulAnswer(noul=0.6)}
    )
    inner = _StubJudgment(response)
    options: dict[str, Any] = {"task_id": "fraud-message", "backend": "llama_cpp"}
    options.update(kwargs)
    return CalibratedJudgment(inner, {"fraud": _map()}, **options), inner


# --- validator -------------------------------------------------------------


def test_fixture_map_validates_with_every_recorded_field() -> None:
    cmap = calibration_map_from_mapping(_base(), sha256=DIGEST)
    assert cmap.method == "platt"
    assert cmap.sha256 == DIGEST
    assert cmap.fitted_on.task_id == "fraud-message"
    assert cmap.fitted_on.model == "fake-judgment"
    assert cmap.fitted_on.backend == "llama_cpp"
    assert cmap.fitted_on.n_calibration == 68
    assert cmap.evaluation.n_evaluation == 90
    assert cmap.evaluation.before["ece"] == pytest.approx(0.2)
    assert cmap.evaluation.after["accuracy"] == pytest.approx(0.89)
    assert cmap.producer == "0.6.0"
    assert CALIBRATION_MAP_SCHEMA == "typevet.calibration_map/1"


def test_failed_rule_map_is_recorded_not_refused() -> None:
    data = _base()
    data["evaluation"]["rule_met"] = False
    data["evaluation"]["after"]["ece"] = 0.5
    cmap = calibration_map_from_mapping(data, sha256=DIGEST)
    assert cmap.evaluation.rule_met is False


def _broken(path: tuple[str, ...], value: Any) -> dict[str, Any]:
    data = copy.deepcopy(_base())
    target = data
    for key in path[:-1]:
        target = target[key]
    if value is _DELETE:
        del target[path[-1]]
    else:
        target[path[-1]] = value
    return data


_DELETE = object()

_MALFORMED = [
    (("schema",), "typevet.calibration_map/2"),
    (("schema",), _DELETE),
    (("method",), "beta"),
    (("parameters",), {"slope": 2.0}),
    (("parameters",), {"slope": 2.0, "intercept": "x"}),
    (("parameters",), {"slope": float("nan"), "intercept": 0.0}),
    (("parameters",), {"slope": True, "intercept": 0.0}),
    (("parameters",), {"slope": 1.0, "intercept": 0.0, "extra": 1.0}),
    (("parameters",), []),
    (("fitted_on",), _DELETE),
    (("fitted_on", "task_id"), ""),
    (("fitted_on", "model"), _DELETE),
    (("fitted_on", "backend"), 3),
    (("fitted_on", "receipt_sha256"), "abc"),
    (("fitted_on", "n_calibration"), 0),
    (("fitted_on", "n_calibration"), 1.5),
    (("evaluation", "n_evaluation"), -1),
    (("evaluation", "before"), {"ece": 0.1, "brier": 0.1}),
    (("evaluation", "after", "brier"), float("inf")),
    (("evaluation", "rule_met"), "yes"),
    (("producer",), {"other": "1"}),
    (("producer",), "0.6.0"),
]


@pytest.mark.parametrize(("path", "value"), _MALFORMED)
def test_malformed_map_raises_the_base_error(path: tuple[str, ...], value: Any) -> None:
    with pytest.raises(CalibrationMapError) as caught:
        calibration_map_from_mapping(_broken(path, value), sha256=DIGEST)
    assert type(caught.value) is CalibrationMapError
    assert isinstance(caught.value, JudgmentError)


def test_non_mapping_document_raises_the_base_error() -> None:
    document: Any = []
    with pytest.raises(CalibrationMapError):
        calibration_map_from_mapping(document, sha256=DIGEST)


@pytest.mark.parametrize(
    "parameters",
    [
        {"temperature": 0.0},
        {"temperature": -1.0},
        {"knots": [], "values": []},
        {"knots": [0.2, 0.2], "values": [0.1, 0.2]},
        {"knots": [0.5, 0.2], "values": [0.1, 0.2]},
        {"knots": [0.2, 0.5], "values": [0.3, 0.1]},
        {"knots": [0.2, 0.5], "values": [0.1]},
        {"knots": [0.2, 1.5], "values": [0.1, 0.2]},
        {"knots": [0.2, 0.5], "values": [0.1, 1.2]},
        {"knots": "0.2", "values": [0.1]},
    ],
)
def test_bad_method_parameters_raise_the_base_error(
    parameters: dict[str, Any],
) -> None:
    method = "temperature" if "temperature" in parameters else "isotonic"
    with pytest.raises(CalibrationMapError):
        _map(method=method, parameters=parameters)


@pytest.mark.parametrize("digest", ["", "abc", "g" * 64, "A" * 63])
def test_domain_refuses_malformed_digest(digest: str) -> None:
    with pytest.raises(CalibrationDigestError):
        calibration_map_from_mapping(_base(), sha256=digest)


# --- maths, clip and knot edges -------------------------------------------


def test_platt_matches_the_closed_form() -> None:
    cmap = _map()
    z = 2.0 * math.log(0.6 / 0.4) - 1.0
    assert cmap.apply(0.6) == pytest.approx(1.0 / (1.0 + math.exp(-z)))


def test_temperature_matches_the_closed_form() -> None:
    cmap = _map(method="temperature", parameters={"temperature": 2.0})
    z = math.log(0.8 / 0.2) / 2.0
    assert cmap.apply(0.8) == pytest.approx(1.0 / (1.0 + math.exp(-z)))


def test_inputs_and_outputs_are_clipped() -> None:
    steep = _map(parameters={"slope": 1000.0, "intercept": 0.0})
    assert steep.apply(1.0) == pytest.approx(1.0 - CALIBRATION_CLIP)
    assert steep.apply(0.0) == pytest.approx(CALIBRATION_CLIP)
    identity = _map(method="temperature", parameters={"temperature": 1.0})
    assert identity.apply(0.0) == pytest.approx(CALIBRATION_CLIP)
    assert identity.apply(1.0) == pytest.approx(1.0 - CALIBRATION_CLIP)
    hard = _map(method="isotonic", parameters={"knots": [0.5], "values": [1.0]})
    assert hard.apply(0.9) == pytest.approx(1.0 - CALIBRATION_CLIP)
    zero = _map(method="isotonic", parameters={"knots": [0.5], "values": [0.0]})
    assert zero.apply(0.9) == pytest.approx(CALIBRATION_CLIP)


@pytest.mark.parametrize(
    ("p", "expected"),
    [
        (0.0, 0.1),  # below the first knot: the first knot value
        (0.1999, 0.1),
        (0.2, 0.1),  # on a knot: that knot's value
        (0.3, 0.1),  # between knots: the nearest lower knot value
        (0.5, 0.4),
        (0.7999, 0.4),
        (0.8, 0.9),
        (1.0, 0.9),  # above the last knot: the last knot value
    ],
)
def test_isotonic_knot_edges(p: float, expected: float) -> None:
    assert _isotonic().apply(p) == pytest.approx(expected)


@pytest.mark.parametrize("p", [-0.1, 1.1, float("nan"), True])
def test_apply_refuses_a_non_probability(p: Any) -> None:
    with pytest.raises(CalibrationMapError):
        _map().apply(p)


def test_calibration_record_fields() -> None:
    record = CalibrationRecord(
        raw=0.6, calibrated=0.45, method="platt", map_sha256=DIGEST
    )
    assert (record.raw, record.calibrated) == (0.6, 0.45)


# --- file reader and digest -----------------------------------------------


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_loader_accepts_matching_digest_and_records_it() -> None:
    digest = _sha(PLATT)
    cmap = load_calibration_map(PLATT, sha256=digest.upper())
    assert cmap.sha256 == digest
    assert cmap.method == "platt"


@pytest.mark.parametrize("digest", [None, "", "xyz", "0" * 65])
def test_loader_refuses_missing_or_malformed_digest(digest: Any) -> None:
    with pytest.raises(CalibrationDigestError) as caught:
        load_calibration_map(PLATT, sha256=digest)
    if digest:
        assert digest not in str(caught.value)


def test_loader_refuses_a_tampered_file(tmp_path: Path) -> None:
    digest = _sha(PLATT)
    tampered = tmp_path / "map.json"
    tampered.write_bytes(PLATT.read_bytes().replace(b"2.0", b"3.0", 1))
    with pytest.raises(CalibrationDigestError) as caught:
        load_calibration_map(tampered, sha256=digest)
    message = str(caught.value)
    assert digest not in message
    assert _sha(tampered) not in message


@pytest.mark.parametrize("body", [b"{not json", b"[1, 2]", b"\xff\xfe"])
def test_loader_refuses_a_malformed_document(tmp_path: Path, body: bytes) -> None:
    path = tmp_path / "map.json"
    path.write_bytes(body)
    with pytest.raises(CalibrationMapError) as caught:
        load_calibration_map(path, sha256=_sha(path))
    assert type(caught.value) is CalibrationMapError


# --- wrapper refusals and application -------------------------------------


def test_wrapper_calibrates_noul_and_records_raw() -> None:
    wrapper, _ = _wrapper()
    response = wrapper.judge("text", {"fraud": Noul()}, "fake-judgment")
    record = response.calibration["fraud"]
    assert record.raw == pytest.approx(0.6)
    assert response.nouls["fraud"].noul == pytest.approx(record.calibrated)
    assert record.calibrated == pytest.approx(_map().apply(0.6))
    assert record.method == "platt"
    assert record.map_sha256 == DIGEST


def test_wrapper_leaves_unmapped_questions_raw() -> None:
    response = JudgmentResponse(
        model="fake-judgment",
        answers={"other": NoulAnswer(noul=0.3)},
    )
    wrapper = CalibratedJudgment(
        _StubJudgment(response),
        {"fraud": _map()},
        task_id="fraud-message",
        backend="llama_cpp",
    )
    out = wrapper.judge("text", {"other": Noul()}, "fake-judgment")
    assert out.nouls["other"].noul == pytest.approx(0.3)
    assert out.calibration == {}


def test_task_mismatch_refused_at_construction() -> None:
    with pytest.raises(CalibrationTaskMismatchError) as caught:
        _wrapper(task_id="other-task")
    assert "other-task" not in str(caught.value)
    assert "fraud-message" not in str(caught.value)


def test_backend_mismatch_refused_at_construction() -> None:
    with pytest.raises(CalibrationModelMismatchError) as caught:
        _wrapper(backend="vllm")
    assert "vllm" not in str(caught.value)


def test_model_mismatch_refused_at_judgment_time() -> None:
    wrapper, _ = _wrapper(model="other-model")
    with pytest.raises(CalibrationModelMismatchError) as caught:
        wrapper.judge("text", {"fraud": Noul()}, "other-model")
    assert "other-model" not in str(caught.value)
    assert "fake-judgment" not in str(caught.value)


@pytest.mark.parametrize(
    "question",
    [Choice(criteria={"a": "A", "b": "B"}), Score(criteria=["Low", "High"])],
)
def test_mapped_choice_or_score_question_refused_before_io(
    question: Question,
) -> None:
    wrapper, inner = _wrapper()
    with pytest.raises(CalibrationTargetError):
        wrapper.judge("text", {"fraud": question}, "fake-judgment")
    assert inner.calls == 0


@pytest.mark.parametrize(
    "answer",
    [
        ChoiceAnswer(choice="a", confidence=0.7, probabilities={"a": 0.7, "b": 0.3}),
        ScoreAnswer(
            score=0.7,
            confidence=0.7,
            legend={0: "L", 1: "H"},
            probabilities={0: 0.3, 1: 0.7},
        ),
    ],
)
def test_mapped_choice_or_score_answer_refused(answer: Any) -> None:
    wrapper, _ = _wrapper(answer)
    with pytest.raises(CalibrationTargetError):
        wrapper.judge("text", {"fraud": {"type": "wire"}}, "fake-judgment")


def test_already_calibrated_answer_refused() -> None:
    wrapper, _ = _wrapper()
    outer = CalibratedJudgment(
        wrapper, {"fraud": _map()}, task_id="fraud-message", backend="llama_cpp"
    )
    with pytest.raises(CalibrationMapError):
        outer.judge("text", {"fraud": Noul()}, "fake-judgment")


@pytest.mark.parametrize("maps", [{}, {"": None}, {"fraud": None}])
def test_wrapper_refuses_empty_or_malformed_maps(maps: dict[str, Any]) -> None:
    inner = _StubJudgment(JudgmentResponse(model="m"))
    with pytest.raises(CalibrationMapError):
        CalibratedJudgment(inner, maps, task_id="fraud-message", backend="llama_cpp")
