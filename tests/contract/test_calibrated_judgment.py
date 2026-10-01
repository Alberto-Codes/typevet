"""Contract: a calibration map over ScoringJudgmentAdapter (#352).

The real file reader loads a fixed Platt map. ``CalibratedJudgment`` wraps
``ScoringJudgmentAdapter`` over ``ScriptedScoringFake``. The answer carries
the calibrated Noul or Score; the response records the raw value next to it.
The Score map is synthetic: #343 fitted no Score series.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import pytest

from typevet.adapters.inbound import load_calibration_map
from typevet.domain import CalibrationModelMismatchError, Noul, Score
from typevet.runtime import CalibratedJudgment, ScoringJudgmentAdapter
from typevet.testing import ScriptedScoringFake

pytestmark = pytest.mark.contract

PLATT = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "calibration"
    / "platt_noul_map.json"
)
PLATT_SCORE = PLATT.with_name("platt_score_map.json")


def _adapter() -> ScoringJudgmentAdapter:
    fake = ScriptedScoringFake(logprobs={"True": math.log(0.6), "False": math.log(0.4)})
    return ScoringJudgmentAdapter(fake, tokenize_content=lambda s: (ord(s[0]),))


def _maps() -> dict:
    digest = hashlib.sha256(PLATT.read_bytes()).hexdigest()
    return {"fraud": load_calibration_map(PLATT, sha256=digest)}


def test_platt_map_calibrates_the_scoring_adapter_noul() -> None:
    maps = _maps()
    raw = _adapter().judge("Send the code now.", {"fraud": Noul()}, "fake-judgment")
    port = CalibratedJudgment(
        _adapter(), maps, task_id="fraud-message", backend="llama_cpp"
    )

    response = port.judge("Send the code now.", {"fraud": Noul()}, "fake-judgment")

    record = response.calibration["fraud"]
    expected = 1.0 / (1.0 + math.exp(-(2.0 * math.log(0.6 / 0.4) - 1.0)))
    assert record.raw == pytest.approx(raw.nouls["fraud"].noul)
    assert record.raw == pytest.approx(0.6)
    assert record.calibrated == pytest.approx(expected)
    assert record.calibrated != pytest.approx(record.raw)
    assert response.nouls["fraud"].noul == pytest.approx(record.calibrated)
    assert record.method == "platt"
    assert record.map_sha256 == maps["fraud"].sha256
    assert response.model == "fake-judgment"


def test_wrapped_adapter_refuses_another_model() -> None:
    port = CalibratedJudgment(
        _adapter(), _maps(), task_id="fraud-message", backend="llama_cpp"
    )
    with pytest.raises(CalibrationModelMismatchError):
        port.judge("Send the code now.", {"fraud": Noul()}, "other-model")


def test_pooled_score_map_calibrates_the_scoring_adapter_score() -> None:
    raw_levels = {0: 0.5, 1: 0.3, 2: 0.2}
    fake = ScriptedScoringFake(
        logprobs={str(k): math.log(p) for k, p in raw_levels.items()}
    )
    adapter = ScoringJudgmentAdapter(fake, tokenize_content=lambda s: (ord(s[0]),))
    digest = hashlib.sha256(PLATT_SCORE.read_bytes()).hexdigest()
    cmap = load_calibration_map(PLATT_SCORE, sha256=digest)
    port = CalibratedJudgment(
        adapter, {"quality": cmap}, task_id="answer-quality", backend="llama_cpp"
    )
    question = Score(criteria=["Poor", "Fair", "Good"], instructions="Rate:")

    response = port.judge("Charged twice.", {"quality": question}, "fake-judgment")

    def platt(p: float) -> float:
        return 1.0 / (1.0 + math.exp(-(2.0 * math.log(p / (1.0 - p)) - 1.0)))

    mapped = {k: platt(p) for k, p in raw_levels.items()}
    expected = {k: v / sum(mapped.values()) for k, v in mapped.items()}
    answer = response.scores["quality"]
    record = response.calibration["quality"]
    assert cmap.levels == 3
    assert answer.probabilities == pytest.approx(expected)
    assert answer.score == pytest.approx(sum(k * p for k, p in expected.items()))
    assert answer.confidence == pytest.approx(max(expected.values()))
    assert answer.legend == {0: "Poor", 1: "Fair", 2: "Good"}
    assert record.raw == pytest.approx(0.7)
    assert record.calibrated == pytest.approx(answer.score)
    assert record.raw_levels == pytest.approx(raw_levels)
    assert record.calibrated_levels == pytest.approx(expected)
    assert record.map_sha256 == digest


def test_calibrated_noul_keeps_the_off_option_flag() -> None:
    fake = ScriptedScoringFake(
        logprobs={"True": math.log(0.42), "False": math.log(0.28)},
        off_option_mass=0.3,
    )
    adapter = ScoringJudgmentAdapter(fake, tokenize_content=lambda s: (ord(s[0]),))
    port = CalibratedJudgment(
        adapter, _maps(), task_id="fraud-message", backend="llama_cpp"
    )

    response = port.judge(
        "Send the code now.",
        {"fraud": Noul()},
        "fake-judgment",
        off_option_threshold=0.1,
    )

    assert response.off_option["fraud"].off_option_flag is True
    assert response.nouls["fraud"].off_option_flag is True
    assert "fraud" in response.calibration


def test_calibrated_score_keeps_the_off_option_flag() -> None:
    fake = ScriptedScoringFake(
        logprobs={"0": math.log(0.35), "1": math.log(0.21), "2": math.log(0.14)},
        off_option_mass=0.3,
    )
    adapter = ScoringJudgmentAdapter(fake, tokenize_content=lambda s: (ord(s[0]),))
    digest = hashlib.sha256(PLATT_SCORE.read_bytes()).hexdigest()
    cmap = load_calibration_map(PLATT_SCORE, sha256=digest)
    port = CalibratedJudgment(
        adapter, {"quality": cmap}, task_id="answer-quality", backend="llama_cpp"
    )
    question = Score(criteria=["Poor", "Fair", "Good"], instructions="Rate:")

    response = port.judge(
        "Charged twice.",
        {"quality": question},
        "fake-judgment",
        off_option_threshold=0.1,
    )

    assert response.off_option["quality"].off_option_flag is True
    assert response.scores["quality"].off_option_flag is True
    assert "quality" in response.calibration
