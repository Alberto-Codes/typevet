"""Contract: a calibration map over ScoringJudgmentAdapter (#352 slice 1).

The real file reader loads a fixed Platt map. ``CalibratedJudgment`` wraps
``ScoringJudgmentAdapter`` over ``ScriptedScoringFake``. The answer carries
the calibrated Noul; the response records the raw value next to it.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import pytest

from typevet.adapters.inbound import load_calibration_map
from typevet.domain import CalibrationModelMismatchError, Noul
from typevet.runtime import CalibratedJudgment, ScoringJudgmentAdapter
from typevet.testing import ScriptedScoringFake

pytestmark = pytest.mark.contract

PLATT = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "calibration"
    / "platt_noul_map.json"
)


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
