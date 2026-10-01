"""Offline tests for the calibration map writer (#352 slice 2).

The Noul path fits one committed #343 series, the CEDAR vLLM signature
receipt. The fitted parameters and metrics must equal the committed
``post_hoc_receipt.json`` row for that series. The Score path uses a small
synthetic series from sha256 draws. Every written map loads through the
slice-1 reader and validator in the library. No test calls a model.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from typevet.adapters.inbound import load_calibration_map
from typevet_evals.calibration import (
    CLIP,
    Series,
    fit_isotonic,
    fit_platt,
    fit_temperature,
    split_indices,
)
from typevet_evals.calibration_artifact import (
    calibration_map_artifact,
    score_level_series,
    write_calibration_map,
)

pytestmark = pytest.mark.unit

FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"
SOURCE = "cedar/receipts/signature_match_vllm.json"
SERIES_NAME = "signatures/signature_match_vllm"
POST_HOC = FIXTURES / "calibration" / "post_hoc_receipt.json"
FITTERS = {
    "temperature": fit_temperature,
    "platt": fit_platt,
    "isotonic": fit_isotonic,
}
METHODS = tuple(FITTERS)
ROUND_TRIP = 1e-12


def _reproduces(apply: Any, fitted: Any, probabilities: list[float]) -> bool:
    """Return whether the loaded map equals the fitter after the domain clip.

    The domain clips each output to ``[CLIP, 1 - CLIP]``. An isotonic step
    can be exactly 0 or 1, so the raw difference reaches ``CLIP`` plus float
    noise. The clipped comparison is exact to ``ROUND_TRIP`` and so stays
    inside the ``1e-6`` bound of the contract.
    """
    return all(
        abs(apply(p) - min(max(fitted(p), CLIP), 1.0 - CLIP)) <= ROUND_TRIP
        for p in probabilities
    )


def _receipt() -> dict[str, Any]:
    return json.loads((FIXTURES / SOURCE).read_text(encoding="utf-8"))


def _signature_series() -> Series:
    rows = _receipt()["pairs"]
    return Series(
        SERIES_NAME,
        SOURCE,
        tuple(str(r["pair_id"]) for r in rows),
        tuple(float(r["same_writer_confidence"]) for r in rows),
        tuple(bool(r["gold_same_writer"]) for r in rows),
    )


def _post_hoc_row() -> dict[str, Any]:
    receipt = json.loads(POST_HOC.read_text(encoding="utf-8"))
    return next(s for s in receipt["series"] if s["name"] == SERIES_NAME)


def _noul_artifact(method: str) -> dict[str, Any]:
    receipt = _receipt()
    return calibration_map_artifact(
        _signature_series(),
        method=method,
        task_id="signature-same-writer",
        model=receipt["model"],
        backend=receipt["backend"],
        receipt=FIXTURES / SOURCE,
    )


def _halves(series: Series) -> tuple[list[float], list[bool], list[float]]:
    cal, ev = split_indices(series.ids)
    return (
        [series.probabilities[i] for i in cal],
        [series.labels[i] for i in cal],
        [series.probabilities[i] for i in ev],
    )


def _uniform(stream: str, index: int) -> float:
    digest = hashlib.sha256(f"352:{stream}:{index}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def _score_rows(n: int) -> tuple[list[str], list[dict[int, float]], list[int]]:
    ids, probabilities, gold = [], [], []
    for i in range(n):
        weights = [0.05 + _uniform(f"w{k}", i) for k in range(3)]
        top = max(range(3), key=lambda k: weights[k])
        weights[top] *= 6.0
        total = sum(weights)
        ids.append(f"row-{i}")
        probabilities.append({k: w / total for k, w in enumerate(weights)})
        gold.append(top if _uniform("y", i) < 0.6 else (top + 1) % 3)
    return ids, probabilities, gold


@pytest.mark.parametrize("method", METHODS)
def test_noul_map_loads_and_reproduces_the_fitter(method: str, tmp_path: Path) -> None:
    artifact = _noul_artifact(method)
    path = tmp_path / f"{method}.json"
    digest = write_calibration_map(path, artifact)
    cmap = load_calibration_map(path, sha256=digest)
    series = _signature_series()
    cal_p, cal_y, ev_p = _halves(series)
    fitted = FITTERS[method](cal_p, cal_y)
    assert cmap.method == method
    assert cmap.fitted_on.model == "google/gemma-4-31B-it"
    assert cmap.fitted_on.backend == "vllm"
    assert cmap.fitted_on.receipt == SOURCE
    assert cmap.fitted_on.n_calibration == len(cal_p)
    assert cmap.evaluation.n_evaluation == len(ev_p)
    assert cmap.levels is None
    assert _reproduces(cmap.apply, fitted, ev_p)


@pytest.mark.parametrize("method", METHODS)
def test_noul_map_matches_the_committed_post_hoc_row(method: str) -> None:
    artifact = _noul_artifact(method)
    row = _post_hoc_row()
    evaluation = artifact["evaluation"]
    assert artifact["fitted_on"]["n_calibration"] == row["n_calibration"]
    assert evaluation["n_evaluation"] == row["n_evaluation"]
    assert evaluation["before"] == pytest.approx(row["before"], abs=1e-9)
    assert evaluation["after"] == pytest.approx(
        row["methods"][method]["after"], abs=1e-9
    )
    assert evaluation["rule_met"] is row["methods"][method]["rule_met"]
    params = artifact["parameters"]
    if method == "isotonic":
        assert len(params["knots"]) == row["methods"][method]["params"]["knots"]
    else:
        assert params == pytest.approx(row["methods"][method]["params"], abs=1e-9)


def test_receipt_sha256_is_lower_case_hex_of_the_receipt_bytes() -> None:
    artifact = _noul_artifact("platt")
    digest = artifact["fitted_on"]["receipt_sha256"]
    committed = json.loads(POST_HOC.read_text(encoding="utf-8"))["sources"][SOURCE]
    assert digest == hashlib.sha256((FIXTURES / SOURCE).read_bytes()).hexdigest()
    assert digest == committed
    assert digest == digest.lower()
    assert len(digest) == 64


def test_artifact_names_the_schema_and_the_producer() -> None:
    artifact = _noul_artifact("platt")
    assert artifact["schema"] == "typevet.calibration_map/1"
    assert artifact["producer"] == {"typevet_evals": "0.0.0"}
    assert "levels" not in artifact


def test_write_returns_the_sha256_of_the_written_bytes(tmp_path: Path) -> None:
    artifact = _noul_artifact("isotonic")
    path = tmp_path / "map.json"
    digest = write_calibration_map(path, artifact)
    data = path.read_bytes()
    assert digest == hashlib.sha256(data).hexdigest()
    assert json.loads(data) == artifact
    assert data.endswith(b"\n")
    assert write_calibration_map(tmp_path / "again.json", artifact) == digest


def test_score_level_series_keeps_the_levels_of_a_row_in_one_half() -> None:
    ids, probabilities, gold = _score_rows(60)
    series = score_level_series(
        "synthetic/score", "synthetic.json", ids, probabilities, gold
    )
    assert len(series.ids) == 3 * len(ids)
    assert series.ids[:3] == ("row-0",) * 3
    assert series.probabilities[:3] == tuple(probabilities[0][k] for k in range(3))
    assert series.labels[:3] == tuple(k == gold[0] for k in range(3))
    cal, ev = split_indices(series.ids)
    cal_rows = {series.ids[i] for i in cal}
    assert cal_rows.isdisjoint(series.ids[i] for i in ev)
    assert sum(series.labels) == len(ids)


@pytest.mark.parametrize("method", METHODS)
def test_pooled_score_map_carries_levels_and_round_trips(
    method: str, tmp_path: Path
) -> None:
    ids, probabilities, gold = _score_rows(120)
    receipt = tmp_path / "score_receipt.json"
    receipt.write_text('{"rows": []}\n', encoding="utf-8")
    series = score_level_series(
        "synthetic/score", "score_receipt.json", ids, probabilities, gold
    )
    artifact = calibration_map_artifact(
        series,
        method=method,
        task_id="essay-quality",
        model="fake-judgment",
        backend="llama_cpp",
        receipt=receipt,
        levels=3,
    )
    assert artifact["levels"] == 3
    path = tmp_path / "score_map.json"
    cmap = load_calibration_map(path, sha256=write_calibration_map(path, artifact))
    assert cmap.levels == 3
    cal_p, cal_y, ev_p = _halves(series)
    fitted = FITTERS[method](cal_p, cal_y)
    assert _reproduces(cmap.apply, fitted, ev_p)


@pytest.mark.parametrize(
    ("levels", "error"),
    [(1, ValueError), (0, ValueError), (True, TypeError), (2.0, TypeError)],
)
def test_artifact_refuses_a_bad_level_count(
    levels: Any, error: type[Exception]
) -> None:
    with pytest.raises(error, match="levels"):
        calibration_map_artifact(
            _signature_series(),
            method="platt",
            task_id="t",
            model="m",
            backend="vllm",
            receipt=FIXTURES / SOURCE,
            levels=levels,
        )


def test_artifact_refuses_an_unknown_method() -> None:
    with pytest.raises(ValueError, match="method"):
        calibration_map_artifact(
            _signature_series(),
            method="beta",
            task_id="t",
            model="m",
            backend="vllm",
            receipt=FIXTURES / SOURCE,
        )


def test_artifact_refuses_a_series_with_an_empty_half() -> None:
    cal, _ = split_indices(_signature_series().ids)
    one = _signature_series().ids[cal[0]]
    series = Series("x/y", SOURCE, (one, one), (0.2, 0.8), (False, True))
    with pytest.raises(ValueError, match="half"):
        calibration_map_artifact(
            series,
            method="platt",
            task_id="t",
            model="m",
            backend="vllm",
            receipt=FIXTURES / SOURCE,
        )


@pytest.mark.parametrize(
    ("probabilities", "gold"),
    [
        ([{0: 0.5, 1: 0.5}, {0: 0.2, 1: 0.3, 2: 0.5}], [0, 1]),
        ([{0: 1.0}, {0: 1.0}], [0, 0]),
        ([{0: 0.5, 1: 0.5}, {0: 0.4, 1: 0.6}], [0, 2]),
        ([{0: 0.5, 1: 0.5}], [0, 1]),
    ],
)
def test_score_level_series_refuses_inconsistent_rows(
    probabilities: list[dict[int, float]], gold: list[int]
) -> None:
    with pytest.raises(ValueError, match="level"):
        score_level_series("x/y", "y.json", ["a", "b"], probabilities, gold)
