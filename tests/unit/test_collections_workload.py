"""Unit tests: finvet collections workload mapping and parity ([#236][i236]).

Every record and seed here is synthetic and uses finvet field names only.

Examples:
    ```bash
    uv run pytest -q tests/unit/test_collections_workload.py
    ```

See Also:
    - [typevet.evaluation.collections_workload][]: workload and parity

[i236]: https://github.com/Alberto-Codes/typevet/issues/236
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from typevet.domain.judgment_questions import Choice, Noul
from typevet.evaluation.collections_workload import (
    BASELINE_ECE,
    ENGAGED,
    JEV_DIR_ENV,
    NOT_ENGAGED,
    PARITY_MAX_ECE,
    QUESTIONS_ENV,
    ece,
    load_questions,
    load_records,
    map_record,
    parity,
    reliability,
    workload_paths,
)


def _record(
    rid: str, *, engaged: bool, offer: str | None = None, **state: Any
) -> dict[str, Any]:
    return {
        "id": rid,
        "state": {"balance": 1200, "days_past_due": 45, **state},
        "logged": {
            "action_taken": {"action": "REMINDER_ONLY", "channel": "sms"},
            "outcome": {"engaged": engaged, "accepted_offer": offer},
            "compliance_issues": [],
        },
        "meta": {"scenario_id": "s-1"},
    }


def _seed() -> dict[str, Any]:
    return {
        "will_engage": {
            "type": "noul",
            "instructions": "Synthetic engage question?",
            "criteria": {"true": "Engages.", "false": "Does not engage."},
        },
        "accepted_offer": {
            "type": "choice",
            "instructions": "Synthetic offer question?",
            "criteria": {"NONE": "No offer.", "PLAN_A": "Plan A."},
        },
    }


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> Path:
    path.write_text(
        "".join(json.dumps(r) + "\n" for r in records) + "\n", encoding="utf-8"
    )
    return path


@pytest.mark.unit
def test_map_record_drops_id_meta_outcome_and_adds_proposed_action() -> None:
    raw = _record("r-1", engaged=True, offer="PLAN_A")
    before = json.dumps(raw, sort_keys=True)
    mapped = map_record(raw)
    assert mapped.state == {
        "balance": 1200,
        "days_past_due": 45,
        "proposed_action": {"action": "REMINDER_ONLY", "channel": "sms"},
    }
    wire = json.dumps(mapped.state)
    for hidden in ("r-1", "scenario_id", "engaged", "accepted_offer", "outcome"):
        assert hidden not in wire
    assert mapped.label == ENGAGED
    assert mapped.engaged is True
    assert mapped.accepted_offer == "PLAN_A"
    assert json.dumps(raw, sort_keys=True) == before


@pytest.mark.unit
def test_load_records_reads_labels_in_file_order(tmp_path: Path) -> None:
    path = _write_jsonl(
        tmp_path / "val.jsonl",
        [_record("a", engaged=False), _record("b", engaged=True, offer="PLAN_A")],
    )
    rows = load_records(path)
    assert [r.label for r in rows] == [NOT_ENGAGED, ENGAGED]
    assert [r.accepted_offer for r in rows] == ["NONE", "PLAN_A"]


@pytest.mark.unit
def test_load_questions_maps_seed_verbatim(tmp_path: Path) -> None:
    path = tmp_path / "seed.json"
    path.write_text(json.dumps(_seed()), encoding="utf-8")
    questions = load_questions(path)
    noul, choice = questions["will_engage"], questions["accepted_offer"]
    assert isinstance(noul, Noul)
    assert isinstance(choice, Choice)
    assert noul.instructions == "Synthetic engage question?"
    assert noul.criteria == {"true": "Engages.", "false": "Does not engage."}
    assert choice.instructions == "Synthetic offer question?"
    assert dict(choice.criteria) == {"NONE": "No offer.", "PLAN_A": "Plan A."}


def _without(key: str) -> dict[str, Any]:
    seed = _seed()
    del seed[key]
    return seed


def _swap_types() -> dict[str, Any]:
    seed = _seed()
    seed["will_engage"]["type"] = "choice"
    return seed


def _bad_field(key: str, field: str, value: Any) -> dict[str, Any]:
    seed = _seed()
    seed[key][field] = value
    return seed


@pytest.mark.unit
@pytest.mark.parametrize(
    "seed",
    [
        _without("will_engage"),
        _without("accepted_offer"),
        _swap_types(),
        _bad_field("accepted_offer", "type", "noul"),
        _bad_field("will_engage", "instructions", 3),
        _bad_field("accepted_offer", "criteria", ["NONE"]),
        ["not", "an", "object"],
        {"will_engage": "text", "accepted_offer": {}},
    ],
    ids=[
        "no-noul",
        "no-choice",
        "noul-typed-choice",
        "choice-typed-noul",
        "bad-instructions",
        "bad-criteria",
        "not-object",
        "entry-not-object",
    ],
)
def test_load_questions_rejects_bad_seed(tmp_path: Path, seed: Any) -> None:
    path = tmp_path / "seed.json"
    path.write_text(json.dumps(seed), encoding="utf-8")
    with pytest.raises(ValueError, match="seed"):
        load_questions(path)


@pytest.mark.unit
def test_workload_paths_reads_both_env_vars(tmp_path: Path) -> None:
    environ = {JEV_DIR_ENV: str(tmp_path), QUESTIONS_ENV: str(tmp_path / "q.json")}
    split, seed = workload_paths(environ, "test")
    assert split == tmp_path / "test.jsonl"
    assert seed == tmp_path / "q.json"


@pytest.mark.unit
@pytest.mark.parametrize("missing", [JEV_DIR_ENV, QUESTIONS_ENV])
def test_workload_paths_rejects_missing_env(missing: str) -> None:
    environ = {JEV_DIR_ENV: "/d", QUESTIONS_ENV: "/q.json"}
    environ[missing] = ""
    with pytest.raises(ValueError, match=missing):
        workload_paths(environ)


@pytest.mark.unit
def test_probability_one_falls_in_last_bin() -> None:
    bins = reliability([1.0, 0.0], [True, False])
    assert bins[9].count == 1
    assert bins[9].mean_prob == 1.0
    assert bins[0].count == 1
    assert sum(b.count for b in bins) == 2


@pytest.mark.unit
def test_ece_matches_hand_computed_value() -> None:
    value = ece([0.05, 0.15, 0.95, 1.0], [False, True, True, True])
    assert value == pytest.approx(0.2375, abs=1e-12)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("probs", "labels", "n_bins"),
    [([0.5], [True, False], 10), ([0.5], [True], 0), ([1.5], [True], 10)],
    ids=["length", "bins", "range"],
)
def test_ece_rejects_bad_input(
    probs: list[float], labels: list[bool], n_bins: int
) -> None:
    with pytest.raises(ValueError):
        ece(probs, labels, n_bins)


@pytest.mark.unit
def test_empty_ece_is_zero() -> None:
    assert ece([], []) == 0.0


@pytest.mark.unit
def test_parity_puts_failed_rows_in_no_bin() -> None:
    rows = [(0.05, False), (0.15, True), (None, True), (0.95, True), (1.0, True)]
    report = parity(rows)
    assert report["ece"] == pytest.approx(0.2375, abs=1e-12)
    assert report["scored"] == 4
    assert report["failures"] == 1
    assert sum(b["count"] for b in report["bins"]) == 4
    assert report["base_rate"] == pytest.approx(0.8)
    assert report["baseline_ece"] == BASELINE_ECE == 0.1423
    assert report["delta"] == pytest.approx(0.2375 - 0.1423)
    assert report["meets_parity"] is False


@pytest.mark.unit
def test_parity_passes_when_all_answered_within_tolerance() -> None:
    rows = [(0.1, False), (0.9, True), (0.2, False), (0.8, True)]
    report = parity(rows)
    assert report["failures"] == 0
    assert report["ece"] <= PARITY_MAX_ECE == pytest.approx(0.1723)
    assert report["max_ece"] == PARITY_MAX_ECE
    assert report["meets_parity"] is True
    assert report["bins"][1] == {
        "lower": 0.1,
        "upper": 0.2,
        "count": 1,
        "mean_prob": 0.1,
        "frac_engaged": 0.0,
    }


@pytest.mark.unit
def test_parity_fails_on_a_failed_row_even_when_calibrated() -> None:
    rows = [(0.1, False), (0.9, True), (0.2, False), (0.8, True), (None, True)]
    report = parity(rows)
    assert report["ece"] <= PARITY_MAX_ECE
    assert report["failures"] == 1
    assert report["meets_parity"] is False


@pytest.mark.unit
def test_parity_fails_above_tolerance_and_on_empty() -> None:
    assert parity([(0.9, False), (0.1, True)])["meets_parity"] is False
    empty = parity([])
    assert empty["meets_parity"] is False
    assert empty["base_rate"] == 0.0
