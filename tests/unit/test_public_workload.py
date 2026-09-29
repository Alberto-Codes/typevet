"""Unit tests: public-dataset workloads and the runner baseline ([#236][i236]).

Banking77 and DIFrauD rows come from the vendored CI subsets under
``tests/fixtures``. The download test uses ``httpx.MockTransport``.

Examples:
    ```bash
    uv run pytest -q tests/unit/test_public_workload.py
    ```

See Also:
    - [typevet.evaluation.public_workload][]: the workload builders

[i236]: https://github.com/Alberto-Codes/typevet/issues/236
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest

from tests.unit.test_collections_throughput import FakeVllm, env, records
from typevet.domain.judgment_questions import Choice, Noul
from typevet.evaluation import public_workload
from typevet.evaluation.collections_throughput import run_throughput
from typevet.evaluation.collections_workload import (
    BASELINE_BASE_RATE,
    BASELINE_ECE,
    COLLECTIONS_BASELINE,
    PARITY_MAX_ECE,
    Baseline,
    CollectionsRecord,
    parity,
)
from typevet.evaluation.datasets import banking77, difraud
from typevet.evaluation.public_workload import (
    BANKING77_BASELINE,
    BANKING77_FILE,
    DIFRAUD_BASELINE,
    DIFRAUD_FILE,
    MAX_CHOICE_OPTIONS,
    PublicRecord,
    banking77_workload,
    difraud_workload,
    fetch_public_data,
    load_public_workloads,
    missing_data,
    validate_questions,
)

pytestmark = pytest.mark.unit

_FIX = Path(__file__).resolve().parents[1] / "fixtures"
_CSV = (_FIX / "banking77" / "test_subset.csv").read_text(encoding="utf-8")
_JSONL = (_FIX / "difraud" / "sms_test_subset.jsonl").read_text(encoding="utf-8")


def _data_dir(tmp_path: Path) -> Path:
    (tmp_path / BANKING77_FILE).write_text(_CSV, encoding="utf-8")
    (tmp_path / DIFRAUD_FILE).write_text(_JSONL, encoding="utf-8")
    return tmp_path


def test_banking77_maps_text_state_fraud_positive_and_finvet_questions() -> None:
    rows = banking77.load_test_split(csv_text=_CSV)
    work = banking77_workload(rows)
    assert [r.state for r in work.records] == [r.text for r in rows]
    assert [r.positive for r in work.records] == [
        r.proxy_label == "fraud" for r in rows
    ]
    assert any(r.positive for r in work.records)
    assert not all(r.positive for r in work.records)
    assert work.noul == "reports_unauthorized"
    assert work.baseline == BANKING77_BASELINE == Baseline(0.17, 0.5, 0.20)
    noul = work.questions["reports_unauthorized"]
    assert isinstance(noul, Noul)
    assert noul.instructions == (
        "Does the customer report a transaction they did not authorize?"
    )
    choice = work.questions["fraud_type"]
    assert isinstance(choice, Choice)
    assert choice.instructions == (
        "Which kind of fraud, if any, does the customer describe?"
    )
    assert list(choice.criteria) == [
        "unauthorized_transaction",
        "duplicate_charge",
        "phishing_or_scam",
        "account_takeover",
        "not_fraud",
        "unclear",
    ]
    assert choice.criteria["unclear"] == "The message does not say enough to decide."


def test_difraud_maps_scam_positive_and_only_the_is_scam_noul() -> None:
    rows = difraud.load_test_split(jsonl_text=_JSONL)
    work = difraud_workload(rows)
    assert [r.state for r in work.records] == [r.text for r in rows]
    assert [r.positive for r in work.records] == [r.label == "scam" for r in rows]
    assert set(work.questions) == {"is_scam"}
    noul = work.questions["is_scam"]
    assert isinstance(noul, Noul)
    assert noul.instructions == (
        "Is this message a scam, phishing or social-engineering attempt?"
    )
    assert work.noul == "is_scam"
    assert work.baseline == DIFRAUD_BASELINE
    assert (DIFRAUD_BASELINE.ece, DIFRAUD_BASELINE.max_ece) == (0.07, 0.10)


def test_collections_record_positive_equals_engaged() -> None:
    yes = CollectionsRecord(state={}, label="engaged", accepted_offer="NONE")
    no = CollectionsRecord(state={}, label="not_engaged", accepted_offer="NONE")
    assert (yes.positive, no.positive) == (True, False)
    assert PublicRecord(state="t", positive=True).positive is True


def test_choice_over_ten_options_raises_and_ten_passes() -> None:
    def choice(n: int) -> dict[str, Noul | Choice]:
        return {"c": Choice(criteria={f"o{i}": None for i in range(n)})}

    assert MAX_CHOICE_OPTIONS == 10
    validate_questions(choice(10))
    with pytest.raises(ValueError, match="11 options"):
        validate_questions(choice(11))


def test_collections_parity_default_is_unchanged() -> None:
    rows = [(0.9, True), (0.1, False)]
    assert parity(rows) == parity(rows, COLLECTIONS_BASELINE)
    expected = Baseline(BASELINE_ECE, BASELINE_BASE_RATE, PARITY_MAX_ECE)
    assert expected == COLLECTIONS_BASELINE
    report = parity(rows)
    assert report["baseline_ece"] == 0.1423
    assert report["max_ece"] == PARITY_MAX_ECE
    assert report["meets_parity"] is True


def test_baseline_none_records_only() -> None:
    report = parity([(0.9, True), (0.1, False)], None)
    assert report["meets_parity"] is None
    assert report["baseline_ece"] is None
    assert report["baseline_base_rate"] is None
    assert report["max_ece"] is None
    assert report["delta"] is None
    assert report["ece"] == pytest.approx(0.1)
    assert report["scored"] == 2


def test_parity_uses_the_supplied_baseline() -> None:
    rows = [(0.9, True), (0.1, False)]
    strict = parity(rows, Baseline(ece=0.05, base_rate=0.5, max_ece=0.05))
    assert strict["meets_parity"] is False
    assert strict["delta"] == pytest.approx(0.05)


def test_runner_default_keeps_the_collections_noul_and_baseline() -> None:
    server = FakeVllm()
    receipt = run_throughput(
        env(), records(4), _collections(), transport=httpx.MockTransport(server)
    )
    level = receipt["levels"][0]
    assert level["answered"] == 4
    assert level["parity"]["baseline_ece"] == BASELINE_ECE


def test_runner_baseline_none_records_only() -> None:
    work = difraud_workload(difraud.load_test_split(jsonl_text=_JSONL))
    receipt = run_throughput(
        env(),
        work.records,
        work.questions,
        transport=httpx.MockTransport(FakeVllm()),
        levels=(1,),
        noul=work.noul,
        baseline=None,
    )
    level = receipt["levels"][0]
    assert level["answered"] == len(work.records)
    assert level["parity"]["meets_parity"] is None
    assert level["parity"]["base_rate"] == pytest.approx(
        sum(r.positive for r in work.records) / len(work.records)
    )


def test_missing_data_names_each_absent_file(tmp_path: Path) -> None:
    reason = missing_data(tmp_path)
    assert reason is not None
    assert BANKING77_FILE in reason and DIFRAUD_FILE in reason
    assert "fetch_public_data" in reason
    assert missing_data(_data_dir(tmp_path)) is None


def test_load_public_workloads_builds_the_three_sets(tmp_path: Path) -> None:
    sets = load_public_workloads(_data_dir(tmp_path))
    assert set(sets) == {"banking77_balanced", "difraud_sms", "banking77_full"}
    balanced = sets["banking77_balanced"].records
    full = sets["banking77_full"].records
    assert len(full) == 8
    assert sum(r.positive for r in balanced) == sum(r.positive for r in full) == 6
    assert sorted(r.state for r in balanced) == sorted(r.state for r in full)
    assert sets["banking77_balanced"].baseline == BANKING77_BASELINE
    assert sets["banking77_full"].baseline is None
    assert len(sets["difraud_sms"].records) == 4


def test_fetch_public_data_writes_both_files(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = _CSV if request.url.path.endswith(".csv") else _JSONL
        return httpx.Response(200, text=body)

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        sizes = fetch_public_data(tmp_path / "public", client=client)
    assert sizes == {
        BANKING77_FILE: len(_CSV.encode()),
        DIFRAUD_FILE: len(_JSONL.encode()),
    }
    assert missing_data(tmp_path / "public") is None


def _collections() -> dict[str, Noul | Choice]:
    return {
        "will_engage": Noul(instructions="Will the customer engage?"),
        "accepted_offer": Choice(criteria={"NONE": "None.", "PLAN_A": "Plan A."}),
    }


def test_banking77_workload_rejects_an_eleven_option_choice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wide = {"fraud_type": Choice(criteria={f"o{i}": None for i in range(11)})}
    monkeypatch.setattr(public_workload, "_BANKING77_QUESTIONS", wide)
    rows = banking77.load_test_split(csv_text=_CSV)
    with pytest.raises(ValueError, match="11 options"):
        banking77_workload(rows)


def test_runner_rejects_an_unknown_scoring_keyword() -> None:
    server = FakeVllm()
    extra: dict[str, Any] = {"baselne": None}
    with pytest.raises(TypeError, match="baselne"):
        run_throughput(
            env(),
            records(1),
            _collections(),
            transport=httpx.MockTransport(server),
            **extra,
        )
    assert server.requests == []
