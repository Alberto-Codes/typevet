"""Real upstream scoring and the offline module command."""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
from jevbench.budget import Ledger
from jevbench.runner import Runner
from jevbench.tasks import Task
from judgevet import JevError, ScoreAnswer, SystemOneResponse, Usage

from typevet_evals.jevbench import SystemOneAdapter

pytest_plugins = ["pytester"]
pytestmark = pytest.mark.contract


class Port:
    """Return a rounded typed distribution or an HTTP failure."""

    def __init__(self, error: bool = False, rounded: bool = False) -> None:
        """Select the canned score or HTTP failure."""
        self.calls = 0
        self.error = error
        self.rounded = rounded

    def system_one(
        self,
        state: str | dict[str, Any] | list[Any],
        questions: Mapping[str, Any],
        model: str,
    ) -> SystemOneResponse:
        """Provide a score whose scalar cannot stand in for its class."""
        self.calls += 1
        if self.error:
            raise JevError("SECRET", 429)
        probabilities = {0: 0.4, 1: 0.1, 2: 0.49 if self.rounded else 0.5}
        return SystemOneResponse(
            "resolved",
            Usage(),
            {
                "decision": ScoreAnswer(
                    1.1, 0.5, {0: "low", 1: "mid", 2: "high"}, probabilities
                )
            },
        )


def tasks() -> list[Task]:
    """Build three independent records for stop-rule tests."""
    return [
        Task(
            str(index),
            "ordinal",
            "text",
            {
                "type": "score",
                "instructions": "Rate",
                "criteria": ["low", "mid", "high"],
            },
            ["0", "1", "2"],
            2,
            "public",
        )
        for index in range(3)
    ]


@pytest.mark.parametrize("rounded", [False, True])
def test_upstream_scoring(tmp_path: Path, rounded: bool) -> None:
    """Upstream owns argmax, ordinal EV and rounding tolerance."""
    runner = Runner(
        SystemOneAdapter(Port(rounded=rounded), model="requested"),
        Ledger(tmp_path / "ledger"),
        tmp_path / "raw",
    )
    row = runner.run_all(tasks()[:1])[0]
    assert row["predicted"] == "2"
    assert row["ordinal_ev"] == pytest.approx(
        (1 * 0.1 + 2 * (0.49 if rounded else 0.5)) / (0.99 if rounded else 1)
    )
    assert row["correct"] and row["valid"]
    assert row["strict_valid"] is (not rounded)
    assert row["renormalized"] is rounded
    assert row["probs_as_returned"]["2"] == (0.49 if rounded else 0.5)
    assert row["cost_usd"] is None
    assert row["usage"] == {"input_tokens": None, "output_tokens": None}
    assert row["charged_usd"] == row["reserved_usd"]


def test_upstream_stops_on_429(tmp_path: Path) -> None:
    """A preserved rate-limit status stops upstream after one request."""
    port = Port(error=True)
    rows = Runner(
        SystemOneAdapter(port, model="test"),
        Ledger(tmp_path / "ledger"),
        tmp_path / "raw",
    ).run_all(tasks())
    assert port.calls == len(rows) == 1
    assert rows[0]["status_code"] == 429
    assert "SECRET" not in json.dumps(rows)


@pytest.mark.parametrize("cap,code,count", [(1.0, 0, 2), (0.0, 1, 0)])
def test_offline_module_run(
    tmp_path: Path,
    cap: float,
    code: int,
    count: int,
    pytester: pytest.Pytester,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The public module writes results, raw evidence and a hashed manifest."""
    source = tmp_path / "tasks.jsonl"
    source.write_text("\n".join(item.to_json() for item in tasks()) + "\n")
    monkeypatch.setenv("TYPEVET_BACKEND", "fake")
    result = pytester.run(
        sys.executable,
        "-m",
        "typevet_evals.jevbench_run",
        "--tasks",
        str(source),
        "--model",
        "requested",
        "--results",
        str(tmp_path / "results.jsonl"),
        "--raw-dir",
        str(tmp_path / "raw"),
        "--ledger",
        str(tmp_path / "ledger"),
        "--manifest",
        str(tmp_path / "manifest.json"),
        "--limit",
        "2",
        "--cap-usd",
        str(cap),
    )
    assert result.ret == code, result.stderr.str() + result.stdout.str()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["upstream_commit"] == "bb05a335bc809e61b20c0f745d25499a82b326fc"
    assert manifest["requested_count"] == 2
    assert manifest["attempted_count"] == count
    assert len(manifest["dataset_sha256"]) == 64
    assert manifest["requested_model"] == "requested"
    rows = [
        json.loads(line)
        for line in (tmp_path / "results.jsonl").read_text().splitlines()
    ]
    assert len(rows) == count
    if count:
        assert manifest["resolved_models"] == ["requested"]
    for row in rows:
        assert row["cost_usd"] is None
        assert row["model"] == "requested"
    raw_files = list((tmp_path / "raw").glob("*.json"))
    assert len(raw_files) == count
    for raw_file in raw_files:
        raw = json.loads(raw_file.read_text())
        assert set(raw["request"]) == {"state", "model", "questions"}
        assert "expected" not in raw["request"]["questions"]["decision"]
