"""Unit checks for the held-out and comparison receipts of an Ollama judge (#333).

A scripted port reports ``nimble:latest`` for each call, so no call leaves the
process. The checks build the receipts the live tests write and read their
judge identity.

Examples:
    ```bash
    uv run pytest -q evals/tests/unit/test_wording_ollama_receipts.py
    ```

See Also:
    - [typevet_evals.wording.comparison][]: the comparison backends
    - [typevet_evals.wording.held_out][]: the held-out receipt
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

import pytest
from judgevet import NoulAnswer, SystemOneResponse, Usage

from evals.tests.live import test_wording_evolution_live as live
from typevet_evals.datasets.difraud import DIFrauDRecord, map_row, record_id
from typevet_evals.wording.calls import TimedJudgePort
from typevet_evals.wording.comparison import (
    COMPARISON_BACKENDS,
    EVOLUTION_PROVIDER,
    ComparisonSubject,
    comparison_receipt,
)
from typevet_evals.wording.held_out import (
    HeldOutRows,
    HeldOutRun,
    held_out_receipt,
    score_held_out,
)
from typevet_evals.wording.parts import WordingParts

pytestmark = pytest.mark.unit

BASE = "http://localhost:11434"
DIGEST = "24e550a1" + "0" * 51 + "67e0c"
FACTS = {
    "base_url": BASE,
    "served_template": None,
    "ollama_version": "0.35.0",
    "model_digest": DIGEST,
}
IDENTITY = {
    "requested_model": "nimble",
    "reported_models": ["nimble:latest"],
    "base_url": BASE,
    "ollama_version": "0.35.0",
    "model_digest": DIGEST,
}
EVOLVED_TEXT = "Does the message ask for money, a code or a gift card?"


def _rows() -> HeldOutRows:
    records = tuple(
        DIFrauDRecord(record_id(text), replace(map_row(text, int(scam)), split="test"))
        for text, scam in (("Send gift cards", True), ("Dinner at six", False))
    )
    return HeldOutRows(records, frozenset())


def _artifact(provider: str = "ollama") -> dict[str, Any]:
    return {
        "seed_text": live.SEED_TEXT,
        "evolved_text": EVOLVED_TEXT,
        "judge_provider": provider,
        "valid": True,
        "budget_refusals": 0,
    }


class _NimblePort:
    """Answer 0.5 to every question and report ``nimble:latest``."""

    def system_one(
        self, state: str, questions: Mapping[str, Any], model: str
    ) -> SystemOneResponse:
        """Answer 0.5 under the primary question name.

        Returns:
            One ``Noul`` answer, reported by ``nimble:latest``.
        """
        return SystemOneResponse(
            model="nimble:latest",
            usage=Usage(),
            answers={live.PRIMARY_NOUL_NAME: NoulAnswer(noul=0.5)},
        )


def _scored(parts: WordingParts, seed: Any) -> HeldOutRun:
    timed = TimedJudgePort(_NimblePort())
    run = score_held_out(
        timed,
        seed,
        live.PRIMARY_NOUL_NAME,
        evolved=parts,
        rows=_rows(),
        judge_model="nimble",
        failures=(),
    )
    return replace(run, call_records=timed.records)


def _pins(run: HeldOutRun) -> dict[str, object]:
    return {
        "server": FACTS,
        **live.judge_pins("ollama", "nimble", FACTS, run.call_records),
    }


def test_ollama_is_a_comparison_backend_of_its_own_artifacts() -> None:
    assert COMPARISON_BACKENDS == ("jev", "llama_cpp", "vllm", "ollama")
    assert EVOLUTION_PROVIDER["ollama"] == "ollama"
    assert live._COMPARISON_JUDGE["ollama"] == "ollama"


def test_judge_identity_has_the_evolution_artifact_shape() -> None:
    assert live.judge_identity("nimble", FACTS, ["nimble:latest"]) == IDENTITY
    jev_facts = {"base_url": "u", "served_template": None}
    assert live.judge_identity("jev-latest", jev_facts, []) == {
        "requested_model": "jev-latest",
        "reported_models": [],
        "base_url": "u",
    }


@pytest.mark.parametrize("backend", ["jev", "llama_cpp", "vllm"])
def test_judge_pins_add_nothing_for_the_other_backends(backend: str) -> None:
    assert live.judge_pins(backend, "m", {"build_info": "b"}, ()) == {}


def test_the_runtime_build_of_an_ollama_judge_is_its_version() -> None:
    assert live.runtime_build(FACTS) == "0.35.0"
    assert live.runtime_build({"build_info": "b1"}) == "b1"
    assert live.runtime_build({"base_url": "u", "served_template": None}) == "unknown"


def test_a_held_out_receipt_of_an_ollama_artifact_records_the_identity() -> None:
    seed, parts = live.held_out_parts(_artifact(), {})
    run = _scored(parts, seed)
    receipt = held_out_receipt(
        run,
        seed_text=parts.seed_text,
        evolved_text=parts.evolved_text,
        backend="ollama",
        model="nimble",
        pins=_pins(run),
        identity={},
    )
    assert (receipt["backend"], receipt["model"]) == ("ollama", "nimble")
    assert receipt["evolved_text"] == EVOLVED_TEXT
    assert receipt["rows"] == 2
    assert receipt["pins"]["judge_identity"] == IDENTITY
    assert receipt["pins"]["server"]["model_digest"] == DIGEST


def test_a_comparison_receipt_accepts_an_ollama_artifact() -> None:
    seed = live.wording_seed({})
    parts = live.comparison_parts(_artifact(), seed, backend="ollama")
    run = _scored(parts, seed)
    judge = live._COMPARISON_JUDGE["ollama"]
    receipt = comparison_receipt(
        run,
        ComparisonSubject(judge, "ollama", "nimble", "test"),
        seed_text=parts.seed_text,
        evolved_text=parts.evolved_text,
        pins=_pins(run),
        identity={},
    )
    assert (receipt["judge"], receipt["backend"]) == ("ollama", "ollama")
    assert receipt["evolved_text"] == EVOLVED_TEXT
    assert receipt["pins"]["judge_identity"] == IDENTITY
    assert receipt["call_summary"]["models"] == ["nimble:latest"]


@pytest.mark.parametrize("provider", ["gemma", "jev"])
def test_the_ollama_comparison_refuses_another_judges_artifact(provider: str) -> None:
    seed = live.wording_seed({})
    with pytest.raises(ValueError, match="judge_provider"):
        live.comparison_parts(_artifact(provider), seed, backend="ollama")
