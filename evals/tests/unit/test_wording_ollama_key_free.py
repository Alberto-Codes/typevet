"""Unit check that the Ollama placeholder key stays out of the written JSON (#379).

The Ollama judge passes ``OLLAMA_PLACEHOLDER_KEY`` to the adapter. This check
opens ``_ollama_judge`` with scripted Ollama JSON and a scripted adapter that
answers each call, so no call leaves the process. It builds the judge parts of
the evolution artifact and the held-out and comparison receipts from the judge
facts, and reads each JSON for the key.

Examples:
    ```bash
    uv run pytest -q evals/tests/unit/test_wording_ollama_key_free.py
    ```

See Also:
    - [typevet_evals.wording.comparison][]: the comparison receipt
    - [typevet_evals.wording.held_out][]: the held-out receipt
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace
from types import TracebackType
from typing import Any, Self

import pytest
from judgevet import NoulAnswer, SystemOneResponse, Usage

from evals.tests.live import test_wording_evolution_live as live
from typevet_evals.datasets.difraud import DIFrauDRecord, map_row, record_id
from typevet_evals.wording.calls import TimedJudgePort, call_summary
from typevet_evals.wording.comparison import ComparisonSubject, comparison_receipt
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
RESPONSES = {
    f"{BASE}/api/version": {"version": "0.35.0"},
    f"{BASE}/api/tags": {"models": [{"name": "nimble:latest", "digest": DIGEST}]},
}
ARTIFACT = {
    "seed_text": live.SEED_TEXT,
    "evolved_text": "Does the message ask for money, a code or a gift card?",
    "judge_provider": "ollama",
    "valid": True,
    "budget_refusals": 0,
}


class _NimbleAdapter:
    """Stand in for ``HTTPSystemOneAdapter``; answer 0.5 as ``nimble:latest``."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        trace: TracebackType | None,
    ) -> None:
        return None

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


def _rows() -> HeldOutRows:
    records = tuple(
        DIFrauDRecord(record_id(text), replace(map_row(text, int(scam)), split="test"))
        for text, scam in (("Send gift cards", True), ("Dinner at six", False))
    )
    return HeldOutRows(records, frozenset())


def _scored(port: Any, parts: WordingParts, seed: Any, model: str) -> HeldOutRun:
    timed = TimedJudgePort(port)
    run = score_held_out(
        timed,
        seed,
        live.PRIMARY_NOUL_NAME,
        evolved=parts,
        rows=_rows(),
        judge_model=model,
        failures=(),
    )
    return replace(run, call_records=timed.records)


def _pins(model: str, facts: Mapping[str, object], run: HeldOutRun) -> dict[str, Any]:
    return {
        "served_template": None,
        "server": dict(facts),
        "reported_models": call_summary(run.call_records)["models"],
        **live.judge_pins("ollama", model, facts, run.call_records),
    }


def test_the_placeholder_key_stays_out_of_the_artifact_and_receipts() -> None:
    with live._ollama_judge(
        {}, adapter=_NimbleAdapter, get=lambda url: RESPONSES[url]
    ) as judge:
        assert isinstance(judge.port, _NimbleAdapter)
        assert judge.port.kwargs["api_key"] == live.OLLAMA_PLACEHOLDER_KEY
        model, facts = judge.model, dict(judge.facts)
        seed, parts = live.held_out_parts(ARTIFACT, {})
        held_run = _scored(judge.port, parts, seed, model)
        noul = live.wording_seed({})
        compared = live.comparison_parts(ARTIFACT, noul, backend="ollama")
        comparison_run = _scored(judge.port, compared, noul, model)
    artifact = {
        "served_template": facts["served_template"],
        "judge_provider": "ollama",
        "judge_identity": live.judge_identity(
            model, facts, call_summary(held_run.call_records)["models"]
        ),
    }
    held = held_out_receipt(
        held_run,
        seed_text=parts.seed_text,
        evolved_text=parts.evolved_text,
        backend="ollama",
        model=model,
        pins=_pins(model, facts, held_run),
        identity={},
    )
    comparison = comparison_receipt(
        comparison_run,
        ComparisonSubject(live._COMPARISON_JUDGE["ollama"], "ollama", model, "test"),
        seed_text=compared.seed_text,
        evolved_text=compared.evolved_text,
        pins=_pins(model, facts, comparison_run),
        identity={},
    )
    assert held["pins"]["judge_identity"] == artifact["judge_identity"]
    for name, written in (
        ("artifact", artifact),
        ("held_out", held),
        ("comparison", comparison),
    ):
        text = json.dumps(written, indent=2)
        assert live.OLLAMA_PLACEHOLDER_KEY not in text, f"key in the {name} JSON"
