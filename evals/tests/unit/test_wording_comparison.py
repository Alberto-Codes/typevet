"""Unit checks for the #252 Jev vs Gemma comparison receipt (#329).

Examples:
    ```bash
    uv run pytest -q evals/tests/unit/test_wording_comparison.py
    ```

See Also:
    - [typevet_evals.wording.comparison][]: the comparison module
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from judgevet import NoulAnswer, SystemOneResponse, Usage
from judgevet.domain.questions import Noul

from typevet_evals.datasets.difraud import DIFrauDRecord, map_row, record_id
from typevet_evals.wording.calls import CallRecord, call_summary
from typevet_evals.wording.comparison import (
    COMPARISON_BACKENDS,
    ComparisonSubject,
    ValidationRows,
    comparison_receipt,
    evolved_text_for,
)
from typevet_evals.wording.held_out import (
    HeldOutRows,
    HeldOutRun,
    ScoredPair,
    score_held_out,
)
from typevet_evals.wording.metrics import cohen_kappa

pytestmark = pytest.mark.unit

_RECEIPTS = Path(__file__).resolve().parents[2] / "fixtures" / "difraud" / "receipts"
SEED_TEXT = "Is this message a scam?"


def _record(text: str, scam: bool, split: str) -> DIFrauDRecord:
    """Return one record whose example names ``split``."""
    return DIFrauDRecord(
        record_id(text), replace(map_row(text, int(scam)), split=split)
    )


def _artifact(provider: str, **changes: Any) -> dict[str, Any]:
    """Return a minimal evolution artifact of ``provider``."""
    artifact: dict[str, Any] = {
        "seed_text": SEED_TEXT,
        "evolved_text": f"evolved by {provider}",
        "judge_provider": provider,
        "valid": True,
        "budget_refusals": 0,
    }
    artifact.update(changes)
    return artifact


def _committed(name: str) -> dict[str, Any]:
    return json.loads((_RECEIPTS / name).read_text(encoding="utf-8"))


def test_each_backend_takes_its_own_judges_committed_evolved_text() -> None:
    jev = _committed("wording252_evolution_jev.json")
    gemma = _committed("wording252_evolution_gemma.json")
    seed = jev["seed_text"]

    assert evolved_text_for(jev, backend="jev", seed_text=seed) == jev["evolved_text"]
    for backend in ("llama_cpp", "vllm"):
        text = evolved_text_for(gemma, backend=backend, seed_text=seed)
        assert text == gemma["evolved_text"]
    assert jev["evolved_text"] != gemma["evolved_text"]


@pytest.mark.parametrize(
    ("backend", "provider"),
    [("jev", "gemma"), ("llama_cpp", "jev"), ("vllm", "jev")],
)
def test_a_backend_refuses_another_judges_artifact(backend: str, provider: str) -> None:
    with pytest.raises(ValueError, match="judge_provider"):
        evolved_text_for(_artifact(provider), backend=backend, seed_text=SEED_TEXT)


def test_an_unknown_backend_is_refused() -> None:
    assert COMPARISON_BACKENDS == ("jev", "llama_cpp", "vllm")
    with pytest.raises(ValueError, match="backend"):
        evolved_text_for(_artifact("gemma"), backend="ollama", seed_text=SEED_TEXT)


@pytest.mark.parametrize(
    ("changes", "match"),
    [
        ({"valid": False}, "valid"),
        ({"budget_refusals": 3}, "budget_refusals"),
        ({"seed_text": "Another seed"}, "seed"),
        ({"evolved_text": SEED_TEXT}, "same as the seed"),
    ],
)
def test_an_unusable_artifact_is_refused(changes: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        evolved_text_for(
            _artifact("jev", **changes), backend="jev", seed_text=SEED_TEXT
        )


def test_validation_rows_refuse_a_held_out_row() -> None:
    rows = (_record("Hi", False, "validation"), _record("Win", True, "test"))

    with pytest.raises(ValueError, match="'test'"):
        ValidationRows(rows)


class _Port:
    """Answer 0.7 to every question."""

    def system_one(self, state: str, questions: Any, model: str) -> Any:
        """Return one Noul answer.

        Returns:
            A response with ``is_scam`` at 0.7.
        """
        del state, questions
        return SystemOneResponse(
            model=model, usage=Usage(), answers={"is_scam": NoulAnswer(noul=0.7)}
        )


def _scored(rows: HeldOutRows | ValidationRows) -> HeldOutRun:
    return score_held_out(
        _Port(),
        Noul(instructions=SEED_TEXT),
        "is_scam",
        evolved_text="other",
        rows=rows,
        judge_model="m",
        failures=(RuntimeError,),
    )


_VALIDATION = ValidationRows((_record("Win", True, "validation"),))
_HELD_OUT = HeldOutRows((_record("Win", True, "test"),), frozenset())


def test_validation_rows_score_through_the_held_out_loop() -> None:
    run = _scored(_VALIDATION)

    assert run.calls == 2
    assert run.pairs[0].label == 1


# Labels 1, 0, 1, 0. Seed reads 1, 1, 0, 0 (kappa 0); evolved reads 1, 0, 1, 0
# (kappa 1).
_PAIRS = (
    ScoredPair("a", 1, 0.6, 0.9),
    ScoredPair("b", 0, 0.6, 0.1),
    ScoredPair("c", 1, 0.4, 0.9),
    ScoredPair("d", 0, 0.4, 0.1),
)


def _receipt(run: HeldOutRun, split: str = "test") -> dict[str, Any]:
    return comparison_receipt(
        run,
        ComparisonSubject("gemma_llama_cpp", "llama_cpp", "gemma", split),
        seed_text=SEED_TEXT,
        evolved_text="evolved",
        pins={"served_template": "native_gemma4_turn"},
        identity={"run_id": "x"},
    )


def test_receipt_holds_kappa_per_arm_and_no_pass_verdict() -> None:
    records = (CallRecord(0, "a", ("is_scam",), 0.5, 11, None, "gemma"),)
    run = HeldOutRun(_PAIRS, 4, 4, None, records, split="test")

    receipt = _receipt(run)

    assert receipt["issue"] == 329
    assert receipt["parent_issue"] == 252
    assert receipt["pass_rule"] is None
    assert "verdict" not in receipt
    assert receipt["judge"] == "gemma_llama_cpp"
    assert receipt["split"] == "test"
    assert receipt["positive_threshold"] == 0.5
    assert receipt["ece_bins"] == 10
    assert receipt["metrics"]["seed"]["kappa"] == pytest.approx(0.0)
    assert receipt["metrics"]["evolved"]["kappa"] == pytest.approx(1.0)
    assert receipt["metrics"]["seed"]["accuracy"] == pytest.approx(0.5)
    assert receipt["metrics"]["evolved"]["brier"] == pytest.approx(0.01)
    assert receipt["bootstrap"]["resamples"] == 2000
    assert receipt["per_call"] == [r.to_mapping() for r in records]
    assert receipt["call_summary"] == call_summary(records)
    assert receipt["calls"] == {"seed": 4, "evolved": 4, "total": 8}
    assert receipt["pins"] == {"served_template": "native_gemma4_turn"}
    assert receipt["identity"] == {"run_id": "x"}
    assert len(receipt["pairs"]) == 4
    json.dumps(receipt)


def test_receipt_pins_the_seed_and_evolved_wording_by_digest() -> None:
    receipt = _receipt(HeldOutRun(_PAIRS, 4, 4, None, split="test"))

    for arm, text in (("seed", SEED_TEXT), ("evolved", "evolved")):
        canonical = json.dumps({"instructions": text}, separators=(",", ":"))
        loose = json.dumps({"instructions": text}, ensure_ascii=False)
        assert receipt["wording_digests"][arm] == {
            "components": {"instructions": hashlib.sha256(text.encode()).hexdigest()},
            "mapping": hashlib.sha256(canonical.encode()).hexdigest(),
            "gepa_candidate_id": hashlib.sha256(loose.encode()).hexdigest()[:12],
        }
    assert receipt["seed_text"] == SEED_TEXT
    assert receipt["evolved_text"] == "evolved"


def test_receipt_kappa_equals_the_metric_function() -> None:
    receipt = _receipt(HeldOutRun(_PAIRS, 4, 4, None, split="test"))
    labels = [p.label for p in _PAIRS]

    assert receipt["metrics"]["seed"]["kappa"] == cohen_kappa(
        [p.seed_probability for p in _PAIRS], labels
    )


def test_receipt_of_a_stopped_run_has_no_metrics() -> None:
    receipt = _receipt(HeldOutRun(_PAIRS[:1], 2, 1, "RuntimeError: down", split="test"))

    assert receipt["metrics"] is None
    assert receipt["bootstrap"] is None
    assert receipt["stopped"] == "RuntimeError: down"


def test_subject_refuses_an_unknown_split_or_backend() -> None:
    with pytest.raises(ValueError, match="split"):
        ComparisonSubject("gemma_llama_cpp", "llama_cpp", "gemma", "train")
    with pytest.raises(ValueError, match="backend"):
        ComparisonSubject("gemma", "ollama", "gemma", "test")


@pytest.mark.parametrize(
    ("rows", "split"), [(_HELD_OUT, "test"), (_VALIDATION, "validation")]
)
def test_receipt_split_comes_from_the_row_type(
    rows: HeldOutRows | ValidationRows, split: str
) -> None:
    assert _receipt(_scored(rows), split)["split"] == split


@pytest.mark.parametrize(
    ("rows", "claimed", "actual"),
    [(_VALIDATION, "test", "validation"), (_HELD_OUT, "validation", "test")],
)
def test_receipt_refuses_a_subject_split_that_disagrees_with_the_rows(
    rows: HeldOutRows | ValidationRows, claimed: str, actual: str
) -> None:
    run = _scored(rows)

    with pytest.raises(ValueError, match="split") as caught:
        _receipt(run, claimed)
    assert repr(claimed) in str(caught.value)
    assert repr(actual) in str(caught.value)


def test_a_validation_smoke_receipt_names_its_split() -> None:
    run = HeldOutRun(_PAIRS, 4, 4, None, split="validation")

    assert _receipt(run, "validation")["split"] == "validation"


def test_a_run_refuses_an_unknown_split() -> None:
    with pytest.raises(ValueError, match="'train'"):
        HeldOutRun(_PAIRS, 4, 4, None, split="train")
