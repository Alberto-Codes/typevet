"""Unit checks for the held-out scoring, subsets and receipts (#309).

A fake port answers from the wording and the message, so the scoring loop,
the receipt and the evolution artifact run offline.

Examples:
    ```bash
    uv run pytest -q evals/tests/unit/test_wording_held_out.py
    ```

See Also:
    - [typevet_evals.wording.held_out][]: the held-out module
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

import pytest
from gepa_adk import EvolutionResult
from judgevet import NoulAnswer, SystemOneResponse, Usage
from judgevet.domain.questions import Noul

from typevet_evals.datasets.difraud import DIFrauDRecord, map_row, record_id
from typevet_evals.wording import WordingRun, WordingRunConfig
from typevet_evals.wording.calls import CallRecord, call_summary
from typevet_evals.wording.digests import wording_digests
from typevet_evals.wording.held_out import (
    DEFAULT_TRAIN_ROWS,
    REFERENCE_ECE,
    HeldOutRows,
    ValidationRows,
    evolution_artifact,
    held_out_receipt,
    score_held_out,
    stratified_subset,
    train_subset,
)
from typevet_evals.wording.metrics import pass_verdict, wording_metrics
from typevet_evals.wording.parts import WordingParts, seed_mapping

pytestmark = pytest.mark.unit

KEY = "is_scam"
SEED_TEXT = "Is this message a scam?"
EVOLVED_TEXT = "Does it ask for money?"
CRITERIA = {"true": "It is a scam", "false": "It is legitimate"}
SEED = Noul(instructions=SEED_TEXT, criteria=CRITERIA)
TEXT_PARTS = WordingParts.instructions_only(SEED, EVOLVED_TEXT)


def _record(text: str, scam: bool, split: str = "test") -> DIFrauDRecord:
    """Return one record whose example names ``split``."""
    return DIFrauDRecord(
        record_id(text), replace(map_row(text, int(scam)), split=split)
    )


HELD_OUT = (
    _record("Win cash now", True),
    _record("See you at six", False),
    _record("Claim your prize", True),
    _record("Lunch at noon", False),
)


@dataclass
class AnsweringPort:
    """Answer 0.6/0.4 for the seed wording and 0.9/0.1 for any other wording.

    Attributes:
        fail_at (int | None): The call number that raises, or None.
        calls (list[tuple[str, str, Any, str]]): State, wording, criteria, model.
    """

    fail_at: int | None = None
    calls: list[tuple[str, str, Any, str]] = field(default_factory=list)

    def system_one(
        self, state: str, questions: Mapping[str, Any], model: str
    ) -> SystemOneResponse:
        """Record the call and answer, or raise at ``fail_at``.

        Returns:
            One Noul answer for ``KEY``.

        Raises:
            RuntimeError: At call number ``fail_at``.
        """
        noul = questions[KEY]
        self.calls.append((state, str(noul.instructions), noul.criteria, model))
        if self.fail_at is not None and len(self.calls) == self.fail_at:
            raise RuntimeError("backend down")
        scam = state in {"Win cash now", "Claim your prize"}
        high, low = (0.6, 0.4) if noul.instructions == SEED_TEXT else (0.9, 0.1)
        return SystemOneResponse(
            model=model,
            usage=Usage(),
            answers={KEY: NoulAnswer(noul=high if scam else low)},
        )


def _records(scam: int, legit: int, split: str) -> list[DIFrauDRecord]:
    """Return ``scam`` scam rows and ``legit`` legit rows of ``split``."""
    return [_record(f"scam {i}", True, split) for i in range(scam)] + [
        _record(f"legit {i}", False, split) for i in range(legit)
    ]


def test_stratified_subset_keeps_the_label_shares() -> None:
    records = _records(127, 530, "validation")

    subset = stratified_subset(records, 200, seed=0)

    # 200 * 127 / 657 = 38.66 and 200 * 530 / 657 = 161.34; the one row left
    # goes to the larger remainder.
    assert Counter(r.example.label for r in subset) == {"scam": 39, "legit": 161}
    assert len({r.record_id for r in subset}) == 200
    assert set(subset) <= set(records)


def test_stratified_subset_is_fixed_by_the_seed_not_the_input_order() -> None:
    records = _records(20, 80, "validation")

    first = stratified_subset(records, 10, seed=0)

    assert stratified_subset(list(reversed(records)), 10, seed=0) == first
    assert stratified_subset(records, 10, seed=1) != first


@pytest.mark.parametrize("size", [0, 101])
def test_stratified_subset_refuses_a_bad_size(size: int) -> None:
    with pytest.raises(ValueError, match="size"):
        stratified_subset(_records(20, 80, "validation"), size)


def test_score_held_out_asks_both_wordings_per_row() -> None:
    port = AnsweringPort()

    run = score_held_out(
        port,
        SEED,
        KEY,
        evolved=TEXT_PARTS,
        rows=HeldOutRows(HELD_OUT, frozenset()),
        judge_model="judge",
        failures=(RuntimeError,),
    )

    assert run.stopped is None
    assert run.calls == 8
    assert [(s, w) for s, w, _, _ in port.calls[:2]] == [
        ("Win cash now", SEED_TEXT),
        ("Win cash now", EVOLVED_TEXT),
    ]
    assert all(c == CRITERIA for _, _, c, _ in port.calls)
    assert {m for _, _, _, m in port.calls} == {"judge"}
    first = run.pairs[0]
    assert (first.record_id, first.label) == (HELD_OUT[0].record_id, 1)
    assert (first.seed_probability, first.evolved_probability) == (0.6, 0.9)
    assert [p.label for p in run.pairs] == [1, 0, 1, 0]


def test_score_held_out_refuses_a_non_held_out_row_before_any_call() -> None:
    port = AnsweringPort()

    with pytest.raises(ValueError, match="'train'"):
        score_held_out(
            port,
            SEED,
            KEY,
            evolved=TEXT_PARTS,
            rows=HeldOutRows((*HELD_OUT, _record("Hi", False, "train")), frozenset()),
            judge_model="judge",
            failures=(RuntimeError,),
        )
    assert port.calls == []


def test_score_held_out_stops_at_the_first_failure() -> None:
    port = AnsweringPort(fail_at=4)

    run = score_held_out(
        port,
        SEED,
        KEY,
        evolved=TEXT_PARTS,
        rows=HeldOutRows(HELD_OUT, frozenset()),
        judge_model="judge",
        failures=(RuntimeError,),
    )

    assert run.stopped == "RuntimeError: backend down"
    assert run.calls == 4
    assert len(run.pairs) == 1


def _run() -> Any:
    return score_held_out(
        AnsweringPort(),
        SEED,
        KEY,
        evolved=TEXT_PARTS,
        rows=HeldOutRows(HELD_OUT, frozenset()),
        judge_model="judge",
        failures=(RuntimeError,),
    )


def test_receipt_holds_the_per_call_records_and_their_summary() -> None:
    records = (
        CallRecord(0, "a", (KEY,), 0.5, 11, None),
        CallRecord(1, "b", (KEY,), 1.5, 13, None),
    )

    receipt = held_out_receipt(
        replace(_run(), call_records=records),
        seed_text=SEED_TEXT,
        evolved_text=EVOLVED_TEXT,
        backend="llama_cpp",
        model="judge",
        pins={},
        identity={},
    )

    assert receipt["per_call"] == [r.to_mapping() for r in records]
    assert receipt["call_summary"] == call_summary(records)
    assert receipt["call_summary"]["input_tokens_total"] == 24


def test_receipt_holds_metrics_bootstrap_verdict_and_the_133_reference() -> None:
    run = _run()
    labels = [1, 0, 1, 0]

    receipt = held_out_receipt(
        run,
        seed_text=SEED_TEXT,
        evolved_text=EVOLVED_TEXT,
        backend="llama_cpp",
        model="judge",
        pins={"dataset_revision": "abc"},
        identity={"run_id": "r1"},
    )

    seed = wording_metrics([0.6, 0.4, 0.6, 0.4], labels)
    evolved = wording_metrics([0.9, 0.1, 0.9, 0.1], labels)
    assert receipt["metrics"] == {
        "seed": seed.to_mapping(),
        "evolved": evolved.to_mapping(),
    }
    assert receipt["verdict"] == pass_verdict(seed, evolved).to_mapping()
    assert receipt["verdict"]["passed"] is True
    assert receipt["bootstrap"]["resamples"] == 2000
    assert receipt["bootstrap"]["seed"] == 0
    assert receipt["bootstrap"]["brier_difference"]["high"] < 0
    assert receipt["reference_133"] == {
        "prior_ece": REFERENCE_ECE,
        "seed_ece": seed.ece,
        "difference": seed.ece - REFERENCE_ECE,
    }
    assert receipt["calls"] == {"seed": 4, "evolved": 4, "total": 8}
    assert receipt["seed_text"] == SEED_TEXT
    assert receipt["evolved_text"] == EVOLVED_TEXT
    assert receipt["pins"] == {"dataset_revision": "abc"}
    assert receipt["identity"] == {"run_id": "r1"}
    assert receipt["stopped"] is None
    assert receipt["pairs"][0] == {
        "record_id": HELD_OUT[0].record_id,
        "label": 1,
        "seed": 0.6,
        "evolved": 0.9,
    }
    json.dumps(receipt)


def test_receipt_of_a_stopped_run_has_no_verdict() -> None:
    run = score_held_out(
        AnsweringPort(fail_at=1),
        SEED,
        KEY,
        evolved=TEXT_PARTS,
        rows=HeldOutRows(HELD_OUT, frozenset()),
        judge_model="judge",
        failures=(RuntimeError,),
    )

    receipt = held_out_receipt(
        run,
        seed_text=SEED_TEXT,
        evolved_text=EVOLVED_TEXT,
        backend="vllm",
        model="judge",
        pins={},
        identity={},
    )

    assert receipt["stopped"] == "RuntimeError: backend down"
    assert receipt["metrics"] is None
    assert receipt["bootstrap"] is None
    assert receipt["verdict"] is None
    assert receipt["reference_133"] is None


def _digests(text: str) -> dict[str, Any]:
    """Return the expected digests of ``{"instructions": text}`` (#362)."""
    canonical = json.dumps(
        {"instructions": text}, sort_keys=True, separators=(",", ":")
    )
    loose = json.dumps({"instructions": text}, sort_keys=True, ensure_ascii=False)
    return {
        "components": {"instructions": hashlib.sha256(text.encode()).hexdigest()},
        "mapping": hashlib.sha256(canonical.encode()).hexdigest(),
        "gepa_candidate_id": hashlib.sha256(loose.encode()).hexdigest()[:12],
    }


def test_receipt_pins_the_seed_and_evolved_wording_by_digest() -> None:
    plain = Noul(instructions=SEED_TEXT)
    run = score_held_out(
        AnsweringPort(),
        plain,
        KEY,
        evolved=WordingParts.instructions_only(plain, EVOLVED_TEXT),
        rows=HeldOutRows(HELD_OUT, frozenset()),
        judge_model="judge",
        failures=(RuntimeError,),
    )
    receipt = held_out_receipt(
        run,
        seed_text=SEED_TEXT,
        evolved_text=EVOLVED_TEXT,
        backend="vllm",
        model="judge",
        pins={},
        identity={},
    )

    assert receipt["wording_digests"] == {
        "seed": _digests(SEED_TEXT),
        "evolved": _digests(EVOLVED_TEXT),
    }
    assert receipt["seed_text"] == SEED_TEXT
    assert receipt["evolved_text"] == EVOLVED_TEXT
    json.dumps(receipt)


def test_evolution_artifact_holds_both_texts_and_the_selection_rows() -> None:
    result = EvolutionResult(
        original_score=1.2,
        final_score=1.5,
        evolved_components={KEY: EVOLVED_TEXT},
        iteration_history=[],
        total_iterations=0,
        valset_score=0.75,
    )
    run = WordingRun(WordingParts.from_texts(SEED_TEXT, EVOLVED_TEXT), result)
    train = _records(2, 3, "train")
    validation = _records(1, 1, "validation")
    config = WordingRunConfig(
        reflector="openai/reflector",
        judge_model="judge",
        reflection_minibatch_size=4,
    )

    artifact = evolution_artifact(
        run, config=config, train=train, validation=validation
    )

    assert artifact["seed_text"] == SEED_TEXT
    assert artifact["evolved_text"] == EVOLVED_TEXT
    assert artifact["reflector"] == "openai/reflector"
    assert artifact["judge_model"] == "judge"
    assert artifact["train_rows"] == 5
    assert artifact["validation_ids"] == [r.record_id for r in validation]
    assert artifact["length_cap"] == {"instructions": int(1.5 * len(SEED_TEXT))}
    assert artifact["components"] == ["instructions"]
    assert artifact["seed_parts"] == {"instructions": SEED_TEXT}
    assert artifact["evolved_parts"] == {"instructions": EVOLVED_TEXT}
    assert artifact["wording_digests"] == wording_digests(run.parts)
    assert artifact["settings"]["max_iterations"] == 10
    assert artifact["settings"]["reflection_minibatch_size"] == 4
    assert artifact["result"]["valset_score"] == 0.75
    assert artifact["per_call"] == []
    assert artifact["call_summary"]["calls"] == 0
    json.dumps(artifact)


def test_evolution_artifact_holds_the_per_call_records() -> None:
    result = EvolutionResult(
        original_score=1.2,
        final_score=1.5,
        evolved_components={KEY: EVOLVED_TEXT},
        iteration_history=[],
        total_iterations=0,
        valset_score=0.75,
    )
    records = (
        CallRecord(0, "a", (KEY,), 0.5, 11, None),
        CallRecord(1, "b", (KEY,), 1.5, None, "RuntimeError: down"),
    )

    artifact = evolution_artifact(
        WordingRun(WordingParts.from_texts(SEED_TEXT, EVOLVED_TEXT), result),
        config=WordingRunConfig(reflector="openai/reflector", judge_model="judge"),
        train=_records(1, 1, "train"),
        validation=_records(1, 1, "validation"),
        call_records=records,
    )

    assert artifact["per_call"] == [r.to_mapping() for r in records]
    assert artifact["call_summary"] == call_summary(records)
    json.dumps(artifact)


def test_score_held_out_lets_an_unexpected_error_propagate() -> None:
    with pytest.raises(RuntimeError, match="backend down"):
        score_held_out(
            AnsweringPort(fail_at=1),
            SEED,
            KEY,
            evolved=TEXT_PARTS,
            rows=HeldOutRows(HELD_OUT, frozenset()),
            judge_model="judge",
            failures=(OSError,),
        )


def test_train_subset_defaults_to_1000_stratified_rows() -> None:
    records = _records(1019, 4240, "train")

    subset = train_subset(records)

    # 1000 * 1019 / 5259 = 193.76 and 1000 * 4240 / 5259 = 806.24.
    assert DEFAULT_TRAIN_ROWS == 1000
    assert Counter(r.example.label for r in subset) == {"scam": 194, "legit": 806}
    assert subset == stratified_subset(records, 1000, seed=0)
    assert train_subset(list(reversed(records))) == subset


def test_train_subset_takes_an_override_or_every_row() -> None:
    records = _records(20, 80, "train")

    assert len(train_subset(records, 10)) == 10
    assert train_subset(records, None) == tuple(records)


@pytest.mark.parametrize(
    ("fail_at", "seed_calls", "evolved_calls"),
    [(None, 4, 4), (3, 2, 1), (4, 2, 2), (1, 1, 0)],
)
def test_calls_are_counted_per_arm_as_they_happen(
    fail_at: int | None, seed_calls: int, evolved_calls: int
) -> None:
    run = score_held_out(
        AnsweringPort(fail_at=fail_at),
        SEED,
        KEY,
        evolved=TEXT_PARTS,
        rows=HeldOutRows(HELD_OUT, frozenset()),
        judge_model="judge",
        failures=(RuntimeError,),
    )

    assert (run.seed_calls, run.evolved_calls) == (seed_calls, evolved_calls)
    assert run.calls == seed_calls + evolved_calls
    receipt = held_out_receipt(
        run,
        seed_text=SEED_TEXT,
        evolved_text=EVOLVED_TEXT,
        backend="llama_cpp",
        model="judge",
        pins={},
        identity={},
    )
    assert receipt["calls"] == {
        "seed": seed_calls,
        "evolved": evolved_calls,
        "total": seed_calls + evolved_calls,
    }


def test_score_held_out_refuses_a_prior_measured_row_before_any_call() -> None:
    port = AnsweringPort()

    with pytest.raises(ValueError, match="#236"):
        score_held_out(
            port,
            SEED,
            KEY,
            evolved=TEXT_PARTS,
            rows=HeldOutRows(HELD_OUT, frozenset({HELD_OUT[2].record_id})),
            judge_model="judge",
            failures=(RuntimeError,),
        )
    assert port.calls == []


def test_the_309_receipt_refuses_a_validation_run() -> None:
    run = score_held_out(
        AnsweringPort(),
        SEED,
        KEY,
        evolved=TEXT_PARTS,
        rows=ValidationRows((_record("Win cash now", True, "validation"),)),
        judge_model="judge",
        failures=(RuntimeError,),
    )

    with pytest.raises(ValueError, match="'validation'"):
        held_out_receipt(
            run,
            seed_text=SEED_TEXT,
            evolved_text=EVOLVED_TEXT,
            backend="vllm",
            model="judge",
            pins={},
            identity={},
        )


FULL_PARTS = seed_mapping(SEED)
EVOLVED_PARTS = FULL_PARTS | {
    "instructions": EVOLVED_TEXT,
    "criteria_true": "It asks for money",
}
MULTI = WordingParts(("instructions", "criteria_true"), FULL_PARTS, EVOLVED_PARTS)


def _multi_run() -> Any:
    return score_held_out(
        AnsweringPort(),
        SEED,
        KEY,
        evolved=MULTI,
        rows=HeldOutRows(HELD_OUT, frozenset()),
        judge_model="judge",
        failures=(RuntimeError,),
    )


def test_receipt_records_the_selection_and_both_full_mappings() -> None:
    run = _multi_run()

    receipt = held_out_receipt(
        run,
        seed_text=SEED_TEXT,
        evolved_text=EVOLVED_TEXT,
        backend="vllm",
        model="judge",
        pins={},
        identity={},
    )

    assert run.parts is MULTI

    assert receipt["components"] == ["instructions", "criteria_true"]
    assert receipt["seed_parts"] == FULL_PARTS
    assert receipt["evolved_parts"] == EVOLVED_PARTS
    assert receipt["wording_digests"] == wording_digests(MULTI)
    assert (receipt["seed_text"], receipt["evolved_text"]) == (SEED_TEXT, EVOLVED_TEXT)
    json.dumps(receipt)


def test_receipt_without_parts_records_the_instructions_only() -> None:
    receipt = held_out_receipt(
        replace(_run(), parts=None),
        seed_text=SEED_TEXT,
        evolved_text=EVOLVED_TEXT,
        backend="vllm",
        model="judge",
        pins={},
        identity={},
    )

    assert receipt["components"] == ["instructions"]
    assert receipt["seed_parts"] == {"instructions": SEED_TEXT}
    assert receipt["evolved_parts"] == {"instructions": EVOLVED_TEXT}


def test_receipt_refuses_parts_that_disagree_with_the_texts() -> None:
    with pytest.raises(ValueError, match="seed_text") as caught:
        held_out_receipt(
            _multi_run(),
            seed_text="Another seed",
            evolved_text=EVOLVED_TEXT,
            backend="vllm",
            model="judge",
            pins={},
            identity={},
        )
    assert "Another seed" not in str(caught.value)


def test_score_held_out_sends_the_evolved_criteria_with_the_evolved_arm() -> None:
    port = AnsweringPort()

    score_held_out(
        port,
        SEED,
        KEY,
        evolved=MULTI,
        rows=HeldOutRows(HELD_OUT[:1], frozenset()),
        judge_model="judge",
        failures=(RuntimeError,),
    )

    assert [(w, c) for _, w, c, _ in port.calls] == [
        (SEED_TEXT, CRITERIA),
        (EVOLVED_TEXT, {"true": "It asks for money", "false": "It is legitimate"}),
    ]


def test_score_held_out_refuses_parts_of_another_seed() -> None:
    port = AnsweringPort()
    other = Noul(instructions="Another seed", criteria=CRITERIA)

    with pytest.raises(ValueError, match="seed's parts") as caught:
        score_held_out(
            port,
            other,
            KEY,
            evolved=MULTI,
            rows=HeldOutRows(HELD_OUT[:1], frozenset()),
            judge_model="judge",
            failures=(RuntimeError,),
        )
    assert "Another seed" not in str(caught.value)
    assert port.calls == []


def test_score_held_out_takes_the_parts_under_the_question_name() -> None:
    port = AnsweringPort()

    run = score_held_out(
        port,
        SEED,
        question_name=KEY,
        evolved=MULTI,
        rows=HeldOutRows(HELD_OUT[:1], frozenset()),
        judge_model="judge",
        failures=(RuntimeError,),
    )

    assert run.parts is MULTI
    assert [w for _, w, _, _ in port.calls] == [SEED_TEXT, EVOLVED_TEXT]


def test_receipt_of_an_instructions_only_run_records_the_seed_criteria() -> None:
    receipt = held_out_receipt(
        _run(),
        seed_text=SEED_TEXT,
        evolved_text=EVOLVED_TEXT,
        backend="vllm",
        model="judge",
        pins={},
        identity={},
    )

    assert receipt["components"] == ["instructions"]
    assert receipt["seed_parts"] == FULL_PARTS
    assert receipt["evolved_parts"] == FULL_PARTS | {"instructions": EVOLVED_TEXT}
