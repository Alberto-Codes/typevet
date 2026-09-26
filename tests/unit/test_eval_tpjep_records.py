"""Offline tests for TPJEP v0 per-attempt result records (#131)."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from typevet.evaluation.tpjep.records import (
    TPJEP_PROTOCOL_V0,
    TpjepAttemptRecord,
    iter_records_jsonl,
    records_to_jsonl,
    summarize_tpjep_records,
)

PINS = {
    "dataset_git_commit": "f8ce71361165846101d02ebc83ad44e47ae44fc3",
    "dataset_hash_recipe": "typellm_manifest_sha256",
    "dataset_hash": "dc3995d8ae1e2fc8e81ce38431add300eb8bb39b85aadfd0c7c32079382dde51",
    "model": "gemma-test",
}


_ANSWERED_TEMPLATE = TpjepAttemptRecord(
    task_id="easy-fact-00",
    source_tier="easy",
    question_type="Noul",
    model=PINS["model"],
    dataset_git_commit=PINS["dataset_git_commit"],
    dataset_hash_recipe=PINS["dataset_hash_recipe"],
    dataset_hash=PINS["dataset_hash"],
    protocol=TPJEP_PROTOCOL_V0,
    outcome="answered",
    predicted=True,
    expected=True,
    probabilities={"yes": 0.6, "no": 0.4},
    prob_valid=True,
    correct=True,
    error_type=None,
    error_message=None,
    usage={"prompt_tokens": 10, "completion_tokens": 2},
    duration_ms=120,
    server_build="build-1",
    template_class="gemma",
)


def _base(**overrides: object) -> TpjepAttemptRecord:
    return replace(_ANSWERED_TEMPLATE, **overrides)


@pytest.mark.unit
def test_mixed_records_exact_scheduled_count_and_summary() -> None:
    records = [
        _base(task_id="original-policy-01-0", correct=True, prob_valid=True),
        _base(
            task_id="original-intent-01-0",
            outcome="schema_invalid",
            predicted=None,
            prob_valid=False,
            correct=None,
            error_type="SchemaValidationError",
            error_message="bad shape",
        ),
        _base(
            task_id="hard-opus-a-long_policy-01",
            outcome="transport_failed",
            predicted=None,
            prob_valid=False,
            correct=None,
            error_type="TransportError",
            error_message="connection reset",
        ),
        _base(
            task_id="easy-intent-00",
            outcome="skipped",
            predicted=None,
            prob_valid=False,
            correct=None,
            error_type="live_gate",
            error_message="router unset",
        ),
    ]
    summary = summarize_tpjep_records(records)
    assert summary.n_scheduled == 4
    assert summary.n_answered == 1
    assert summary.n_prob_valid == 1
    assert summary.n_correct == 1
    assert summary.n_schema_invalid == 1
    assert summary.n_transport_failed == 1
    assert summary.n_skipped == 1
    assert summary.accuracy_on_prob_valid == 1.0
    assert summary.n_success == 1
    assert summary.dataset_git_commit == PINS["dataset_git_commit"]
    assert summary.dataset_hash_recipe == PINS["dataset_hash_recipe"]
    assert summary.dataset_hash == PINS["dataset_hash"]
    assert summary.protocol == TPJEP_PROTOCOL_V0


@pytest.mark.unit
def test_skip_not_counted_as_success() -> None:
    records = [
        _base(outcome="skipped", predicted=None, prob_valid=False, correct=None),
        _base(
            task_id="other",
            outcome="answered",
            prob_valid=True,
            correct=False,
            predicted=False,
        ),
    ]
    summary = summarize_tpjep_records(records)
    assert summary.n_scheduled == 2
    assert summary.n_success == 0
    assert summary.accuracy_on_prob_valid == 0.0


@pytest.mark.unit
def test_jsonl_round_trip_preserves_hash_and_protocol_fields() -> None:
    records = [
        _base(task_id="original-ordinal-01-0", question_type="Score"),
        _base(
            task_id="hard-opus-a-temporal_numeric-12",
            outcome="prob_invalid",
            prob_valid=False,
            correct=None,
            predicted="x",
        ),
    ]
    text = records_to_jsonl(records)
    round_tripped = list(iter_records_jsonl(text))
    assert len(round_tripped) == 2
    for original, restored in zip(records, round_tripped, strict=True):
        assert restored == original
    first_line = json.loads(text.splitlines()[0])
    assert first_line["protocol"] == TPJEP_PROTOCOL_V0
    assert first_line["dataset_hash"] == PINS["dataset_hash"]
    assert "prompt" not in first_line
    assert "thought" not in first_line


@pytest.mark.unit
def test_fixture_mixed_smoke_jsonl_matches_summary() -> None:
    fixture = (
        Path(__file__).resolve().parents[1]
        / "fixtures"
        / "tpjep"
        / "mixed_attempts_smoke.jsonl"
    ).read_text(encoding="utf-8")
    records = list(iter_records_jsonl(fixture))
    summary = summarize_tpjep_records(records)
    assert summary.n_scheduled == 5
    assert summary.n_answered == 2
    assert summary.n_prob_valid == 2
    assert summary.n_correct == 1
    assert summary.n_schema_invalid == 1
    assert summary.n_transport_failed == 1
    assert summary.n_skipped == 1
