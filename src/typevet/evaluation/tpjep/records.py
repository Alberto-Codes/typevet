"""Frozen TPJEP v0 per-attempt result records and offline JSONL (#131).

One JSON object per scheduled attempt. Skips and transport failures stay in
``n_scheduled``. Aggregate summaries rebuild from records only (for #106).

Examples:
    ```python
    from typevet.evaluation.tpjep.records import (
        TpjepAttemptRecord,
        iter_records_jsonl,
        records_to_jsonl,
        summarize_tpjep_records,
        TPJEP_PROTOCOL_V0,
    )

    record = TpjepAttemptRecord(
        task_id="easy-fact-00",
        source_tier="easy",
        question_type="Noul",
        model="gemma-test",
        dataset_git_commit="f8ce71361165846101d02ebc83ad44e47ae44fc3",
        dataset_hash_recipe="typellm_manifest_sha256",
        dataset_hash="dc3995d8ae1e2fc8e81ce38431add300eb8bb39b85aadfd0c7c32079382dde51",
        protocol=TPJEP_PROTOCOL_V0,
        outcome="answered",
        predicted=True,
        expected=True,
        probabilities={"yes": 0.6, "no": 0.4},
        prob_valid=True,
        correct=True,
        error_type=None,
        error_message=None,
        usage={"prompt_tokens": 1, "completion_tokens": 1},
        duration_ms=10,
        server_build=None,
        template_class="gemma",
    )
    summary = summarize_tpjep_records([record])
    assert summary.n_scheduled == 1
    ```

See Also:
    - [typevet.evaluation.runner.report][]: Banking77/BoolQ slice counters (#98)
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Final, Literal, cast

TPJEP_PROTOCOL_V0: Final[str] = "TPJEP-v0"

TpjepOutcome = Literal[
    "answered",
    "prob_invalid",
    "schema_invalid",
    "transport_failed",
    "skipped",
]

_OUTCOMES: Final[frozenset[str]] = frozenset(
    {"answered", "prob_invalid", "schema_invalid", "transport_failed", "skipped"}
)

_FORBIDDEN_RECORD_KEYS: Final[frozenset[str]] = frozenset(
    {"prompt", "thought", "raw_prompt", "chain_of_thought"}
)

_ERROR_MESSAGE_MAX: Final[int] = 512

RecordJson = dict[str, Any]


@dataclass(frozen=True, slots=True)
class TpjepAttemptRecord:
    """One TPJEP v0 attempt outcome (scoring fields are not model inputs).

    Attributes:
        task_id (str): JevBench public task id.
        source_tier (str): ``original``, ``easy``, or ``hard``.
        question_type (str): ``Noul``, ``Choice``, or ``Score``.
        model (str): Model id for the attempt.
        dataset_git_commit (str): Pinned corpus git commit.
        dataset_hash_recipe (str): How ``dataset_hash`` was produced.
        dataset_hash (str): Published or recomputed corpus hash.
        protocol (str): ``TPJEP-v0``.
        outcome (TpjepOutcome): Terminal attempt status.
        predicted (object): Model decision when present.
        expected (object): Gold label (scoring only, not prompted).
        probabilities (Mapping[str, float] | None): Optional label probs.
        prob_valid (bool): Strict sum-to-one check passed.
        correct (bool | None): Gold match when scored.
        error_type (str | None): Failure class when not answered.
        error_message (str | None): Bounded error detail.
        usage (Mapping[str, int] | None): Token counts when known.
        duration_ms (int): Wall time for the attempt.
        server_build (str | None): llama.cpp build id when known.
        template_class (str | None): Chat template class when known.

    Examples:
        ```python
        from typevet.evaluation.tpjep.records import (
            TpjepAttemptRecord,
            TPJEP_PROTOCOL_V0,
        )

        TpjepAttemptRecord(
            task_id="easy-fact-00",
            source_tier="easy",
            question_type="Noul",
            model="gemma-test",
            dataset_git_commit="abc",
            dataset_hash_recipe="typellm_manifest_sha256",
            dataset_hash="def",
            protocol=TPJEP_PROTOCOL_V0,
            outcome="skipped",
            predicted=None,
            expected=False,
            probabilities=None,
            prob_valid=False,
            correct=None,
            error_type="live_gate",
            error_message="router unset",
            usage=None,
            duration_ms=0,
            server_build=None,
            template_class=None,
        )
        ```
    """

    task_id: str
    source_tier: str
    question_type: str
    model: str
    dataset_git_commit: str
    dataset_hash_recipe: str
    dataset_hash: str
    protocol: str
    outcome: TpjepOutcome
    predicted: object
    expected: object
    probabilities: Mapping[str, float] | None
    prob_valid: bool
    correct: bool | None
    error_type: str | None
    error_message: str | None
    usage: Mapping[str, int] | None
    duration_ms: int
    server_build: str | None
    template_class: str | None


@dataclass(frozen=True, slots=True)
class TpjepRunSummary:
    """Aggregate counts derived only from attempt records.

    Attributes:
        protocol (str): Shared ``TPJEP-v0`` pin across rows.
        model (str): Shared model id.
        dataset_git_commit (str): Shared corpus commit pin.
        dataset_hash_recipe (str): Shared hash recipe pin.
        dataset_hash (str): Shared corpus hash pin.
        n_scheduled (int): Total attempt lines (includes skips).
        n_answered (int): Rows with outcome ``answered``.
        n_prob_valid (int): Rows with ``prob_valid`` true.
        n_correct (int): Rows with ``correct`` true.
        n_success (int): Same as ``n_correct`` (skips never succeed).
        n_schema_invalid (int): ``schema_invalid`` outcomes.
        n_prob_invalid (int): ``prob_invalid`` outcomes.
        n_transport_failed (int): ``transport_failed`` outcomes.
        n_skipped (int): ``skipped`` outcomes.
        accuracy_on_prob_valid (float | None): ``n_correct / n_prob_valid``.
        prob_valid_rate (float | None): Valid rate excluding skips.

    Examples:
        ```python
        from typevet.evaluation.tpjep.records import summarize_tpjep_records

        summary = summarize_tpjep_records(records)
        assert summary.n_scheduled == len(records)
        ```
    """

    protocol: str
    model: str
    dataset_git_commit: str
    dataset_hash_recipe: str
    dataset_hash: str
    n_scheduled: int
    n_answered: int
    n_prob_valid: int
    n_correct: int
    n_success: int
    n_schema_invalid: int
    n_prob_invalid: int
    n_transport_failed: int
    n_skipped: int
    accuracy_on_prob_valid: float | None
    prob_valid_rate: float | None


def _bound_error_message(message: str | None) -> str | None:
    if message is None:
        return None
    text = message.strip()
    if not text:
        return None
    if len(text) <= _ERROR_MESSAGE_MAX:
        return text
    return text[: _ERROR_MESSAGE_MAX - 3] + "..."


def _parse_probabilities(raw: object) -> Mapping[str, float] | None:
    if raw is None:
        return None
    if isinstance(raw, Mapping):
        return {str(k): float(v) for k, v in raw.items()}
    msg = "TPJEP record probabilities must be an object or null"
    raise ValueError(msg)


def _parse_usage(raw: object) -> Mapping[str, int] | None:
    if raw is None:
        return None
    if isinstance(raw, Mapping):
        return {str(k): int(v) for k, v in raw.items()}
    msg = "TPJEP record usage must be an object or null"
    raise ValueError(msg)


def _parse_correct(raw: object) -> bool | None:
    if raw is None:
        return None
    if isinstance(raw, bool):
        return raw
    msg = "TPJEP record correct must be a boolean or null"
    raise ValueError(msg)


def _validate_outcome_and_protocol(row: Mapping[str, Any]) -> TpjepOutcome:
    extra = _FORBIDDEN_RECORD_KEYS.intersection(row)
    if extra:
        msg = f"TPJEP record must not include {sorted(extra)!r}"
        raise ValueError(msg)
    outcome = row.get("outcome")
    if outcome not in _OUTCOMES:
        msg = f"TPJEP record outcome must be one of {_OUTCOMES}, got {outcome!r}"
        raise ValueError(msg)
    protocol = row.get("protocol")
    if protocol != TPJEP_PROTOCOL_V0:
        msg = f"TPJEP record protocol must be {TPJEP_PROTOCOL_V0!r}, got {protocol!r}"
        raise ValueError(msg)
    return cast(TpjepOutcome, outcome)


def record_to_dict(record: TpjepAttemptRecord) -> RecordJson:
    """Serialize a record for JSONL (no prompts or thoughts).

    Args:
        record: Frozen attempt row.

    Returns:
        JSON-ready mapping for one JSONL line.
    """
    return {
        "task_id": record.task_id,
        "source_tier": record.source_tier,
        "question_type": record.question_type,
        "model": record.model,
        "dataset_git_commit": record.dataset_git_commit,
        "dataset_hash_recipe": record.dataset_hash_recipe,
        "dataset_hash": record.dataset_hash,
        "protocol": record.protocol,
        "outcome": record.outcome,
        "predicted": record.predicted,
        "expected": record.expected,
        "probabilities": dict(record.probabilities) if record.probabilities else None,
        "prob_valid": record.prob_valid,
        "correct": record.correct,
        "error_type": record.error_type,
        "error_message": record.error_message,
        "usage": dict(record.usage) if record.usage else None,
        "duration_ms": record.duration_ms,
        "server_build": record.server_build,
        "template_class": record.template_class,
    }


def record_from_dict(row: Mapping[str, Any]) -> TpjepAttemptRecord:
    """Parse one JSON object into a frozen attempt record.

    Args:
        row: Parsed JSON object for one attempt.

    Returns:
        Validated frozen record.

    Raises:
        ValueError: When protocol, outcome, or field shapes are invalid.
    """
    outcome = _validate_outcome_and_protocol(row)
    protocol = row["protocol"]
    return TpjepAttemptRecord(
        task_id=str(row["task_id"]),
        source_tier=str(row["source_tier"]),
        question_type=str(row["question_type"]),
        model=str(row["model"]),
        dataset_git_commit=str(row["dataset_git_commit"]),
        dataset_hash_recipe=str(row["dataset_hash_recipe"]),
        dataset_hash=str(row["dataset_hash"]),
        protocol=str(protocol),
        outcome=outcome,
        predicted=row.get("predicted"),
        expected=row.get("expected"),
        probabilities=_parse_probabilities(row.get("probabilities")),
        prob_valid=bool(row["prob_valid"]),
        correct=_parse_correct(row.get("correct")),
        error_type=(None if row.get("error_type") is None else str(row["error_type"])),
        error_message=_bound_error_message(
            None if row.get("error_message") is None else str(row["error_message"])
        ),
        usage=_parse_usage(row.get("usage")),
        duration_ms=int(row["duration_ms"]),
        server_build=(
            None if row.get("server_build") is None else str(row["server_build"])
        ),
        template_class=(
            None if row.get("template_class") is None else str(row["template_class"])
        ),
    )


def records_to_jsonl(records: list[TpjepAttemptRecord]) -> str:
    """Encode records as UTF-8 JSONL (one attempt per line).

    Args:
        records: Attempt rows in schedule order.

    Returns:
        JSONL text with trailing newline when non-empty.
    """
    lines = [json.dumps(record_to_dict(r), ensure_ascii=False) for r in records]
    return "\n".join(lines) + ("\n" if lines else "")


def iter_records_jsonl(jsonl_text: str) -> Iterator[TpjepAttemptRecord]:
    """Yield attempt records from JSONL text.

    Args:
        jsonl_text: UTF-8 JSONL body.

    Yields:
        Parsed attempt records in file order.

    Raises:
        ValueError: When a line is invalid JSON or record fields fail checks.
        TypeError: When a line is not a JSON object.
    """
    for line_number, line in enumerate(jsonl_text.splitlines(), 1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            row = json.loads(stripped)
        except json.JSONDecodeError as exc:
            msg = f"TPJEP JSONL line {line_number} is not valid JSON"
            raise ValueError(msg) from exc
        if not isinstance(row, dict):
            msg = f"TPJEP JSONL line {line_number} must be a JSON object"
            raise TypeError(msg)
        yield record_from_dict(row)


def _pin_field(records: list[TpjepAttemptRecord], name: str) -> str:
    values = {getattr(r, name) for r in records}
    if len(values) != 1:
        msg = f"TPJEP records disagree on {name}"
        raise ValueError(msg)
    return next(iter(values))


def summarize_tpjep_records(records: list[TpjepAttemptRecord]) -> TpjepRunSummary:
    """Rebuild run counters from attempt records alone.

    Args:
        records: Non-empty list sharing dataset and model pins.

    Returns:
        Summary with validity, accuracy, and failure denominators.

    Raises:
        ValueError: When ``records`` is empty or pins disagree.
    """
    if not records:
        msg = "records must be non-empty"
        raise ValueError(msg)
    n_answered = sum(1 for r in records if r.outcome == "answered")
    n_prob_valid = sum(1 for r in records if r.prob_valid)
    n_correct = sum(1 for r in records if r.correct is True)
    n_success = n_correct
    n_schema_invalid = sum(1 for r in records if r.outcome == "schema_invalid")
    n_prob_invalid = sum(1 for r in records if r.outcome == "prob_invalid")
    n_transport_failed = sum(1 for r in records if r.outcome == "transport_failed")
    n_skipped = sum(1 for r in records if r.outcome == "skipped")
    scored = len(records) - n_skipped
    accuracy = (n_correct / n_prob_valid) if n_prob_valid else None
    prob_valid_rate = (n_prob_valid / scored) if scored else None
    return TpjepRunSummary(
        protocol=_pin_field(records, "protocol"),
        model=_pin_field(records, "model"),
        dataset_git_commit=_pin_field(records, "dataset_git_commit"),
        dataset_hash_recipe=_pin_field(records, "dataset_hash_recipe"),
        dataset_hash=_pin_field(records, "dataset_hash"),
        n_scheduled=len(records),
        n_answered=n_answered,
        n_prob_valid=n_prob_valid,
        n_correct=n_correct,
        n_success=n_success,
        n_schema_invalid=n_schema_invalid,
        n_prob_invalid=n_prob_invalid,
        n_transport_failed=n_transport_failed,
        n_skipped=n_skipped,
        accuracy_on_prob_valid=accuracy,
        prob_valid_rate=prob_valid_rate,
    )
