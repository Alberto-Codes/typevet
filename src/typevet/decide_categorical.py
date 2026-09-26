"""Public M1 entry for closed categorical decisions via candidate scoring.

Examples:
    ```python
    from typevet import decide_categorical
    from typevet.domain.candidate_scoring_request import CandidateTokenSpec

    schema = {
        "type": "object",
        "properties": {
            "label": {"type": "string", "enum": ["a", "b"]},
        },
        "required": ["label"],
        "additionalProperties": False,
    }
    # scoring_port: offline fake or LlamaCppCandidateScoringAdapter
    decide_categorical(
        field=schema,
        prompt="Pick one.",
        prefix="Answer:",
        model="fake",
        scoring_port=scoring_port,
        candidates=(
            CandidateTokenSpec("a", (1,)),
            CandidateTokenSpec("b", (2,)),
        ),
    )
    ```

See Also:
    - [typevet.domain.decision_execute][]: ``execute_categorical_decision``
    - [typevet.domain.decision_compile][]: ``compile_json_schema``
    - [typevet.ports.scoring][]: ``CandidateScoringPort``
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from typevet.domain.candidate_scoring_request import CandidateTokenSpec
from typevet.domain.decision_compile import compile_json_schema
from typevet.domain.decision_execute import (
    CategoricalExecutionResult,
    execute_categorical_decision,
)
from typevet.domain.decisions import Decision
from typevet.domain.errors import DecisionExecutionError
from typevet.ports.scoring import CandidateScoringPort

_CATEGORICAL_SYNTAX = frozenset({"Choice", "Bool"})


def _require_non_empty(name: str, value: str) -> None:
    if not value.strip():
        msg = f"{name} must be a non-empty string"
        raise DecisionExecutionError(msg)


def _validate_m1_decision(decision: Decision) -> None:
    if decision.syntax not in _CATEGORICAL_SYNTAX:
        msg = (
            f"M1 decide_categorical requires a single Choice or Bool field; "
            f"got syntax {decision.syntax!r}"
        )
        raise DecisionExecutionError(msg)
    if decision.depends_on:
        msg = "decision dependencies are not supported in M1 decide_categorical"
        raise DecisionExecutionError(msg)
    if decision.nullable:
        msg = "nullable categorical fields are not supported in M1 decide_categorical"
        raise DecisionExecutionError(msg)


def _decision_from_schema(schema: Mapping[str, Any]) -> Decision:
    compiled = compile_json_schema(schema)
    if len(compiled) != 1:
        msg = (
            f"M1 decide_categorical requires exactly one schema property; "
            f"got {len(compiled)}"
        )
        raise DecisionExecutionError(msg)
    decision = compiled[0]
    _validate_m1_decision(decision)
    return decision


def _resolve_field(field: Decision | Mapping[str, Any]) -> Decision:
    if isinstance(field, Decision):
        _validate_m1_decision(field)
        return field
    return _decision_from_schema(field)


def decide_categorical(
    *,
    scoring_port: CandidateScoringPort,
    model: str,
    candidates: tuple[CandidateTokenSpec, ...],
    prefix: str,
    prompt: str,
    field: Decision | Mapping[str, Any],
    temperature: float = 1.0,
    **unsupported: Any,
) -> CategoricalExecutionResult:
    """Score categorical candidates and return the greedy choice plus distribution.

    Validates the ask before any scoring IO. Compiles ``schema`` when given;
    accepts a pre-compiled ``Decision`` instead. Injected ``scoring_port`` values
    are never closed by this function (construct and own
    ``LlamaCppCandidateScoringAdapter`` at the call site when needed).

    Args:
        scoring_port: Offline fake or llama.cpp adapter implementing scoring.
        model: Backend model id forwarded to the scorer.
        candidates: Single-token specs aligned with the decision choices.
        prefix: Rendered answer prefix before candidate tokens (offline or live).
        prompt: User message content; must be non-empty (context validation).
        field: Compiled ``Decision`` or JSON Schema with one categorical property.
        temperature: Softmax temperature for the executor (default 1.0).

    Other Parameters:
        unsupported: Extra keyword arguments are rejected with
            ``DecisionExecutionError`` before scoring IO.

    Returns:
        Greedy value, full probability table, and raw logprobs from the executor.

    Raises:
        DecisionExecutionError: Invalid ask, unsupported M1 shape, or unknown kwargs.
        SchemaError: Schema cannot be compiled.
        ScoringValidationError: Propagated from the scoring port or executor.
    """
    if unsupported:
        keys = ", ".join(sorted(unsupported))
        msg = f"unsupported decide_categorical arguments: {keys}"
        raise DecisionExecutionError(msg)
    _require_non_empty("prompt", prompt)
    _require_non_empty("model", model)
    _require_non_empty("prefix", prefix)
    resolved = _resolve_field(field)
    return execute_categorical_decision(
        resolved,
        prefix=prefix,
        candidates=candidates,
        port=scoring_port,
        model=model,
        temperature=temperature,
    )
