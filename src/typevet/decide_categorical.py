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
        context="Pick one.",
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
from typevet.field_prompt import (
    choice_criteria_from_schema,
    compose_scoring_prefix,
    render_field_instructions,
)
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
    if decision.permutations != 1:
        msg = (
            "permutation averaging is not supported in M1 decide_categorical; "
            "permutations must be 1"
        )
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


def _resolve_context(
    *,
    context: str,
    prompt_alias: str | None,
    inject_prefix: bool,
) -> str:
    if prompt_alias is not None:
        if context.strip():
            msg = "pass only one of context= or prompt= to decide_categorical"
            raise DecisionExecutionError(msg)
        context = prompt_alias
    if not inject_prefix:
        _require_non_empty("context", context)
    elif context.strip():
        pass
    return context


def _native_scoring_prefix(
    *,
    context: str,
    decision: Decision,
    field: Decision | Mapping[str, Any],
) -> str:
    criteria = None
    if isinstance(field, Mapping):
        criteria = choice_criteria_from_schema(field, decision.name)
    field_block = render_field_instructions(decision, choice_criteria=criteria)
    return compose_scoring_prefix(context=context, field_block=field_block)


def decide_categorical(
    *,
    scoring_port: CandidateScoringPort,
    model: str,
    candidates: tuple[CandidateTokenSpec, ...],
    field: Decision | Mapping[str, Any],
    context: str = "",
    inject_prefix: bool = False,
    temperature: float = 1.0,
    **legacy: Any,
) -> CategoricalExecutionResult:
    """Score categorical candidates and return the greedy choice plus distribution.

    Validates the ask before any scoring IO. Compiles ``schema`` when given;
    accepts a pre-compiled ``Decision`` instead. Injected ``scoring_port`` values
    are never closed by this function (construct and own
    ``LlamaCppCandidateScoringAdapter`` at the call site when needed).

    By default the library composes the scoring prefix from ``context`` plus
    rendered field instructions. Set ``inject_prefix=True`` and pass ``prefix=``
    to score an exact caller-owned prefix (``context`` does not alter it).

    Args:
        scoring_port: Offline fake or llama.cpp adapter implementing scoring.
        model: Backend model id forwarded to the scorer.
        candidates: Single-token specs aligned with the decision choices.
        field: Compiled ``Decision`` or JSON Schema with one categorical property.
        context: User or task text for the judgment (native path).
        inject_prefix: When true, require ``prefix=`` and score it verbatim.
        temperature: Softmax temperature for the executor (default 1.0).

    Other Parameters:
        legacy: ``prefix`` for inject mode; ``prompt`` as deprecated ``context``
            alias. Other keys raise ``DecisionExecutionError``.

    Returns:
        Greedy value, full probability table, and raw logprobs from the executor.

    Raises:
        DecisionExecutionError: Invalid ask, unsupported M1 shape, permutations
            other than ``1``, or unknown kwargs.
        SchemaError: Schema cannot be compiled.
        ScoringValidationError: Propagated from the scoring port or executor.
    """
    prefix = legacy.pop("prefix", None)
    prompt_alias = legacy.pop("prompt", None)
    if legacy:
        keys = ", ".join(sorted(legacy))
        msg = f"unsupported decide_categorical arguments: {keys}"
        raise DecisionExecutionError(msg)
    _require_non_empty("model", model)
    resolved_context = _resolve_context(
        context=context,
        prompt_alias=prompt_alias,
        inject_prefix=inject_prefix,
    )
    resolved = _resolve_field(field)
    if inject_prefix:
        if prefix is None:
            msg = "inject_prefix=True requires prefix="
            raise DecisionExecutionError(msg)
        _require_non_empty("prefix", prefix)
        scoring_prefix = prefix
    else:
        if prefix is not None:
            msg = (
                "prefix= is only valid with inject_prefix=True; "
                "omit prefix on the native path"
            )
            raise DecisionExecutionError(msg)
        scoring_prefix = _native_scoring_prefix(
            context=resolved_context,
            decision=resolved,
            field=field,
        )
    return execute_categorical_decision(
        resolved,
        prefix=scoring_prefix,
        candidates=candidates,
        port=scoring_port,
        model=model,
        temperature=temperature,
    )
