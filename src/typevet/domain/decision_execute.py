"""Categorical execution for closed Choice and Bool decisions.

The module performs no I/O itself. It calls the injected
``CandidateScoringPort``, so the effects are those of that port.

M1 execute rejects compiled ``permutations`` other than ``1`` before any
scoring IO (permutation averaging is out of scope). The port result is checked
against the exact scoring request before softmax, so a reordered or partial
result cannot assign one choice's score to another.

Examples:
    ```python
    from typevet.domain.decision_execute import execute_categorical_decision
    from typevet.domain.decisions import Decision
    from typevet.domain.candidate_scoring_request import CandidateTokenSpec

    decision = Decision("c", "Pick.", ("a", "b"), syntax="Choice")
    # port implements CandidateScoringPort (offline fake in tests)
    ```

See Also:
    - [typevet.domain.decisions][]: Decision compile shape
    - [typevet.ports.scoring][]: ``CandidateScoringPort`` (TYPE_CHECKING only)
    - [typevet.domain.candidate_scoring_validate][]: Fail-closed score coverage
    - [typevet.domain.media][]: Images the ``media`` argument carries
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from typevet.domain.candidate_scoring_request import (
    CandidateScoringRequest,
    CandidateTokenSpec,
)
from typevet.domain.candidate_scoring_validate import validate_result_against_request
from typevet.domain.decisions import MAX_ENUM_CHOICES, Decision
from typevet.domain.errors import DecisionExecutionError
from typevet.domain.judgment_response import TokenUsage
from typevet.domain.media import ImageInput
from typevet.domain.scoring_stage import ScoreStage

_PROB_SUM_TOLERANCE = 1e-6
_MIN_CATEGORICAL_CHOICES = 2
_CATEGORICAL_SYNTAX = frozenset({"Choice", "Bool"})

if TYPE_CHECKING:
    from typevet.ports.scoring import CandidateScoringPort


@dataclass(frozen=True, slots=True)
class CategoricalExecutionResult:
    """Greedy categorical decision outcome with full softmax distribution.

    Attributes:
        decision (Decision): The categorical field that was executed.
        value (Any): Selected choice value from ``decision.choices``.
        probabilities (tuple[tuple[Any, float], ...]): Ordered choice/probability
            pairs aligned with ``decision.choices`` after softmax.
        logprobs (tuple[float, ...]): Raw pre-sampling logprobs in choice order.
        model (str): Model id from the scoring port result.
        usage (TokenUsage): Token usage metadata from the scoring port.

    Examples:
        ```python
        # Constructed by execute_categorical_decision; not built by callers.
        ```
    """

    decision: Decision
    value: Any
    probabilities: tuple[tuple[Any, float], ...]
    logprobs: tuple[float, ...]
    model: str
    usage: TokenUsage


def _choice_label(choice: Any) -> str:
    if isinstance(choice, bool):
        return "True" if choice else "False"
    return str(choice)


def _reject_unsupported_permutations(decision: Decision) -> None:
    if decision.permutations == 1:
        return
    msg = (
        "permutation averaging is not supported in M1 categorical execute; "
        "permutations must be 1"
    )
    raise DecisionExecutionError(msg)


def _validate_inputs(
    decision: Decision,
    candidates: tuple[CandidateTokenSpec, ...],
    *,
    temperature: float,
) -> None:
    _reject_unsupported_permutations(decision)
    if decision.syntax not in _CATEGORICAL_SYNTAX:
        msg = (
            f"unsupported decision syntax for categorical execute: {decision.syntax!r}"
        )
        raise DecisionExecutionError(msg)
    if decision.nullable:
        msg = "nullable categorical decisions are not supported in M1 execute"
        raise DecisionExecutionError(msg)
    n = len(decision.choices)
    if n < _MIN_CATEGORICAL_CHOICES or n > MAX_ENUM_CHOICES:
        msg = (
            f"choice count must be between {_MIN_CATEGORICAL_CHOICES} and "
            f"{MAX_ENUM_CHOICES}, got {n}"
        )
        raise DecisionExecutionError(msg)
    if len(candidates) != n:
        msg = (
            f"candidate count {len(candidates)!r} does not match "
            f"decision choices length {n!r}"
        )
        raise DecisionExecutionError(msg)
    if temperature <= 0.0 or not math.isfinite(temperature):
        msg = f"temperature must be a finite value > 0, got {temperature!r}"
        raise DecisionExecutionError(msg)
    for spec, choice in zip(candidates, decision.choices, strict=True):
        if len(spec.token_ids) != 1:
            msg = f"candidate {spec.label!r} must be single-token for M1 execute"
            raise DecisionExecutionError(msg)
        expected = _choice_label(choice)
        if spec.label != expected:
            msg = (
                f"candidate label {spec.label!r} does not align with "
                f"decision choice {choice!r} (expected {expected!r})"
            )
            raise DecisionExecutionError(msg)


def _softmax(logprobs: tuple[float, ...], *, temperature: float) -> tuple[float, ...]:
    scaled = [lp / temperature for lp in logprobs]
    peak = max(scaled)
    exps = [math.exp(s - peak) for s in scaled]
    total = sum(exps)
    if total == 0.0 or not math.isfinite(total):
        msg = "softmax normalization produced non-finite total mass"
        raise DecisionExecutionError(msg)
    probs = tuple(e / total for e in exps)
    if not all(math.isfinite(p) for p in probs):
        msg = "softmax produced non-finite probabilities"
        raise DecisionExecutionError(msg)
    prob_sum = sum(probs)
    if abs(prob_sum - 1.0) > _PROB_SUM_TOLERANCE:
        msg = f"probabilities sum to {prob_sum!r}, expected 1 within {_PROB_SUM_TOLERANCE}"
        raise DecisionExecutionError(msg)
    return probs


def _greedy_index(probabilities: tuple[float, ...]) -> int:
    best = probabilities[0]
    index = 0
    for i, prob in enumerate(probabilities[1:], start=1):
        if prob > best:
            best = prob
            index = i
    return index


def execute_categorical_decision(
    decision: Decision,
    *,
    prefix: str,
    candidates: tuple[CandidateTokenSpec, ...],
    port: CandidateScoringPort,
    model: str,
    temperature: float = 1.0,
    media: tuple[ImageInput, ...] = (),
) -> CategoricalExecutionResult:
    """Score candidates, softmax logprobs, and pick the greedy choice.

    Probabilities are the conditional distribution over the declared candidate
    set at temperature ``temperature`` (default 1) via log-sum-exp softmax.

    Args:
        decision: Closed categorical field (``Choice`` or ``Bool`` syntax).
        prefix: Rendered prompt prefix before candidate tokens.
        candidates: Single-token specs aligned 1:1 with ``decision.choices``.
        port: Scoring port; ``ScoreStage.PRE_SAMPLING`` is required.
        model: Model id forwarded to the scoring request.
        temperature: Softmax temperature; must be finite and strictly positive.
        media: Images the ``prefix`` marks, one ``MEDIA_MARKER`` each.

    Returns:
        ``CategoricalExecutionResult`` with the input ``decision``, selected
        value, full probability table, and raw logprobs.

    Raises:
        DecisionExecutionError: Unsupported syntax, nullable field, permutations
            other than ``1``, alignment, candidate shape, or invalid temperature.
        ScoringValidationError: Propagated when the port result does not match
            the exact request (stage, count, label order, token ids) or holds
            a non-finite logprob, or when ``prefix`` markers do not match
            ``media``. The check runs before softmax normalization.
    """
    _validate_inputs(decision, candidates, temperature=temperature)
    request = CandidateScoringRequest(
        model=model,
        prefix=prefix,
        candidates=candidates,
        stage=ScoreStage.PRE_SAMPLING,
        media=media,
    )
    scored = port.score_candidates(request)
    validate_result_against_request(request, scored)
    logprobs = tuple(row.logprob for row in scored.candidates)
    probabilities = _softmax(logprobs, temperature=temperature)
    index = _greedy_index(probabilities)
    value = decision.choices[index]
    pairs = tuple(
        (choice, probabilities[i]) for i, choice in enumerate(decision.choices)
    )
    return CategoricalExecutionResult(
        decision=decision,
        value=value,
        probabilities=pairs,
        logprobs=logprobs,
        model=scored.model,
        usage=scored.usage,
    )
