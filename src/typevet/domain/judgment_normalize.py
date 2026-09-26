"""Normalize native judgment questions to ``Decision`` and control-token specs.

Examples:
    ```python
    from typevet.domain.judgment_normalize import (
        bind_control_candidates,
        control_binding_pairs,
        normalize_question,
    )

    decision = normalize_question(
        Noul(instructions="Is it urgent?"),
        field_name="urgent",
    )
    specs = bind_control_candidates(("true", "false"), lambda s: (42,))
    ```

See Also:
    - [typevet.domain.judgment_questions][]: Noul, Choice, Score
    - [typevet.domain.decisions][]: Decision compile shape
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from typevet.domain.candidate_scoring_request import CandidateTokenSpec
from typevet.domain.decisions import Decision
from typevet.domain.errors import JudgmentValidationError
from typevet.domain.judgment_questions import Choice, Noul, Question, Score

_NOUL_LABELS = ("true", "false")
_MIN_SCORE_LEVELS = 2


def _question_text(instructions: object | None) -> str:
    if instructions is None:
        return ""
    if not isinstance(instructions, str):
        msg = "instructions must be a string or None"
        raise JudgmentValidationError(msg)
    return instructions


def _reject_duplicate_labels(labels: Sequence[str]) -> None:
    if len(set(labels)) != len(labels):
        msg = "duplicate original labels are not allowed"
        raise JudgmentValidationError(msg)


def judgment_original_labels(question: Question) -> tuple[str, ...]:
    """Return ordered original labels for control binding.

    Args:
        question: Native Noul, Choice, or Score question.

    Returns:
        Original label strings in scoring order.

    Raises:
        JudgmentValidationError: Empty or duplicate labels, or too few score levels.
    """
    if isinstance(question, Noul):
        return _NOUL_LABELS
    if isinstance(question, Choice):
        keys = tuple(str(key) for key in question.criteria)
        if not keys or any(not key.strip() for key in keys):
            msg = "choice criteria must be non-empty with labeled keys"
            raise JudgmentValidationError(msg)
        _reject_duplicate_labels(keys)
        return keys
    if isinstance(question, Score):
        n = len(question.criteria)
        if n < _MIN_SCORE_LEVELS:
            msg = "score criteria must contain at least two rubric levels"
            raise JudgmentValidationError(msg)
        labels = tuple(str(i) for i in range(n))
        return labels
    msg = f"unsupported question type: {type(question)!r}"
    raise JudgmentValidationError(msg)


def normalize_noul(question: Noul, *, field_name: str) -> Decision:
    """Map a Noul question to a Bool ``Decision`` with ``(True, False)`` choices.

    Args:
        question: Yes/no judgment question.
        field_name: Compiled property name on the output object.

    Returns:
        Executable Bool decision with ``return_probabilities=True``.

    Raises:
        JudgmentValidationError: When ``instructions`` is not a string or None.
    """
    return Decision(
        field_name,
        _question_text(question.instructions),
        (True, False),
        syntax="Bool",
        return_probabilities=True,
    )


def normalize_choice(question: Choice, *, field_name: str) -> Decision:
    """Map a Choice question to a Choice ``Decision`` preserving key order.

    Args:
        question: Multi-option judgment question.
        field_name: Compiled property name on the output object.

    Returns:
        Executable Choice decision with criteria key order preserved.

    Raises:
        JudgmentValidationError: Invalid instructions or empty criteria.
    """
    labels = judgment_original_labels(question)
    return Decision(
        field_name,
        _question_text(question.instructions),
        labels,
        syntax="Choice",
        return_probabilities=True,
    )


def normalize_score(question: Score, *, field_name: str) -> Decision:
    """Map a Score rubric to a Choice ``Decision`` over ``0..n-1`` levels.

    Args:
        question: Ordered rubric judgment question.
        field_name: Compiled property name on the output object.

    Returns:
        Executable Choice decision whose choices are integer rubric indices.

    Raises:
        JudgmentValidationError: Invalid instructions or fewer than two levels.
    """
    n = len(question.criteria)
    if n < _MIN_SCORE_LEVELS:
        msg = "score criteria must contain at least two rubric levels"
        raise JudgmentValidationError(msg)
    return Decision(
        field_name,
        _question_text(question.instructions),
        tuple(range(n)),
        syntax="Choice",
        return_probabilities=True,
    )


def normalize_question(question: Question, *, field_name: str) -> Decision:
    """Normalize any supported native question to an executable ``Decision``.

    Args:
        question: Native Noul, Choice, or Score question.
        field_name: Compiled property name on the output object.

    Returns:
        Executable decision for categorical scoring.

    Raises:
        JudgmentValidationError: Unsupported question type or invalid payload.
    """
    if isinstance(question, Noul):
        return normalize_noul(question, field_name=field_name)
    if isinstance(question, Choice):
        return normalize_choice(question, field_name=field_name)
    if isinstance(question, Score):
        return normalize_score(question, field_name=field_name)
    msg = f"unsupported question type: {type(question)!r}"
    raise JudgmentValidationError(msg)


def control_binding_pairs(
    original_labels: Sequence[Any],
) -> tuple[tuple[str, str], ...]:
    """Return ordinal ``(control_string, original_label)`` pairs in label order.

    Args:
        original_labels: Public answer keys before execute alignment.

    Returns:
        Pairs such as ``("0", "billing")`` for each scored candidate.

    Raises:
        JudgmentValidationError: Empty, blank, or duplicate original labels.
    """
    labels = tuple(str(label) for label in original_labels)
    if not labels or any(not label.strip() for label in labels):
        msg = "original labels must be non-empty"
        raise JudgmentValidationError(msg)
    _reject_duplicate_labels(labels)
    return tuple((str(index), labels[index]) for index in range(len(labels)))


def bind_control_candidates(
    original_labels: Sequence[Any],
    tokenize_content: Callable[[str], Sequence[int]],
) -> tuple[CandidateTokenSpec, ...]:
    """Bind ordinal control strings to original labels via ``tokenize_content``.

    Uses ``control_binding_pairs`` for ``("0", label)`` … ordering, then attaches
    single-token ids from ``tokenize_content``.

    Args:
        original_labels: Labels preserved on each ``CandidateTokenSpec``.
        tokenize_content: Tokenizer hook; each control ``"0"``, ``"1"``, … must
            encode to exactly one token id.

    Returns:
        Single-token candidate specs in original-label order.

    Raises:
        JudgmentValidationError: Duplicate or empty labels, or multi-token controls.
    """
    specs: list[CandidateTokenSpec] = []
    for control, label in control_binding_pairs(original_labels):
        token_ids = tuple(tokenize_content(control))
        if len(token_ids) != 1:
            msg = (
                f"control string {control!r} for label {label!r} "
                "must tokenize to exactly one token"
            )
            raise JudgmentValidationError(msg)
        specs.append(CandidateTokenSpec(label, token_ids))
    return tuple(specs)
