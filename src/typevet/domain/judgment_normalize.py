"""Normalize native judgment questions to ``Decision`` and control-token specs.

Ordinal controls are ``"0"`` to ``"9"``, then ``"A"`` to ``"Z"``, so at most 36
labels bind (#287). Ten or fewer labels use digit controls only.

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
    # Noul Control bindings are false-first: Control 0 → false, Control 1 → true.
    specs = bind_control_candidates(("false", "true"), lambda s: (42,))
    ```

See Also:
    - [typevet.domain.judgment_questions][]: Noul, Choice, Score
    - [typevet.domain.decisions][]: Decision compile shape
"""

from __future__ import annotations

import string
from collections.abc import Callable, Sequence
from typing import Any

from typevet.domain.candidate_scoring_request import CandidateTokenSpec
from typevet.domain.decisions import Decision
from typevet.domain.errors import JudgmentValidationError
from typevet.domain.judgment_questions import Choice, Noul, Question, Score

_NOUL_LABELS = ("false", "true")
_MIN_SCORE_LEVELS = 2
# Ordinal control strings: "0".."9" first, then "A".."Z" (#286, #287).
_CONTROL_ALPHABET: tuple[str, ...] = tuple(string.digits + string.ascii_uppercase)


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
    """Map a Noul question to a Bool ``Decision`` with ``(False, True)`` choices.

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
        (False, True),
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

    Controls come from the control alphabet: ``"0"`` to ``"9"`` for the first
    ten labels, then ``"A"`` to ``"Z"``. Ten or fewer labels use digits only.

    Args:
        original_labels: Public answer keys before execute alignment.

    Returns:
        Pairs such as ``("0", "billing")`` for each scored candidate.

    Raises:
        JudgmentValidationError: Empty, blank, or duplicate original labels, or
            more labels than the control alphabet has controls.
    """
    labels = tuple(str(label) for label in original_labels)
    if not labels or any(not label.strip() for label in labels):
        msg = "original labels must be non-empty"
        raise JudgmentValidationError(msg)
    _reject_duplicate_labels(labels)
    if len(labels) > len(_CONTROL_ALPHABET):
        msg = (
            f"native Choice supports at most {len(_CONTROL_ALPHABET)} options; "
            f"got {len(labels)}"
        )
        raise JudgmentValidationError(msg)
    return tuple(zip(_CONTROL_ALPHABET, labels, strict=False))


def bind_control_candidates(
    original_labels: Sequence[Any],
    tokenize_content: Callable[[str], Sequence[int]],
) -> tuple[CandidateTokenSpec, ...]:
    """Bind ordinal control strings to original labels via ``tokenize_content``.

    Uses ``control_binding_pairs`` for ``("0", label)`` … ordering, then attaches
    single-token ids from ``tokenize_content``.

    Args:
        original_labels: Labels preserved on each ``CandidateTokenSpec``.
        tokenize_content: Tokenizer hook; each control ``"0"`` … ``"9"``,
            ``"A"`` … must encode to exactly one token id.

    Returns:
        Single-token candidate specs in original-label order.

    Raises:
        JudgmentValidationError: Duplicate or empty labels, more than 36
            labels, or multi-token controls.
            When leading controls are single tokens and a later one is not, the
            message states the tokenizer capacity: ``native Choice supports N
            options on this tokenizer; got M``. The capacity message applies
            only when at least one control bound; a failure at control ``"0"``
            keeps the per-control message.
    """
    specs: list[CandidateTokenSpec] = []
    pairs = control_binding_pairs(original_labels)
    for control, label in pairs:
        token_ids = tuple(tokenize_content(control))
        if len(token_ids) != 1:
            if specs:
                msg = (
                    f"native Choice supports {len(specs)} options "
                    f"on this tokenizer; got {len(pairs)}"
                )
                raise JudgmentValidationError(msg)
            msg = (
                f"control string {control!r} for label {label!r} "
                "must tokenize to exactly one token"
            )
            raise JudgmentValidationError(msg)
        specs.append(CandidateTokenSpec(label, token_ids))
    return tuple(specs)
