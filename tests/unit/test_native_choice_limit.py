"""Unit tests for the tokenizer-based native Choice option limit."""

from __future__ import annotations

import re

import pytest

from typevet.domain.errors import JudgmentValidationError
from typevet.domain.judgment_normalize import bind_control_candidates


def _digit_tokenize(text: str) -> tuple[int, ...]:
    # One token for "0"-"9"; one token per digit for "10" and above.
    return tuple(ord(char) for char in text)


def _labels(count: int) -> tuple[str, ...]:
    return tuple(f"label{index}" for index in range(count))


@pytest.mark.unit
def test_native_choice_over_tokenizer_limit_reports_capacity() -> None:
    expected = "native Choice supports 10 options on this tokenizer; got 12"
    with pytest.raises(JudgmentValidationError, match=f"^{re.escape(expected)}$"):
        bind_control_candidates(_labels(12), _digit_tokenize)


@pytest.mark.unit
def test_native_choice_capacity_is_measured_from_tokenizer() -> None:
    def five_splits(text: str) -> tuple[int, ...]:
        return (1, 2) if text == "5" else (ord(text[0]),)

    expected = "native Choice supports 5 options on this tokenizer; got 8"
    with pytest.raises(JudgmentValidationError, match=f"^{re.escape(expected)}$"):
        bind_control_candidates(_labels(8), five_splits)


@pytest.mark.unit
def test_native_choice_at_tokenizer_limit_binds() -> None:
    specs = bind_control_candidates(_labels(10), _digit_tokenize)
    assert [spec.label for spec in specs] == list(_labels(10))
    assert [spec.token_ids for spec in specs] == [(ord(str(i)),) for i in range(10)]


@pytest.mark.unit
def test_first_control_failure_keeps_single_token_message() -> None:
    def multi(_: str) -> tuple[int, ...]:
        return (1, 2)

    with pytest.raises(JudgmentValidationError, match="exactly one token"):
        bind_control_candidates(_labels(12), multi)
