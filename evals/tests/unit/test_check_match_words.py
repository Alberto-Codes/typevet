"""Unit tests for the written check amount in English words (#315)."""

from __future__ import annotations

import pytest

from typevet_evals.check_match import amount_in_words

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("cents", "words"),
    [
        (1, "Zero and 01/100"),
        (100, "One and 00/100"),
        (1_100, "Eleven and 00/100"),
        (1_300, "Thirteen and 00/100"),
        (1_400, "Fourteen and 00/100"),
        (1_550, "Fifteen and 50/100"),
        (1_899, "Eighteen and 99/100"),
        (1_905, "Nineteen and 05/100"),
        (31_400, "Three hundred fourteen and 00/100"),
        (50_000, "Five hundred and 00/100"),
        (1_101_700, "Eleven thousand seventeen and 00/100"),
        (300_000_000, "Three million and 00/100"),
        (2_000, "Twenty and 00/100"),
        (4_217, "Forty-two and 17/100"),
        (10_000, "One hundred and 00/100"),
        (123_456, "One thousand two hundred thirty-four and 56/100"),
        (100_000, "One thousand and 00/100"),
        (1_000_000, "Ten thousand and 00/100"),
        (9_050_099, "Ninety thousand five hundred and 99/100"),
        (
            99_999_999,
            "Nine hundred ninety-nine thousand nine hundred ninety-nine and 99/100",
        ),
        (100_000_000, "One million and 00/100"),
        (1_234_000_010, "Twelve million three hundred forty thousand and 10/100"),
    ],
)
def test_amount_in_words(cents: int, words: str) -> None:
    assert amount_in_words(cents) == words


@pytest.mark.parametrize("cents", [0, -5, 100_000_000_000])
def test_amount_in_words_rejects_out_of_range(cents: int) -> None:
    with pytest.raises(ValueError, match="cents"):
        amount_in_words(cents)
