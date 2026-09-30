"""Written check amount in English words (#315).

A check prints its amount twice: as a number and in words. This module
writes the words form, for example ``One thousand two hundred thirty-four
and 56/100``. The range is 1 cent up to 999,999,999.99 dollars.

Attributes:
    MAX_CENTS (int): Largest amount in cents that the words form supports.

Examples:
    ```python
    from typevet_evals.check_match.words import amount_in_words

    assert amount_in_words(4_217) == "Forty-two and 17/100"
    ```

See Also:
    - [typevet_evals.check_match.render][]: draws the words on the check
"""

from __future__ import annotations

from typing import Final

MAX_CENTS: Final[int] = 99_999_999_999

_ONES: Final[tuple[str, ...]] = (
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
)
_TENS: Final[tuple[str, ...]] = (
    "",
    "",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
)
_SCALES: Final[tuple[tuple[int, str], ...]] = (
    (1_000_000, "million"),
    (1_000, "thousand"),
)


def _below_thousand(number: int) -> list[str]:
    """Return the words for ``number`` in 1 to 999.

    Args:
        number: Whole number from 1 to 999.

    Returns:
        The words, in order.
    """
    words: list[str] = []
    hundreds, rest = divmod(number, 100)
    if hundreds:
        words += [_ONES[hundreds], "hundred"]
    if rest >= len(_ONES):
        tens, ones = divmod(rest, 10)
        words.append(f"{_TENS[tens]}-{_ONES[ones]}" if ones else _TENS[tens])
    elif rest:
        words.append(_ONES[rest])
    return words


def dollars_in_words(dollars: int) -> str:
    """Return the whole-dollar part of an amount in lower-case words.

    Args:
        dollars: Whole dollars from 0 to 999,999,999.

    Returns:
        The words, for example ``forty-two``. Zero is ``zero``.
    """
    if dollars == 0:
        return _ONES[0]
    words: list[str] = []
    rest = dollars
    for size, name in _SCALES:
        group, rest = divmod(rest, size)
        if group:
            words += [*_below_thousand(group), name]
    if rest:
        words += _below_thousand(rest)
    return " ".join(words)


def amount_in_words(cents: int) -> str:
    """Return the check words form of an amount in cents.

    Args:
        cents: Amount in cents, from 1 to :data:`MAX_CENTS`.

    Returns:
        The dollars in words, then ``and NN/100``, with a capital first
        letter.

    Raises:
        ValueError: When ``cents`` is below 1 or above :data:`MAX_CENTS`.
    """
    if not 1 <= cents <= MAX_CENTS:
        msg = f"cents must be from 1 to {MAX_CENTS}, got {cents}"
        raise ValueError(msg)
    dollars, rest = divmod(cents, 100)
    words = dollars_in_words(dollars)
    return f"{words[0].upper()}{words[1:]} and {rest:02d}/100"
