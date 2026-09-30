"""Synthetic check register rows, variants and expected labels (#315).

A seeded generator makes 20 register rows. Each row gives 7 check faces, one
per :class:`CheckVariant`, so the slice has 140 cases. Amendment A1 of the
#303 design sets the expected labels per variant. Each check is not
negotiable by construction: the routing number fails the ABA check digit and
the account number prints as zeros. Payees come from a fixed list of
invented names.

Attributes:
    ACCOUNT_NUMBER (str): Account number that every check prints.
    PAYEES (tuple[str, ...]): Invented payee names.
    ROW_COUNT (int): Register rows in the default slice.
    DEFAULT_SEED (int): Seed of the default slice.
    SEED_ENV (str): Environment variable that sets the live-run seed.
    BLUR_RANGE (tuple[float, float]): Low and high blur radius, in pixels.
    EXPECTED_LABELS (Mapping[CheckVariant, ExpectedLabels]): Amendment A1.

Examples:
    ```python
    from typevet_evals.check_match.cases import check_cases

    cases = check_cases()
    assert len(cases) == 140
    ```

See Also:
    - [typevet_evals.check_match.render][]: draws one case as a PNG
    - [typevet_evals.check_match.request][]: one case to one judgment
"""

from __future__ import annotations

import datetime as dt
import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Final

ACCOUNT_NUMBER: Final[str] = "000000000000"
ROW_COUNT: Final[int] = 20
DEFAULT_SEED: Final[int] = 0
SEED_ENV: Final[str] = "TYPEVET_CHECK_MATCH_SEED"
BLUR_RANGE: Final[tuple[float, float]] = (3.5, 5.0)
PAYEES: Final[tuple[str, ...]] = (
    "Northwind Supply Co.",
    "Bluefin Paper Works",
    "Copperleaf Landscaping",
    "Quillmore Office Goods",
    "Harborview Plumbing LLC",
    "Maple Hollow Bakery",
    "Stonebridge Tutoring",
    "Pinecrest Auto Repair",
    "Lanternfield Electric",
    "Redwick Garden Center",
    "Silverbrook Catering",
    "Oakmere Print Shop",
    "Foxglove Cleaning Service",
    "Tidewater Moving Co.",
    "Brightmoor Music School",
    "Juniper Lane Florist",
    "Cobalt Ridge Roofing",
    "Willowmere Pet Clinic",
    "Amberfield Hardware",
    "Kestrel Bay Surveying",
    "Marigold Street Deli",
    "Ironvale Tool Rental",
    "Seabright Window Washers",
    "Thistledown Book Binding",
)
_ABA_WEIGHTS: Final[tuple[int, ...]] = (3, 7, 1, 3, 7, 1, 3, 7, 1)
_FIRST_DAY: Final[dt.date] = dt.date(2025, 1, 1)
_MIN_CENTS: Final[int] = 100


class CheckVariant(StrEnum):
    """How the check face differs from its register row.

    Examples:
        ```python
        assert CheckVariant("unsigned") is CheckVariant.UNSIGNED
        ```
    """

    CLEAN = "clean"
    PAYEE_CHANGED = "payee_changed"
    WRITTEN_AMOUNT_CHANGED = "written_amount_changed"
    BOTH_AMOUNTS_CHANGED = "both_amounts_changed"
    WRONG_DATE = "wrong_date"
    UNSIGNED = "unsigned"
    LOW_LEGIBILITY = "low_legibility"


@dataclass(frozen=True, slots=True)
class ExpectedLabels:
    """Expected answers for one variant (amendment A1).

    Attributes:
        accepted_verdicts (frozenset[str]): ``Choice`` labels that count as
            right.
        payee_matches (bool): Truth of the payee ``Noul``.
        amounts_match (bool): Truth of the amounts ``Noul``.
        counts_for_false_clear (bool): Whether the false-clear rate uses
            this variant.

    Examples:
        ```python
        assert EXPECTED_LABELS[CheckVariant.CLEAN].payee_matches
        ```
    """

    accepted_verdicts: frozenset[str]
    payee_matches: bool
    amounts_match: bool
    counts_for_false_clear: bool = True


EXPECTED_LABELS: Final[Mapping[CheckVariant, ExpectedLabels]] = MappingProxyType(
    {
        CheckVariant.CLEAN: ExpectedLabels(frozenset({"consistent"}), True, True),
        CheckVariant.PAYEE_CHANGED: ExpectedLabels(
            frozenset({"payee_mismatch"}), False, True
        ),
        CheckVariant.WRITTEN_AMOUNT_CHANGED: ExpectedLabels(
            frozenset({"amount_mismatch"}), True, False
        ),
        CheckVariant.BOTH_AMOUNTS_CHANGED: ExpectedLabels(
            frozenset({"amount_mismatch"}), True, False
        ),
        CheckVariant.WRONG_DATE: ExpectedLabels(
            frozenset({"date_mismatch"}), True, True
        ),
        CheckVariant.UNSIGNED: ExpectedLabels(frozenset({"unsigned"}), True, True),
        CheckVariant.LOW_LEGIBILITY: ExpectedLabels(
            frozenset({"consistent", "cannot_tell"}),
            True,
            True,
            counts_for_false_clear=False,
        ),
    }
)


@dataclass(frozen=True, slots=True)
class RegisterRow:
    """One row of the synthetic check register.

    Attributes:
        index (int): Row position in the register, from 0.
        check_number (int): Check number.
        date (datetime.date): Date of the check.
        payee (str): Invented payee name.
        amount_cents (int): Amount in cents.

    Examples:
        ```python
        row = register_rows()[0]
        assert row.payee in row.as_text()
        ```
    """

    index: int
    check_number: int
    date: dt.date
    payee: str
    amount_cents: int

    def as_text(self) -> str:
        """Return the row as one line of register text.

        Returns:
            Check number, ISO date, payee and dollar amount.
        """
        whole, cents = divmod(self.amount_cents, 100)
        return (
            f"check number {self.check_number}; date {self.date.isoformat()}; "
            f"payee {self.payee}; amount ${whole:,}.{cents:02d}"
        )


@dataclass(frozen=True, slots=True)
class CheckFace:
    """What one rendered check shows.

    Attributes:
        check_number (int): Printed check number.
        date (datetime.date): Printed date.
        payee (str): Printed payee.
        numeric_cents (int): Amount in the numeric box, in cents.
        written_cents (int): Amount on the words line, in cents.
        signed (bool): Whether synthetic signature strokes are drawn.
        routing_number (str): Printed routing number; fails the check digit.
        account_number (str): Printed account number; all zeros.
        signature_seed (int): Seed of the synthetic signature strokes.
        blur_radius (float): Gaussian blur radius in pixels; 0 for no blur.

    Examples:
        ```python
        face = check_cases()[0].face
        assert face.account_number == ACCOUNT_NUMBER
        ```
    """

    check_number: int
    date: dt.date
    payee: str
    numeric_cents: int
    written_cents: int
    signed: bool
    routing_number: str
    account_number: str
    signature_seed: int
    blur_radius: float = 0.0


@dataclass(frozen=True, slots=True)
class CheckCase:
    """One register row, one variant, its check face and expected labels.

    Attributes:
        row (RegisterRow): Register row sent as text.
        variant (CheckVariant): How the face differs from the row.
        face (CheckFace): What the rendered check shows.
        expected (ExpectedLabels): Expected answers.

    Examples:
        ```python
        case = check_cases()[0]
        assert case.case_id == "r00:clean"
        ```
    """

    row: RegisterRow
    variant: CheckVariant
    face: CheckFace
    expected: ExpectedLabels

    @property
    def case_id(self) -> str:
        """Return a stable id for receipts.

        Returns:
            ``r<row index>:<variant>``.
        """
        return f"r{self.row.index:02d}:{self.variant.value}"


def aba_check_digit_ok(routing: str) -> bool:
    """Return whether ``routing`` passes the ABA routing check digit.

    The weights are 3, 7, 1 repeating. A valid number has a weighted digit
    sum that is a multiple of 10.

    Args:
        routing: Routing number text.

    Returns:
        ``True`` only for 9 digits with a valid check digit.
    """
    if len(routing) != len(_ABA_WEIGHTS) or not routing.isdigit():
        return False
    total = sum(int(d) * w for d, w in zip(routing, _ABA_WEIGHTS, strict=True))
    return total % 10 == 0


class SeededDraws:
    """Deterministic draws from a text key, built on SHA-256.

    Each draw hashes the key and a counter, so the same key gives the same
    draws on every platform and Python version.

    Attributes:
        key (str): Text key of this draw stream.

    Examples:
        ```python
        draws = SeededDraws("check-match:0")
        assert draws.integer(0, 10) == SeededDraws("check-match:0").integer(0, 10)
        ```
    """

    def __init__(self, key: str) -> None:
        """Start a draw stream for ``key``.

        Args:
            key: Text key; the same key gives the same draws.
        """
        self.key = key
        self._counter = 0

    def _next(self) -> int:
        """Return the next 64-bit draw.

        Returns:
            A whole number from 0 to 2**64 - 1.
        """
        digest = hashlib.sha256(f"{self.key}:{self._counter}".encode()).digest()
        self._counter += 1
        return int.from_bytes(digest[:8], "big")

    def integer(self, low: int, high: int) -> int:
        """Return a whole number from ``low`` up to but not including ``high``.

        Args:
            low: Smallest value.
            high: One more than the largest value.

        Returns:
            The draw.
        """
        return low + self._next() % (high - low)

    def uniform(self, low: float, high: float) -> float:
        """Return a real number from ``low`` to ``high``.

        Args:
            low: Smallest value.
            high: Largest value.

        Returns:
            The draw.
        """
        return low + (high - low) * (self._next() / 2**64)


def _failing_routing_number(draws: SeededDraws) -> str:
    """Return 9 digits whose last digit fails the ABA check digit.

    Args:
        draws: Seeded draw stream.

    Returns:
        A routing number that :func:`aba_check_digit_ok` rejects.
    """
    head = "".join(str(draws.integer(0, 10)) for _ in range(8))
    partial = sum(int(d) * w for d, w in zip(head, _ABA_WEIGHTS, strict=False))
    valid = (-partial) % 10
    return f"{head}{(valid + draws.integer(1, 10)) % 10}"


def register_rows(
    seed: int = DEFAULT_SEED, count: int = ROW_COUNT
) -> list[RegisterRow]:
    """Return ``count`` seeded register rows.

    Args:
        seed: Seed of the register.
        count: Number of rows; at most the number of payees.

    Returns:
        Rows with distinct payees, in index order.
    """
    draws = SeededDraws(f"check-match:register:{seed}")
    payees = sorted(
        PAYEES,
        key=lambda name: hashlib.sha256(f"{seed}:{name}".encode()).hexdigest(),
    )[:count]
    return [
        RegisterRow(
            index=index,
            check_number=1001 + index,
            date=_FIRST_DAY + dt.timedelta(days=draws.integer(0, 365)),
            payee=payee,
            amount_cents=draws.integer(1_500, 500_000),
        )
        for index, payee in enumerate(payees)
    ]


def _changed_amount(draws: SeededDraws, cents: int) -> int:
    """Return a different positive amount near ``cents``.

    Args:
        draws: Seeded draw stream.
        cents: Register amount in cents.

    Returns:
        The register amount plus or minus 5 to 500 dollars.
    """
    delta = draws.integer(500, 50_000)
    lower = cents - delta
    if draws.integer(0, 2) and lower >= _MIN_CENTS:
        return lower
    return cents + delta


def _face(row: RegisterRow, variant: CheckVariant, seed: int) -> CheckFace:
    """Return the check face for ``row`` under ``variant``.

    Args:
        row: Register row.
        variant: How the face differs from the row.
        seed: Seed of the slice.

    Returns:
        The face. Routing number and signature are the same for every
        variant of one row.
    """
    row_draws = SeededDraws(f"check-match:face:{seed}:{row.index}")
    routing = _failing_routing_number(row_draws)
    signature_seed = row_draws.integer(0, 2**31)
    draws = SeededDraws(f"check-match:variant:{seed}:{row.index}:{variant.value}")
    payee, date = row.payee, row.date
    numeric = written = row.amount_cents
    blur = 0.0
    if variant is CheckVariant.PAYEE_CHANGED:
        others = [name for name in PAYEES if name != row.payee]
        payee = others[draws.integer(0, len(others))]
    elif variant is CheckVariant.WRITTEN_AMOUNT_CHANGED:
        written = _changed_amount(draws, row.amount_cents)
    elif variant is CheckVariant.BOTH_AMOUNTS_CHANGED:
        numeric = written = _changed_amount(draws, row.amount_cents)
    elif variant is CheckVariant.WRONG_DATE:
        shift = draws.integer(3, 46) * (1 if draws.integer(0, 2) else -1)
        date = row.date + dt.timedelta(days=shift)
    elif variant is CheckVariant.LOW_LEGIBILITY:
        blur = round(draws.uniform(*BLUR_RANGE), 2)
    return CheckFace(
        check_number=row.check_number,
        date=date,
        payee=payee,
        numeric_cents=numeric,
        written_cents=written,
        signed=variant is not CheckVariant.UNSIGNED,
        routing_number=routing,
        account_number=ACCOUNT_NUMBER,
        signature_seed=signature_seed,
        blur_radius=blur,
    )


def check_cases(seed: int = DEFAULT_SEED, count: int = ROW_COUNT) -> list[CheckCase]:
    """Return every variant of every seeded register row.

    Args:
        seed: Seed of the slice.
        count: Number of register rows.

    Returns:
        ``count`` times 7 cases, row by row, variants in enum order.
    """
    return [
        CheckCase(
            row=row,
            variant=variant,
            face=_face(row, variant, seed),
            expected=EXPECTED_LABELS[variant],
        )
        for row in register_rows(seed, count)
        for variant in CheckVariant
    ]


def check_match_seed(environ: Mapping[str, str]) -> int:
    """Return the generator seed that ``TYPEVET_CHECK_MATCH_SEED`` names (#344).

    Args:
        environ: Process environment, or a mapping in its place.

    Returns:
        The seed; :data:`DEFAULT_SEED` (the #316 seed) when the variable is
        unset or blank.

    Raises:
        ValueError: When the value is not a non-negative integer.
    """
    raw = environ.get(SEED_ENV, "").strip()
    if not raw:
        return DEFAULT_SEED
    if not (raw.isascii() and raw.isdigit()):
        msg = f"{SEED_ENV} must be a non-negative integer: {raw!r}"
        raise ValueError(msg)
    return int(raw)


def generator_pins(seed: int) -> dict[str, object]:
    """Return the receipt pins that name the generated slice and its seed.

    Args:
        seed: Seed that made the slice.

    Returns:
        The ``dataset`` and ``generator_seed`` pins.
    """
    return {"dataset": "synthetic checks (#315 generator)", "generator_seed": seed}
