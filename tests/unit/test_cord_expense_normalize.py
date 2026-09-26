"""Amount normalization for the CORD expense smoke (#164)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from typevet.evaluation.datasets.cord_expense import normalize_amount

pytestmark = pytest.mark.unit

MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "cord"
    / "expense_smoke"
    / "manifest.json"
)


def _receipt_totals() -> list[tuple[str, str]]:
    raw = json.loads(MANIFEST.read_text(encoding="utf-8"))
    return [(r["annotated_total"], r["normalized_total"]) for r in raw["receipts"]]


def test_normalize_amount_round_trips_all_six_annotated_totals() -> None:
    totals = _receipt_totals()
    assert len(totals) == 6
    assert [normalize_amount(a) for a, _ in totals] == [n for _, n in totals]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("80,500", "80500"),
        ("72.000", "72000"),
        ("2,352,460", "2352460"),
        ("85000", "85000"),
        (" 276,000 ", "276000"),
        ("0", "0"),
    ],
)
def test_normalize_amount_drops_thousands_separators(text: str, expected: str) -> None:
    assert normalize_amount(text) == expected


def test_normalize_amount_is_idempotent() -> None:
    for annotated, normalized in _receipt_totals():
        assert normalize_amount(normalize_amount(annotated)) == normalized


@pytest.mark.parametrize(
    "text",
    [
        "",
        "   ",
        "8?,5?0",
        "80,50",
        "80,5000",
        "1,000.000",
        ",500",
        "500,",
        "1,,000",
        "Rp 80,500",
        "-80,500",
        "80.50",
        "007",
    ],
)
def test_normalize_amount_rejects_masked_or_malformed_text(text: str) -> None:
    with pytest.raises(ValueError, match="amount"):
        normalize_amount(text)
