"""Unit tests for the synthetic check register and cases (#315).

The expected-label table is amendment A1 of the #303 design. These tests
copy the table by hand, so a change to the code table fails here.
"""

from __future__ import annotations

import pytest

from typevet_evals.check_match import (
    ACCOUNT_NUMBER,
    EXPECTED_LABELS,
    PAYEES,
    CheckVariant,
    aba_check_digit_ok,
    check_cases,
    register_rows,
)

pytestmark = pytest.mark.unit

CONSISTENT = frozenset({"consistent"})

# Amendment A1: variant -> (accepted verdicts, payee truth, amounts truth).
A1_TABLE: dict[str, tuple[frozenset[str], bool, bool]] = {
    "clean": (CONSISTENT, True, True),
    "payee_changed": (frozenset({"payee_mismatch"}), False, True),
    "written_amount_changed": (frozenset({"amount_mismatch"}), True, False),
    "both_amounts_changed": (frozenset({"amount_mismatch"}), True, False),
    "wrong_date": (frozenset({"date_mismatch"}), True, True),
    "unsigned": (frozenset({"unsigned"}), True, True),
    "low_legibility": (frozenset({"consistent", "cannot_tell"}), True, True),
}


def test_expected_labels_equal_amendment_a1() -> None:
    assert [variant.value for variant in CheckVariant] == list(A1_TABLE)
    for variant in CheckVariant:
        expected = EXPECTED_LABELS[variant]
        verdicts, payee, amounts = A1_TABLE[variant.value]
        assert expected.accepted_verdicts == verdicts, variant
        assert expected.payee_matches is payee, variant
        assert expected.amounts_match is amounts, variant
        in_false_clear = variant is not CheckVariant.LOW_LEGIBILITY
        assert expected.counts_for_false_clear is in_false_clear, variant


def test_slice_is_twenty_rows_by_seven_variants() -> None:
    cases = check_cases()
    assert len(register_rows()) == 20
    assert len(cases) == 140
    assert len({case.case_id for case in cases}) == 140
    for case in cases:
        assert case.expected == EXPECTED_LABELS[case.variant]


def test_same_seed_gives_same_cases_and_other_seed_differs() -> None:
    assert check_cases(seed=0) == check_cases(seed=0)
    assert register_rows(seed=0) != register_rows(seed=1)


def test_every_routing_number_fails_the_aba_check_digit() -> None:
    cases = check_cases()
    for case in cases:
        routing = case.face.routing_number
        assert len(routing) == 9
        assert routing.isdigit()
        assert not aba_check_digit_ok(routing), case.case_id


@pytest.mark.parametrize(
    ("routing", "valid"),
    [
        ("011000015", True),
        ("021000021", True),
        ("111000025", True),
        ("011000016", False),
        ("021000020", False),
        ("12345678", False),
        ("12345678a", False),
    ],
)
def test_aba_check_digit_rule(routing: str, valid: bool) -> None:
    assert aba_check_digit_ok(routing) is valid


def test_account_prints_as_zeros_and_payees_come_from_the_fixed_list() -> None:
    assert set(ACCOUNT_NUMBER) == {"0"}
    for case in check_cases():
        assert case.face.account_number == ACCOUNT_NUMBER
        assert case.row.payee in PAYEES
        assert case.face.payee in PAYEES


def test_face_differs_from_register_only_as_the_variant_says() -> None:
    for case in check_cases():
        row, face, variant = case.row, case.face, case.variant
        assert (face.payee == row.payee) is (variant is not CheckVariant.PAYEE_CHANGED)
        written_ok = face.written_cents == row.amount_cents
        numeric_ok = face.numeric_cents == row.amount_cents
        if variant is CheckVariant.WRITTEN_AMOUNT_CHANGED:
            assert numeric_ok
            assert not written_ok
        elif variant is CheckVariant.BOTH_AMOUNTS_CHANGED:
            assert not numeric_ok
            assert not written_ok
            assert face.written_cents == face.numeric_cents
        else:
            assert numeric_ok
            assert written_ok
        assert (face.date == row.date) is (variant is not CheckVariant.WRONG_DATE)
        assert face.signed is (variant is not CheckVariant.UNSIGNED)
        assert face.check_number == row.check_number
        assert (face.blur_radius > 0) is (variant is CheckVariant.LOW_LEGIBILITY)
        assert face.written_cents > 0
        assert face.numeric_cents > 0


def test_low_legibility_blur_radius_is_recorded_in_range() -> None:
    radii = [
        case.face.blur_radius
        for case in check_cases()
        if case.variant is CheckVariant.LOW_LEGIBILITY
    ]
    assert len(radii) == 20
    assert all(3.5 <= radius <= 5.0 for radius in radii)


def test_register_row_text_names_every_register_field() -> None:
    row = register_rows()[0]
    text = row.as_text()
    assert row.payee in text
    assert row.date.isoformat() in text
    assert str(row.check_number) in text
    whole, cents = divmod(row.amount_cents, 100)
    assert f"{whole:,}.{cents:02d}" in text
