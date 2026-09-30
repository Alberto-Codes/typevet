"""Unit tests for the check-match generator seed from the environment (#344)."""

from __future__ import annotations

import os

import pytest

from typevet_evals.check_match import (
    DEFAULT_SEED,
    SEED_ENV,
    CheckMatchRun,
    build_check_match_receipt,
    check_match_seed,
    generator_pins,
)

pytestmark = pytest.mark.unit


def _receipt_seed(seed: int) -> object:
    receipt = build_check_match_receipt(
        CheckMatchRun(outcomes=(), stopped=None, wall_seconds=0.0),
        backend="llama_cpp",
        model="fake-model",
        pins=generator_pins(seed),
        identity={"schema": "x"},
    )
    pins = receipt["pins"]
    assert isinstance(pins, dict)
    return pins["generator_seed"]


def test_seed_variable_is_the_documented_name() -> None:
    assert SEED_ENV == "TYPEVET_CHECK_MATCH_SEED"


def test_non_default_seed_from_the_environment_lands_in_the_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(SEED_ENV, "1")
    seed = check_match_seed(os.environ)
    assert seed == 1
    assert _receipt_seed(seed) == 1


def test_unset_or_blank_seed_is_the_316_seed() -> None:
    assert DEFAULT_SEED == 0
    assert check_match_seed({}) == 0
    assert check_match_seed({SEED_ENV: "  "}) == 0
    assert _receipt_seed(check_match_seed({})) == 0


def test_generator_pins_name_the_dataset() -> None:
    assert generator_pins(3) == {
        "dataset": "synthetic checks (#315 generator)",
        "generator_seed": 3,
    }


@pytest.mark.parametrize("raw", ["one", "1.5", "-1"])
def test_seed_that_is_not_a_non_negative_integer_is_refused(raw: str) -> None:
    with pytest.raises(ValueError, match=SEED_ENV):
        check_match_seed({SEED_ENV: raw})
