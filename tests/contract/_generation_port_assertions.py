"""Shared assertions for GenerationPort contract parity."""

from __future__ import annotations

from typing import Any

import pytest

from tests.fixtures.generation_contract import exc_type_from_name
from typevet.domain.errors import BackendHttpError, GenerationError
from typevet.domain.models import GenerationResult


def assert_results_agree(
    fake: GenerationResult,
    real: GenerationResult,
    *,
    expected_raw_text: str,
) -> None:
    """Assert fake and adapter agree on validated value and model."""
    assert fake.value == real.value
    assert fake.model == real.model
    assert real.raw_text == expected_raw_text


def assert_errors_agree(
    fake_exc: GenerationError,
    real_exc: GenerationError,
    fixture: dict[str, Any],
) -> None:
    """Assert fake and adapter raised equivalent domain errors."""
    assert type(fake_exc) is type(real_exc)
    expect = fixture["expect"]
    if isinstance(fake_exc, BackendHttpError) and isinstance(
        real_exc, BackendHttpError
    ):
        assert fake_exc.status_code == real_exc.status_code == expect["status_code"]
        assert fake_exc.body_snippet == real_exc.body_snippet == expect["body_snippet"]


def run_success_contract(
    *,
    fixture: dict[str, Any],
    fake_generate: Any,
    real_generate: Any,
) -> None:
    """Exercise a success fixture on fake and real ports."""
    expect = fixture["expect"]
    fake_result = fake_generate()
    real_result = real_generate()
    assert_results_agree(
        fake_result,
        real_result,
        expected_raw_text=expect["raw_text"],
    )


def run_error_contract(
    *,
    fixture: dict[str, Any],
    fake_generate: Any,
    real_generate: Any,
) -> None:
    """Exercise an error fixture on fake and real ports."""
    exc_type = exc_type_from_fixture(fixture)
    with pytest.raises(exc_type) as fake_info:
        fake_generate()
    with pytest.raises(exc_type) as real_info:
        real_generate()
    assert_errors_agree(fake_info.value, real_info.value, fixture)


def exc_type_from_fixture(fixture: dict[str, Any]) -> type[GenerationError]:
    return exc_type_from_name(fixture["expect"]["exc_type"])
