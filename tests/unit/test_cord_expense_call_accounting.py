"""Unit tests for CORD expense judgment call accounting ([#186][i186])."""

from __future__ import annotations

import pytest

from typevet.evaluation.cord_expense_call_accounting import (
    cord_expense_smoke_request_totals,
    summarize_cord_expense_judgment_calls,
)

pytestmark = pytest.mark.unit


def test_judgment_totals_include_omission_and_model_calls_not_row_count() -> None:
    """Omission control plus arms must sum model_calls, not dict lengths."""
    text_only = {f"C{i}": {"model_calls": 1} for i in range(18)}
    image_only = {f"R0{i}": {"model_calls": 1} for i in range(1, 7)}
    combined = {f"C{i}": {"model_calls": 1} for i in range(18)}
    total, arms = summarize_cord_expense_judgment_calls(
        text_only=text_only,
        image_only=image_only,
        combined=combined,
        image_only_omission_calls=1,
    )
    assert total == 43
    assert arms["text_only"] == 18
    assert arms["image_only"] == 6
    assert arms["combined"] == 18
    assert arms["image_only_omission"] == 1


def test_deterministic_missing_receipt_rows_charge_zero_model_calls() -> None:
    """Row count must not inflate totals when model_calls is zero."""
    text_only = {"C1": {"model_calls": 1}}
    image_only = {"R01": {"model_calls": 0, "routing": "deterministic_missing_receipt"}}
    combined = {"C1": {"model_calls": 1}}
    total, arms = summarize_cord_expense_judgment_calls(
        text_only=text_only,
        image_only=image_only,
        combined=combined,
        image_only_omission_calls=1,
    )
    assert total == 3
    assert arms["image_only"] == 0


def test_smoke_request_totals_always_count_omission_baseline() -> None:
    """Live wiring must not drop the image_only omission judgment (#186)."""
    text_only = {f"C{i}": {"model_calls": 1} for i in range(18)}
    image_only = {f"R0{i}": {"model_calls": 1} for i in range(1, 7)}
    combined = {f"C{i}": {"model_calls": 1} for i in range(18)}
    total, arms = cord_expense_smoke_request_totals(
        text_only=text_only,
        image_only=image_only,
        combined=combined,
    )
    assert arms["image_only_omission"] == 1
    assert total == 43
    without_omission, _ = summarize_cord_expense_judgment_calls(
        text_only=text_only,
        image_only=image_only,
        combined=combined,
        image_only_omission_calls=0,
    )
    assert without_omission == total - 1
