"""Unit tests for the curated doodle category list (#412)."""

from __future__ import annotations

import pytest

from typevet_evals.doodle_duel import DOODLE_CATEGORIES

pytestmark = pytest.mark.unit

_CHOICE_CONTROLS = 36
_MIN_CATEGORIES = 20


def test_categories_fit_choice_controls() -> None:
    assert _MIN_CATEGORIES <= len(DOODLE_CATEGORIES) <= _CHOICE_CONTROLS
    assert all(name and name == name.strip().lower() for name in DOODLE_CATEGORIES)


def test_categories_are_unique() -> None:
    assert len(set(DOODLE_CATEGORIES)) == len(DOODLE_CATEGORIES)
