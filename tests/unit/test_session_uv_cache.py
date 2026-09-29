"""The ``tests/`` conftest gives the session a private uv cache (#268)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

SESSION_CACHE_ENV = "TYPEVET_TEST_UV_CACHE_DIR"


@pytest.mark.unit
def test_tests_tree_session_has_a_private_uv_cache(session_uv_cache: Path) -> None:
    assert os.environ.get(SESSION_CACHE_ENV) == str(session_uv_cache)
    assert session_uv_cache.is_dir()
    assert session_uv_cache.resolve() != (Path.home() / ".cache" / "uv").resolve()
