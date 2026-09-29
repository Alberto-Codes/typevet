"""Shared pytest fixtures for ``evals/tests/``.

``session_uv_cache`` gives the wheel helpers one private uv cache per pytest
session (#268). The ``tests/`` and ``evals/tests/`` conftests both define it.
The first one to run creates the cache, and the other one reuses it.

See Also:
    - [typevet_evals.wheel_isolated.private_uv_cache][]: cache lifecycle
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from typevet_evals.wheel_isolated import private_uv_cache


@pytest.fixture(scope="session", autouse=True)
def session_uv_cache(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """Keep wheel builds and ``uv run --with`` environments out of the shared cache.

    Yields:
        The private uv cache directory for this session.
    """
    with private_uv_cache(tmp_path_factory.getbasetemp()) as cache:
        yield cache
