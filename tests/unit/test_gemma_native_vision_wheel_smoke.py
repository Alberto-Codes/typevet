"""Unit tests: in-tree native vision wheel smoke ([#177][i177], [#196][i196]).

Examples:
    ```bash
    uv run pytest -q tests/unit/test_gemma_native_vision_wheel_smoke.py
    ```

See Also:
    - [typevet.evaluation.gemma_native_vision_wheel_smoke][]: smoke runner
"""

from __future__ import annotations

import pytest

from typevet.evaluation.gemma_native_vision_wheel_smoke import run_wheel_smoke


@pytest.mark.unit
def test_run_wheel_smoke_offline_router() -> None:
    """Public factory smoke passes against the offline router stub."""
    assert run_wheel_smoke() == 0
