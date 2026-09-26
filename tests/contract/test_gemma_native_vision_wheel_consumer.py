"""Contract: isolated wheel exercises public native vision factory ([#177][i177]).

Examples:
    ```bash
    uv run pytest -q tests/contract/test_gemma_native_vision_wheel_consumer.py
    ```
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.gemma_native_vision_wheel_proof import run_offline_wheel_smoke


@pytest.mark.contract
def test_isolated_wheel_factory_smoke_uses_tmp_only(tmp_path: Path) -> None:
    """Installed wheel calls ``open_gemma_native_vision_judgment`` without checkout imports."""
    fixture_consumer = Path(__file__).resolve().parents[1] / "fixtures" / "consumer"
    before = {
        p.name for p in fixture_consumer.glob("instruction-variant-receipt-*attempt*")
    }
    code, _wheel = run_offline_wheel_smoke(work_dir=tmp_path)
    assert code == 0
    after = {
        p.name for p in fixture_consumer.glob("instruction-variant-receipt-*attempt*")
    }
    assert before == after
