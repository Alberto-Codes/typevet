"""Contract: wheel-isolated instruction-variant consumer proof ([#177][i177]).

Examples:
    ```bash
    uv run pytest -q evals/tests/contract/test_instruction_variant_consumer_wheel_proof.py
    ```
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.build_wheel_for_tests import build_wheel_to_directory
from scripts.consumer_instruction_variant_proof import (
    _EXIT_INVALID,
    TYPEVET_WHEEL_SHA256_ENV,
    run_offline_wheel_proof,
    sha256_hex,
    verify_wheel_sha256,
)

FIXTURE_ROOT = (
    Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "psai" / "vision_smoke"
)


@pytest.mark.contract
def test_offline_wheel_instruction_variant_proof_exits_zero(
    tmp_path: Path,
) -> None:
    """Built wheel runs instruction-variant proof from site-packages."""
    code, _wheel, _digest = run_offline_wheel_proof(
        fixture_root=FIXTURE_ROOT,
        work_dir=tmp_path,
    )
    assert code == 0


@pytest.mark.contract
def test_wheel_sha256_env_mismatch_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``TYPEVET_WHEEL_SHA256`` mismatch aborts before isolated dispatch."""
    dist = tmp_path / "dist"
    build_wheel_to_directory(dist)
    wheel = next(dist.glob("typevet-*.whl"))
    measured = sha256_hex(wheel)
    with pytest.raises(ValueError, match="mismatch"):
        verify_wheel_sha256(measured, "0" * len(measured))
    monkeypatch.setenv(TYPEVET_WHEEL_SHA256_ENV, "0" * len(measured))
    code, _wheel, _digest = run_offline_wheel_proof(
        fixture_root=FIXTURE_ROOT,
        work_dir=tmp_path,
    )
    assert code == _EXIT_INVALID
