"""Installed-wheel onboarding for offline typed judgment (#190).

Examples:
    ```bash
    uv run pytest -q tests/contract/test_typed_judgment_wheel_onboarding.py
    ```

See Also:
    - [docs/tutorials/first-typed-judgment-offline.md][]: Public quick start
    - [typevet.testing.ScriptedScoringFake][]: Shipped scoring fake
"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.build_wheel_for_tests import (
    build_wheel_to_directory,
    run_isolated_wheel_python,
)

# Documented quick-start snippet: public imports only, no checkout paths.
_ONBOARDING_SCRIPT = """
from typevet.domain import Noul
from typevet.runtime import ScoringJudgmentAdapter
from typevet.testing import ScriptedScoringFake

fake = ScriptedScoringFake(logprobs={"True": -0.2, "False": -1.0})
port = ScoringJudgmentAdapter(fake, tokenize_content=lambda t: (ord(t[0]),))
response = port.judge(
    "I did not authorize this charge.",
    {
        "reports_unauthorized": Noul(
            instructions="Did the customer report unauthorized use?",
            criteria={"true": "Yes", "false": "No"},
        ),
    },
    "fake-judgment",
)
noul = response.nouls["reports_unauthorized"].noul
assert 0.6 < noul < 0.75, noul
print("ok", noul)
"""


@pytest.mark.contract
def test_offline_typed_judgment_from_built_wheel(tmp_path: Path) -> None:
    """Documented example runs from wheel install without PYTHONPATH=src."""
    dist = tmp_path / "dist"
    run_cwd = tmp_path / "isolated_cwd"
    run_cwd.mkdir()
    try:
        build_wheel_to_directory(dist)
    except RuntimeError as exc:
        if str(exc) == "uv not on PATH":
            pytest.skip(str(exc))
        pytest.fail(str(exc))
    wheels = sorted(dist.glob("typevet-*.whl"))
    assert len(wheels) == 1
    wheel = wheels[0]

    completed = run_isolated_wheel_python(
        wheel=wheel,
        source=_ONBOARDING_SCRIPT,
        cwd=run_cwd,
    )
    assert completed.returncode == 0, (
        f"stdout={completed.stdout!r} stderr={completed.stderr!r}"
    )
    assert completed.stdout.strip().startswith("ok")


@pytest.mark.contract
def test_tests_import_fails_on_isolated_wheel(tmp_path: Path) -> None:
    """Red guard: checkout-only ``tests.*`` imports must not ship in the wheel path."""
    dist = tmp_path / "dist"
    run_cwd = tmp_path / "isolated_cwd"
    run_cwd.mkdir()
    try:
        build_wheel_to_directory(dist)
    except RuntimeError as exc:
        if str(exc) == "uv not on PATH":
            pytest.skip(str(exc))
        pytest.fail(str(exc))
    wheels = sorted(dist.glob("typevet-*.whl"))
    wheel = wheels[0]
    bad_import = (
        "import typevet; "
        "from tests.fixtures.scoring_contract import ContractScoringFake"
    )
    completed = run_isolated_wheel_python(
        wheel=wheel,
        source=bad_import,
        cwd=run_cwd,
    )
    assert completed.returncode != 0
    assert "No module named 'tests'" in completed.stderr
