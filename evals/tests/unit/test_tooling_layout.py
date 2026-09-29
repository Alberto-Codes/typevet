"""The eval runner CLI and the wheel proof tooling live in the evals member (#256 E2).

The old library paths must not resolve, and the ``typevet_evals`` paths must
import with the same public callables.
"""

from __future__ import annotations

import importlib
import importlib.util

import pytest

pytestmark = pytest.mark.unit

# Old library module -> home in the evals member.
MOVED_MODULES: dict[str, str] = {
    "typevet.adapters.inbound.eval_cli": "typevet_evals.cli.eval_runner",
    "typevet.testing.wheel_isolated": "typevet_evals.wheel_isolated",
    "typevet.evaluation.gemma_native_vision_wheel_smoke": (
        "typevet_evals.gemma_native_vision_wheel_smoke"
    ),
}

PUBLIC_CALLABLES: dict[str, tuple[str, ...]] = {
    "typevet_evals.cli.eval_runner": ("main",),
    "typevet_evals.wheel_isolated": ("run_isolated_wheel_python",),
    "typevet_evals.gemma_native_vision_wheel_smoke": ("run_wheel_smoke",),
}


@pytest.mark.parametrize(("old_name", "new_name"), sorted(MOVED_MODULES.items()))
def test_old_library_path_is_gone_and_member_path_imports(
    old_name: str, new_name: str
) -> None:
    assert importlib.util.find_spec(old_name) is None
    importlib.import_module(new_name)


@pytest.mark.parametrize(("module_name", "names"), sorted(PUBLIC_CALLABLES.items()))
def test_member_module_keeps_its_callables(
    module_name: str, names: tuple[str, ...]
) -> None:
    module = importlib.import_module(module_name)
    for name in names:
        assert callable(getattr(module, name)), f"{module_name} lost {name}"
