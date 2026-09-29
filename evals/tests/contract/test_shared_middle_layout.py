"""Contract: the shared evaluation middle layer lives in the evals member (#256 E7).

The ``typevet.evaluation.runner`` and ``typevet.evaluation.tpjep`` packages and
the ``typevet.evaluation.experiment_identity`` module move to
``typevet_evals``. Each old path must not resolve, each new path must import,
and each package re-exports its defining objects.
"""

from __future__ import annotations

import importlib
import importlib.util
import pkgutil

import pytest

pytestmark = pytest.mark.contract

_RUNNER_MODULES = ("core", "datasets", "live_gate", "report")
_TPJEP_MODULES = ("loader", "outcome", "records", "runner")

# Old module path -> new module path.
MOVED_MODULES: dict[str, str] = {
    "typevet.evaluation.experiment_identity": "typevet_evals.experiment_identity",
    "typevet.evaluation.runner": "typevet_evals.runner",
    "typevet.evaluation.tpjep": "typevet_evals.tpjep",
    **{
        f"typevet.evaluation.runner.{name}": f"typevet_evals.runner.{name}"
        for name in _RUNNER_MODULES
    },
    **{
        f"typevet.evaluation.tpjep.{name}": f"typevet_evals.tpjep.{name}"
        for name in _TPJEP_MODULES
    },
}


def _resolves(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


@pytest.mark.parametrize("old_name", sorted(MOVED_MODULES))
def test_old_module_path_does_not_resolve(old_name: str) -> None:
    assert not _resolves(old_name)


@pytest.mark.parametrize("new_name", sorted(MOVED_MODULES.values()))
def test_new_module_path_imports(new_name: str) -> None:
    importlib.import_module(new_name)


@pytest.mark.parametrize(
    "package_name", ["typevet_evals.runner", "typevet_evals.tpjep"]
)
def test_package_reexports_are_the_defining_objects(package_name: str) -> None:
    package = importlib.import_module(package_name)
    assert package.__doc__
    assert package.__all__
    submodules = [
        importlib.import_module(f"{package_name}.{info.name}")
        for info in pkgutil.iter_modules(package.__path__)
    ]
    for name in package.__all__:
        exported = getattr(package, name)
        assert any(getattr(module, name, None) is exported for module in submodules), (
            f"{package_name}.{name} is not re-exported from a submodule"
        )
