"""Contract: the CORD evaluation family lives in the evals member (#256 E6).

The seven ``typevet.evaluation.cord_expense_*`` and ``cord_semantic_*``
modules move to ``typevet_evals.cord``. Each old path must not resolve, each
new path must import, and the package re-exports its defining objects.
"""

from __future__ import annotations

import importlib
import importlib.util

import pytest

pytestmark = pytest.mark.contract

# Old module path -> new module path.
MOVED_MODULES: dict[str, str] = {
    f"typevet.evaluation.cord_{name}": f"typevet_evals.cord.{name}"
    for name in (
        "expense_call_accounting",
        "expense_live_harness",
        "expense_receipt_requirement",
        "expense_smoke",
        "semantic_acceptance",
        "semantic_acceptance_report",
        "semantic_metrics",
    )
}


def _resolves(name: str) -> bool:
    """Return whether ``name`` resolves; a removed parent package means no."""
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


def test_package_reexports_are_the_defining_objects() -> None:
    package = importlib.import_module("typevet_evals.cord")
    assert package.__doc__
    assert package.__all__
    for name in package.__all__:
        exported = getattr(package, name)
        defining = importlib.import_module(exported.__module__)
        assert getattr(defining, name) is exported
