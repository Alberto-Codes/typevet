"""Contract: the PSAI vision consumer family lives in the evals member (#256 E5).

The twelve ``typevet.evaluation.psai_vision_consumer_*`` modules and
``consumer_http_accounting`` move to ``typevet_evals.psai_vision_consumer``.
``psai_vision_probability_evidence`` moves to the member root, and the CORD
acceptance CLI moves from ``typevet.adapters.inbound`` to
``typevet_evals.cli``. Each old path must not resolve, and each new path must
import.
"""

from __future__ import annotations

import ast
import importlib
import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.contract

_OLD = "typevet.evaluation.psai_vision_consumer_"
_NEW = "typevet_evals.psai_vision_consumer."

# Old module path -> new module path.
MOVED_MODULES: dict[str, str] = {
    **{
        f"{_OLD}{name}": f"{_NEW}{name}"
        for name in (
            "accounting",
            "dispatch",
            "harness",
            "live",
            "live_identity",
            "live_receipt",
            "live_router",
            "offline",
            "outcomes",
            "protocol",
            "receipt",
            "receipt_structure",
        )
    },
    "typevet.evaluation.consumer_http_accounting": f"{_NEW}http_accounting",
    "typevet.evaluation.psai_vision_probability_evidence": (
        "typevet_evals.psai_vision_probability_evidence"
    ),
    "typevet.adapters.inbound.cord_semantic_acceptance_cli": (
        "typevet_evals.cli.cord_semantic_acceptance"
    ),
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
    package = importlib.import_module("typevet_evals.psai_vision_consumer")
    assert package.__doc__
    assert package.__all__
    for name in package.__all__:
        exported = getattr(package, name)
        defining = importlib.import_module(exported.__module__)
        assert getattr(defining, name) is exported


def test_cli_package_lists_the_cord_command() -> None:
    cli = importlib.import_module("typevet_evals.cli")
    assert "cord_semantic_acceptance" in cli.__all__


def _imported_modules(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
        elif isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
    return names


def test_family_does_not_import_the_cli_package() -> None:
    """The family sits below ``typevet_evals.cli`` (#256 E5 finding 1)."""
    package = importlib.import_module("typevet_evals.psai_vision_consumer")
    assert package.__file__
    offenders = sorted(
        f"{path.name}: {name}"
        for path in Path(package.__file__).parent.glob("*.py")
        for name in _imported_modules(path)
        if name == "typevet_evals.cli" or name.startswith("typevet_evals.cli.")
    )
    assert offenders == []
