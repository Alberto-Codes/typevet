"""The ``typevet-evals`` workspace member imports, and the library stays apart (#256 E1)."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

import typevet

_LIBRARY_ROOT = Path(typevet.__file__).resolve().parent


@pytest.mark.unit
def test_member_package_imports_from_the_member_source_tree() -> None:
    module = importlib.import_module("typevet_evals")
    origin = Path(str(module.__file__)).resolve()
    assert origin.parts[-4:] == ("evals", "src", "typevet_evals", "__init__.py")
    assert isinstance(module.__all__, list)
    assert module.__doc__


def _imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    return roots


@pytest.mark.unit
def test_library_sources_never_import_the_member() -> None:
    offenders = [
        str(path)
        for path in sorted(_LIBRARY_ROOT.rglob("*.py"))
        if "typevet_evals" in _imported_roots(path)
    ]
    assert offenders == []
