"""Contract tests: the supported-imports page matches each package surface (#256).

``docs/reference/supported-imports.md`` is the contract for the public import
surface. Each supported package has one section whose table rows name the
package's ``__all__``. Each name in ``__all__`` is a re-export: the package
``__init__`` imports it from a submodule, and it is the same object as the
definition in that submodule (the gepa-adk ADR-014 re-export check).
"""

from __future__ import annotations

import ast
import importlib
import pkgutil
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.contract

REPO_ROOT = Path(__file__).resolve().parents[2]
SUPPORTED_IMPORTS = REPO_ROOT / "docs" / "reference" / "supported-imports.md"

SUPPORTED_PACKAGES: tuple[str, ...] = (
    "typevet",
    "typevet.adapters.diagnostics",
    "typevet.adapters.inbound",
    "typevet.adapters.outbound",
    "typevet.adapters.outbound.gemma",
    "typevet.adapters.outbound.llama_cpp",
    "typevet.adapters.outbound.vllm",
    "typevet.domain",
    "typevet.ports",
    "typevet.runtime",
    "typevet.testing",
)

ORGANIZATIONAL_PACKAGES: tuple[str, ...] = ("typevet.adapters",)

_HEADING = re.compile(r"^#{2,3} (?:Root )?`(typevet(?:\.\w+)*)`\s*$")
_ANY_HEADING = re.compile(r"^#{1,6} ")
_ROW = re.compile(r"^\|\s*(?P<cell>[^|]*?)\s*\|")
_SEPARATOR = re.compile(r"^:?-+:?$")
_CODE = re.compile(r"`([^`]+)`")


def _documented_rows() -> dict[str, list[str]]:
    """Map each package section of the page to the names in its table rows.

    Returns:
        The names in the first table column of each package section, in page
        order. A section without a table maps to an empty list.
    """
    sections: dict[str, list[str]] = {}
    current: str | None = None
    for line in SUPPORTED_IMPORTS.read_text(encoding="utf-8").splitlines():
        heading = _HEADING.match(line)
        if heading is not None:
            current = heading.group(1)
            sections.setdefault(current, [])
            continue
        if _ANY_HEADING.match(line):
            current = None
            continue
        row = _ROW.match(line)
        if current is None or row is None:
            continue
        cell = row.group("cell")
        if cell == "Name" or _SEPARATOR.match(cell):
            continue
        sections[current].extend(_CODE.findall(cell))
    return sections


def _init_import_sources(package_name: str) -> dict[str, str]:
    """Map each name that the package ``__init__`` imports to its source module.

    Args:
        package_name: Dotted name of the package.

    Returns:
        The imported name mapped to the absolute dotted source module.
    """
    package = importlib.import_module(package_name)
    init = Path(str(package.__file__))
    tree = ast.parse(init.read_text(encoding="utf-8"))
    sources: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom):
            continue
        module = node.module or ""
        if node.level:
            parts = package_name.split(".")
            base = parts[: len(parts) - node.level + 1]
            module = ".".join([*base, module] if module else base)
        for alias in node.names:
            sources[alias.asname or alias.name] = module
    return sources


def test_page_has_one_section_per_supported_package() -> None:
    documented = _documented_rows()
    assert sorted(documented) == sorted(SUPPORTED_PACKAGES + ORGANIZATIONAL_PACKAGES)


def test_supported_packages_are_every_package_with_all() -> None:
    root = importlib.import_module("typevet")
    discovered = {"typevet"} | {
        info.name
        for info in pkgutil.walk_packages(root.__path__, prefix="typevet.")
        if info.ispkg and hasattr(importlib.import_module(info.name), "__all__")
    }
    assert sorted(discovered) == sorted(SUPPORTED_PACKAGES)


@pytest.mark.parametrize("package_name", SUPPORTED_PACKAGES)
def test_table_rows_equal_package_all(package_name: str) -> None:
    rows = _documented_rows().get(package_name, [])
    package = importlib.import_module(package_name)
    assert len(rows) == len(set(rows)), f"duplicate rows: {rows}"
    assert sorted(rows) == sorted(package.__all__)


@pytest.mark.parametrize("package_name", ORGANIZATIONAL_PACKAGES)
def test_organizational_package_has_no_surface(package_name: str) -> None:
    package = importlib.import_module(package_name)
    assert not hasattr(package, "__all__")
    assert _documented_rows()[package_name] == []


@pytest.mark.parametrize(
    ("package_name", "name"),
    [
        (package_name, name)
        for package_name in SUPPORTED_PACKAGES
        for name in sorted(importlib.import_module(package_name).__all__)
    ],
)
def test_reexport_is_the_definition(package_name: str, name: str) -> None:
    package = importlib.import_module(package_name)
    sources = _init_import_sources(package_name)
    assert name in sources, f"{package_name}.{name} is not imported by __init__"
    source = importlib.import_module(sources[name])
    exported = getattr(package, name)
    assert getattr(source, name) is exported
    defining_module = getattr(exported, "__module__", None)
    qualname = getattr(exported, "__qualname__", None)
    if isinstance(defining_module, str) and isinstance(qualname, str):
        assert defining_module.startswith("typevet.")
        definition: object = importlib.import_module(defining_module)
        for part in qualname.split("."):
            definition = getattr(definition, part)
        assert definition is exported
