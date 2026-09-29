"""Import-linter rejects injected forbidden edges (#174 child D).

Each case copies ``src/typevet`` into ``tmp_path``, writes the real
``[tool.importlinter]`` section from ``pyproject.toml`` as a temporary config,
injects one top-level import into the copy and runs import-linter on the copy.
The linter runs in a spawned interpreter whose ``sys.path`` starts with the
copy, so the editable install under ``src`` is not the linted tree.

Examples:
    ```bash
    uv run pytest -q tests/contract/test_import_boundary_negatives.py
    ```

See Also:
    - [tool.importlinter.contracts][]: The four enforced import contracts
    - `tests/contract/test_runtime_evaluation_boundary.py`: TOML and AST guard
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import multiprocessing
import shutil
import sys
import tomllib
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from importlinter.cli import lint_imports

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PACKAGE_SRC = _REPO_ROOT / "src" / "typevet"
_CONTRACT_NAMES = (
    "Hexagonal layers",
    "runtime_must_not_import_evaluation",
    "Fakes stay off the adapters",
    "Domain is IO-free",
)


@dataclass(frozen=True)
class _Edge:
    """One forbidden import edge to inject into the copied package.

    Attributes:
        contract (str): Name of the contract that must report BROKEN.
        module (str): Module path in the copy, relative to the package root.
        statement (str): Top-level import statement to append.

    Examples:
        ```python
        _Edge("Domain is IO-free", "domain/errors.py", "import httpx")
        ```
    """

    contract: str
    module: str
    statement: str


_EDGES = (
    _Edge(
        "runtime_must_not_import_evaluation",
        "runtime/judgment.py",
        "import typevet.evaluation",
    ),
    _Edge("Hexagonal layers", "domain/errors.py", "import typevet.adapters.outbound"),
    _Edge(
        "Fakes stay off the adapters",
        "testing/fakes.py",
        "import typevet.adapters.outbound",
    ),
    _Edge("Domain is IO-free", "domain/errors.py", "import httpx"),
)


def _lint_in_child(copy_root: str, config: str) -> tuple[int, str, str]:
    """Run import-linter on the copied package inside a spawned interpreter.

    Args:
        copy_root: Directory that holds the copied ``typevet`` package.
        config: Path to the temporary import-linter TOML config.

    Returns:
        tuple[int, str, str]: Exit code, captured report text and the origin
        of the ``typevet`` package that the linter resolved.
    """
    sys.path.insert(0, copy_root)
    spec = importlib.util.find_spec("typevet")
    origin = str(spec.origin) if spec is not None else ""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = lint_imports(config_filename=config, no_cache=True, no_logo=True)
    return code, buffer.getvalue(), origin


def _toml_value(value: Any) -> str:
    """Render one import-linter option value as TOML.

    Args:
        value: A bool, string or list of strings from the pyproject section.

    Returns:
        str: The TOML literal for ``value``.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    return json.dumps(value)


def _write_config(path: Path, *, drop: str | None = None) -> Path:
    """Write the pyproject import-linter section to ``path``.

    Args:
        path: Target TOML file.
        drop: Optional contract name to leave out of the written config.

    Returns:
        Path: ``path``, after the write.
    """
    data = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    section = dict(data["tool"]["importlinter"])
    contracts = section.pop("contracts")
    lines = ["[tool.importlinter]"]
    lines.extend(f"{key} = {_toml_value(val)}" for key, val in section.items())
    for contract in contracts:
        if contract["name"] == drop:
            continue
        lines.append("")
        lines.append("[[tool.importlinter.contracts]]")
        lines.extend(f"{key} = {_toml_value(val)}" for key, val in contract.items())
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _copy_package(root: Path) -> Path:
    """Copy ``src/typevet`` under ``root`` without bytecode caches.

    Args:
        root: Directory that receives the ``typevet`` package copy.

    Returns:
        Path: The copied package directory.
    """
    target = root / "typevet"
    shutil.copytree(_PACKAGE_SRC, target, ignore=shutil.ignore_patterns("__pycache__"))
    return target


def _run_lint(copy_root: Path, config: Path) -> tuple[int, str]:
    """Lint the copy in a fresh spawned interpreter and check its target.

    Args:
        copy_root: Directory that holds the copied ``typevet`` package.
        config: Path to the temporary import-linter TOML config.

    Returns:
        tuple[int, str]: Exit code and report text from import-linter.
    """
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=1, mp_context=context) as pool:
        code, report, origin = pool.submit(
            _lint_in_child, str(copy_root), str(config)
        ).result()
    assert Path(origin).is_relative_to(copy_root), origin
    return code, report


def _status(report: str, contract: str) -> str | None:
    """Return the KEPT or BROKEN status that the report gives ``contract``.

    Args:
        report: import-linter report text.
        contract: Contract name to find.

    Returns:
        str | None: ``"KEPT"``, ``"BROKEN"`` or ``None`` when absent.
    """
    for line in report.splitlines():
        stripped = line.strip()
        for status in ("KEPT", "BROKEN"):
            if stripped == f"{contract} {status}":
                return status
    return None


@pytest.mark.contract
def test_uninjected_copy_keeps_every_contract(tmp_path: Path) -> None:
    """A clean copy of the package keeps all four real contracts."""
    _copy_package(tmp_path)
    config = _write_config(tmp_path / "importlinter.toml")

    code, report = _run_lint(tmp_path, config)

    assert code == 0, report
    for name in _CONTRACT_NAMES:
        assert _status(report, name) == "KEPT", report


@pytest.mark.contract
@pytest.mark.parametrize("edge", _EDGES, ids=lambda edge: edge.contract)
def test_injected_edge_breaks_its_contract(tmp_path: Path, edge: _Edge) -> None:
    """One injected top-level import turns its target contract BROKEN."""
    package = _copy_package(tmp_path)
    module = package / edge.module
    source = module.read_text(encoding="utf-8")
    module.write_text(f"{source}\n{edge.statement}\n", encoding="utf-8")
    config = _write_config(tmp_path / "importlinter.toml")

    code, report = _run_lint(tmp_path, config)

    assert code != 0, report
    assert _status(report, edge.contract) == "BROKEN", report
