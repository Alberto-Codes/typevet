"""Import boundary: runtime must not depend on evaluation (#174 child B).

Examples:
    ```bash
    uv run pytest -q tests/contract/test_runtime_evaluation_boundary.py
    ```

See Also:
    - [tool.importlinter.contracts][]: ``runtime_must_not_import_evaluation``
"""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RUNTIME_ROOT = _REPO_ROOT / "src" / "typevet" / "runtime"


@pytest.mark.contract
def test_runtime_must_not_import_evaluation_contract_configured() -> None:
    """Import-linter forbids ``typevet.runtime`` from importing ``typevet.evaluation``."""
    data = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    contracts = data["tool"]["importlinter"]["contracts"]
    match = next(
        (c for c in contracts if c.get("name") == "runtime_must_not_import_evaluation"),
        None,
    )
    assert match is not None
    assert match["type"] == "forbidden"
    assert match["source_modules"] == ["typevet.runtime"]
    assert match["forbidden_modules"] == ["typevet.evaluation"]


@pytest.mark.contract
def test_runtime_sources_do_not_import_evaluation() -> None:
    """Static guard: no runtime module textually imports evaluation."""
    offenders: list[str] = []
    for path in sorted(_RUNTIME_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                offenders.extend(
                    f"{path}: import {alias.name}"
                    for alias in node.names
                    if alias.name == "typevet.evaluation"
                    or alias.name.startswith("typevet.evaluation.")
                )
            elif (
                isinstance(node, ast.ImportFrom)
                and node.module
                and (
                    node.module == "typevet.evaluation"
                    or node.module.startswith("typevet.evaluation.")
                )
            ):
                offenders.append(f"{path}: from {node.module}")
    assert offenders == []
