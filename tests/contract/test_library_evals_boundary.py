"""Import boundary: the library does not depend on the evals member (#174, #256).

The ``typevet.runtime`` ban on evaluation code widens to the whole library once
evaluation code lives in ``typevet_evals``. The evals member keeps its own
family order.

Examples:
    ```bash
    uv run pytest -q tests/contract/test_library_evals_boundary.py
    ```

See Also:
    - [tool.importlinter.contracts][]: ``The library does not import the evals``
      and ``Evaluation families``
"""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path
from typing import Any

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_LIBRARY_ROOT = _REPO_ROOT / "src" / "typevet"
_EVALS = "typevet_evals"


def _contract(name: str) -> dict[str, Any]:
    data = tomllib.loads((_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    contracts = data["tool"]["importlinter"]["contracts"]
    match = next((c for c in contracts if c.get("name") == name), None)
    assert match is not None, f"missing import-linter contract {name!r}"
    return match


@pytest.mark.contract
def test_library_must_not_import_evals_contract_configured() -> None:
    """Import-linter forbids ``typevet`` from importing ``typevet_evals``."""
    match = _contract("The library does not import the evals")
    assert match["type"] == "forbidden"
    assert match["source_modules"] == ["typevet"]
    assert match["forbidden_modules"] == [_EVALS]


@pytest.mark.contract
def test_evaluation_families_contract_configured() -> None:
    """A layers contract orders every evaluation family, datasets at the base.

    The leaf modules share the top layer with ``cli``: no family imports them.
    """
    match = _contract("Evaluation families")
    assert match["type"] == "layers"
    layers: list[str] = match["layers"]
    top = {name.strip() for name in layers[0].split("|")}
    assert top == {
        f"{_EVALS}.{leaf}"
        for leaf in (
            "cli",
            "gemma_native_vision_wheel_smoke",
            "psai_vision_probability_evidence",
            "wheel_isolated",
        )
    }
    assert layers[-1] == f"{_EVALS}.datasets"
    listed = {name.strip() for layer in layers for name in layer.split("|")}
    for family in (
        "cli",
        "throughput",
        "vllm_acceptance",
        "instruction_variant",
        "cord",
        "psai_vision_consumer",
        "outcome_replay_metrics",
        "runner",
        "tpjep",
        "experiment_identity",
        "datasets",
    ):
        assert f"{_EVALS}.{family}" in listed, family


@pytest.mark.contract
def test_library_sources_do_not_import_evals() -> None:
    """Static guard: no library module textually imports the evals member."""
    offenders: list[str] = []
    for path in sorted(_LIBRARY_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                offenders.extend(
                    f"{path}: import {alias.name}"
                    for alias in node.names
                    if alias.name.split(".")[0] == _EVALS
                )
            elif (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.split(".")[0] == _EVALS
            ):
                offenders.append(f"{path}: from {node.module}")
    assert offenders == []
