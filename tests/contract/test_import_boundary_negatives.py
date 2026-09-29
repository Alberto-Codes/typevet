"""Import-linter rejects injected forbidden edges (#174 child D).

Each case copies ``src/typevet`` and ``evals/src/typevet_evals`` into
``tmp_path``, writes the real ``[tool.importlinter]`` section from
``pyproject.toml`` as a temporary config, injects one top-level import into the
copy and runs import-linter on the copy. The linter runs in a spawned
interpreter whose ``sys.path`` starts with the copy, so the editable installs
are not the linted trees.

Examples:
    ```bash
    uv run pytest -q tests/contract/test_import_boundary_negatives.py
    ```

See Also:
    - [tool.importlinter.contracts][]: The enforced import contracts
    - `tests/contract/test_library_evals_boundary.py`: TOML and AST guard
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
_PACKAGE_SRCS = {
    "typevet": _REPO_ROOT / "src" / "typevet",
    "typevet_evals": _REPO_ROOT / "evals" / "src" / "typevet_evals",
}
_CONTRACT_NAMES = (
    "Hexagonal layers",
    "Fakes stay off the adapters",
    "Domain is IO-free",
    "The library does not import the evals",
    "Model framing stays off the serving backends",
    "Serving backends stay independent",
    "Evaluation families",
)


@dataclass(frozen=True)
class _Edge:
    """One forbidden import edge to inject into the copied package.

    Attributes:
        contract (str): Name of the contract that must report BROKEN.
        module (str): Module path in the copy, relative to the copy root.
        statement (str): Top-level import statement to append.

    Examples:
        ```python
        _Edge("Domain is IO-free", "typevet/domain/errors.py", "import httpx")
        ```
    """

    contract: str
    module: str
    statement: str


_EDGES = (
    _Edge(
        "Hexagonal layers",
        "typevet/domain/errors.py",
        "import typevet.adapters.outbound",
    ),
    _Edge(
        "Fakes stay off the adapters",
        "typevet/testing/fakes.py",
        "import typevet.adapters.outbound",
    ),
    _Edge("Domain is IO-free", "typevet/domain/errors.py", "import httpx"),
    _Edge(
        "The library does not import the evals",
        "typevet/domain/errors.py",
        "import typevet_evals",
    ),
    _Edge(
        "The library does not import the evals",
        "typevet/runtime/judgment.py",
        "import typevet_evals.datasets",
    ),
    _Edge(
        "Model framing stays off the serving backends",
        "typevet/adapters/outbound/gemma/served_template.py",
        "import typevet.adapters.outbound.llama_cpp",
    ),
    _Edge(
        "Model framing stays off the serving backends",
        "typevet/adapters/outbound/gemma/served_template.py",
        "import typevet.adapters.outbound.vllm",
    ),
    _Edge(
        "Serving backends stay independent",
        "typevet/adapters/outbound/vllm/http_mapping.py",
        "import typevet.adapters.outbound.llama_cpp.http_mapping",
    ),
    _Edge(
        "Evaluation families",
        "typevet_evals/datasets/boolq.py",
        "import typevet_evals.cord",
    ),
    _Edge(
        "Evaluation families",
        "typevet_evals/psai_vision_consumer/harness.py",
        "import typevet_evals.instruction_variant",
    ),
    _Edge(
        "Evaluation families",
        "typevet_evals/runner/core.py",
        "import typevet_evals.cli",
    ),
    _Edge(
        "Evaluation families",
        "typevet_evals/cord/semantic_metrics.py",
        "import typevet_evals.instruction_variant",
    ),
)


def _lint_in_child(copy_root: str, config: str) -> tuple[int, str, tuple[str, ...]]:
    """Run import-linter on the copied package inside a spawned interpreter.

    Args:
        copy_root: Directory that holds the copied ``typevet`` package.
        config: Path to the temporary import-linter TOML config.

    Returns:
        tuple[int, str, tuple[str, ...]]: Exit code, captured report text and
        the origin of each copied package that the linter resolved.
    """
    sys.path.insert(0, copy_root)
    origins: list[str] = []
    for name in _PACKAGE_SRCS:
        spec = importlib.util.find_spec(name)
        origins.append(str(spec.origin) if spec is not None else "")
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        code = lint_imports(config_filename=config, no_cache=True, no_logo=True)
    return code, buffer.getvalue(), tuple(origins)


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


def _copy_packages(root: Path) -> None:
    """Copy ``typevet`` and ``typevet_evals`` under ``root`` without caches.

    Args:
        root: Directory that receives the package copies.
    """
    for name, source in _PACKAGE_SRCS.items():
        shutil.copytree(
            source, root / name, ignore=shutil.ignore_patterns("__pycache__")
        )


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
        code, report, origins = pool.submit(
            _lint_in_child, str(copy_root), str(config)
        ).result()
    for origin in origins:
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
    """A clean copy of the package keeps every real contract."""
    _copy_packages(tmp_path)
    config = _write_config(tmp_path / "importlinter.toml")

    code, report = _run_lint(tmp_path, config)

    assert code == 0, report
    for name in _CONTRACT_NAMES:
        assert _status(report, name) == "KEPT", report


@pytest.mark.contract
@pytest.mark.parametrize("edge", _EDGES, ids=lambda edge: edge.contract)
def test_injected_edge_breaks_its_contract(tmp_path: Path, edge: _Edge) -> None:
    """One injected top-level import turns its target contract BROKEN."""
    _copy_packages(tmp_path)
    module = tmp_path / edge.module
    source = module.read_text(encoding="utf-8")
    module.write_text(f"{source}\n{edge.statement}\n", encoding="utf-8")
    config = _write_config(tmp_path / "importlinter.toml")

    code, report = _run_lint(tmp_path, config)

    assert code != 0, report
    assert _status(report, edge.contract) == "BROKEN", report
