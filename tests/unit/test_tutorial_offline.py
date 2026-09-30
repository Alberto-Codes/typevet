"""The offline tutorial runs as written and uses only supported imports (#280).

The tutorial page is the source of truth. This module extracts its fenced
code blocks, runs the Python blocks in order as one script, and compares the
printed lines with the ``text`` output block that the page states.

Examples:
    ```bash
    uv run pytest -q tests/unit/test_tutorial_offline.py
    ```

See Also:
    - [docs/tutorials/first-typed-judgment-offline.md][]: The tutorial page
    - [docs/reference/supported-imports.md][]: Supported import paths
"""

from __future__ import annotations

import ast
import io
import re
import runpy
from contextlib import redirect_stdout
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TUTORIAL = REPO_ROOT / "docs" / "tutorials" / "first-typed-judgment-offline.md"
SUPPORTED_IMPORTS = REPO_ROOT / "docs" / "reference" / "supported-imports.md"
FENCE = re.compile(r"^```([a-z]*)\n(.*?)^```$", re.MULTILINE | re.DOTALL)
PACKAGE_HEADING = re.compile(r"^## `(typevet(?:\.[a-z_]+)*)`$", re.MULTILINE)
REQUIRED_LINKS = (
    "../how-to/install.md",
    "../how-to/serve-typevet-on-vllm.md",
    "../how-to/run-gemma4-llamacpp.md",
)

pytestmark = pytest.mark.unit


def _blocks(language: str) -> list[str]:
    """Return the tutorial's fenced code blocks for one info-string language.

    Args:
        language: The fence info string, such as ``python`` or ``text``.

    Returns:
        Each matching block body in page order.
    """
    text = TUTORIAL.read_text(encoding="utf-8")
    return [body for lang, body in FENCE.findall(text) if lang == language]


def _supported_packages() -> set[str]:
    """Return the package paths that the supported-imports page documents.

    Returns:
        The root ``typevet`` package and each ``## `typevet.…``` heading path.
    """
    text = SUPPORTED_IMPORTS.read_text(encoding="utf-8")
    return {"typevet", *PACKAGE_HEADING.findall(text)}


def _typevet_imports(source: str) -> list[tuple[str, list[str]]]:
    """Return each typevet import in ``source`` as module and imported names.

    Args:
        source: Python source text.

    Returns:
        One ``(module, names)`` pair per ``import`` or ``from … import`` that
        targets ``typevet``.
    """
    found: list[tuple[str, list[str]]] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            if node.module.split(".")[0] == "typevet":
                found.append((node.module, [alias.name for alias in node.names]))
        elif isinstance(node, ast.Import):
            found.extend(
                (alias.name, [])
                for alias in node.names
                if alias.name.split(".")[0] == "typevet"
            )
    return found


def test_tutorial_imports_only_supported_package_paths() -> None:
    """Every typevet import uses a package path from the supported-imports page."""
    source = "\n".join(_blocks("python"))
    imports = _typevet_imports(source)
    assert imports, "tutorial has no typevet imports"
    supported = _supported_packages()
    unsupported = sorted({module for module, _ in imports} - supported)
    assert unsupported == [], f"imports outside supported packages: {unsupported}"


def test_tutorial_asks_noul_choice_and_score() -> None:
    """The tutorial imports all three question types."""
    source = "\n".join(_blocks("python"))
    imported = {name for _, names in _typevet_imports(source) for name in names}
    missing = sorted({"Noul", "Choice", "Score"} - imported)
    assert missing == [], f"tutorial does not import: {missing}"


def test_tutorial_installs_from_pypi_first() -> None:
    """The first shell block installs typevet from PyPI."""
    shell = _blocks("bash")
    assert shell, "tutorial has no shell blocks"
    assert "pip install typevet" in shell[0], shell[0]


def test_tutorial_keeps_python_out_of_shell_blocks() -> None:
    """No shell block hides Python code that this module does not run."""
    inline = [block for block in _blocks("bash") if "python -c" in block]
    assert inline == [], "shell blocks hold untested inline Python"


def test_tutorial_links_real_backend_how_tos() -> None:
    """The tutorial links the install, vLLM and llama.cpp how-tos."""
    text = TUTORIAL.read_text(encoding="utf-8")
    missing = [link for link in REQUIRED_LINKS if f"]({link}" not in text]
    assert missing == [], f"tutorial does not link: {missing}"


def test_tutorial_script_prints_stated_output(tmp_path: Path) -> None:
    """The Python blocks, run in order as one file, print the stated output."""
    python_blocks = _blocks("python")
    outputs = _blocks("text")
    assert python_blocks, "tutorial has no Python blocks"
    assert len(outputs) == 1, f"expected one stated output block, got {len(outputs)}"
    script = tmp_path / "first_typed_judgment.py"
    script.write_text("\n".join(python_blocks), encoding="utf-8")
    stdout = io.StringIO()
    with redirect_stdout(stdout):
        runpy.run_path(str(script), run_name="__main__")
    assert stdout.getvalue() == outputs[0]
