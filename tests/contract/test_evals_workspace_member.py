"""The ``evals/`` uv workspace member and its gates (#256 E1).

Examples:
    ```bash
    uv run pytest -q tests/contract/test_evals_workspace_member.py
    ```

See Also:
    - `evals/tests/unit/test_member.py`: Member import proof
    - `tests/contract/test_import_boundary_negatives.py`: Injected edges
"""

from __future__ import annotations

import importlib
import tarfile
import tomllib
import zipfile
from pathlib import Path
from typing import Any

import pytest

from scripts.build_wheel_for_tests import build_wheel_to_directory

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MEMBER = _REPO_ROOT / "evals"
_LIBRARY_EDGE = "The library does not import the evals"


def _toml(path: Path) -> dict[str, Any]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


@pytest.mark.contract
def test_root_declares_the_evals_member_through_the_workspace() -> None:
    root = _toml(_REPO_ROOT / "pyproject.toml")
    assert root["tool"]["uv"]["workspace"]["members"] == ["evals"]
    assert root["tool"]["uv"]["sources"]["typevet-evals"] == {"workspace": True}
    assert "typevet-evals" in root["dependency-groups"]["dev"]
    assert "evals/tests" in root["tool"]["pytest"]["ini_options"]["testpaths"]


@pytest.mark.contract
def test_member_is_private_and_depends_on_the_workspace_library() -> None:
    member = _toml(_MEMBER / "pyproject.toml")
    project = member["project"]
    assert project["name"] == "typevet-evals"
    assert "Private :: Do Not Upload" in project["classifiers"]
    assert "typevet" in project["dependencies"]
    assert member["tool"]["uv"]["sources"]["typevet"] == {"workspace": True}
    for kept in ("README.md", "complementary-manifest.yaml", "fixtures"):
        assert (_MEMBER / kept).exists(), kept


@pytest.mark.contract
def test_member_package_imports() -> None:
    module = importlib.import_module("typevet_evals")
    assert Path(str(module.__file__)).is_relative_to(_MEMBER / "src")


@pytest.mark.contract
def test_import_linter_forbids_the_library_importing_the_evals() -> None:
    section = _toml(_REPO_ROOT / "pyproject.toml")["tool"]["importlinter"]
    assert section["root_packages"] == ["typevet", "typevet_evals"]
    match = next(c for c in section["contracts"] if c["name"] == _LIBRARY_EDGE)
    assert match["type"] == "forbidden"
    assert match["source_modules"] == ["typevet"]
    assert match["forbidden_modules"] == ["typevet_evals"]


@pytest.mark.contract
def test_library_build_holds_no_member_code(tmp_path: Path) -> None:
    try:
        build_wheel_to_directory(tmp_path)
    except RuntimeError as exc:
        if str(exc) == "uv not on PATH":
            pytest.skip(str(exc))
        raise
    wheels = sorted(tmp_path.glob("*.whl"))
    sdists = sorted(tmp_path.glob("*.tar.gz"))
    assert [w.name.split("-")[0] for w in wheels] == ["typevet"]
    assert [s.name.split("-")[0] for s in sdists] == ["typevet"]
    with zipfile.ZipFile(wheels[0]) as wheel:
        wheel_names = wheel.namelist()
    with tarfile.open(sdists[0]) as sdist:
        sdist_names = sdist.getnames()
    for name in [*wheel_names, *sdist_names]:
        assert "typevet_evals" not in name, name
        assert "/evals/" not in f"/{name}", name
