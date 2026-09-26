"""Wheel packaging checks for py.typed and single-source versioning."""

from __future__ import annotations

import tomllib
import zipfile
from importlib.metadata import version
from pathlib import Path

import pytest

import typevet
from scripts.build_wheel_for_tests import build_wheel_to_directory

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = PROJECT_ROOT / "pyproject.toml"


def _pyproject_version() -> str:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    project_version = data["project"]["version"]
    assert isinstance(project_version, str)
    return project_version


def test_version_matches_pyproject_and_metadata() -> None:
    assert typevet.__version__ == _pyproject_version()
    assert typevet.__version__ == version("typevet")


def test_py_typed_marker_in_source_tree() -> None:
    package_dir = Path(typevet.__file__).resolve().parent
    assert package_dir.name == "typevet"
    assert (package_dir / "py.typed").is_file()


def test_built_wheel_contains_py_typed_marker(tmp_path: Path) -> None:
    try:
        build_wheel_to_directory(tmp_path)
    except RuntimeError as exc:
        if str(exc) == "uv not on PATH":
            pytest.skip(str(exc))
        raise
    wheels = sorted(tmp_path.glob("typevet-*.whl"))
    assert len(wheels) == 1
    with zipfile.ZipFile(wheels[0]) as wheel:
        assert "typevet/py.typed" in wheel.namelist()


def test_built_wheel_ships_clinc_json_resources(tmp_path: Path) -> None:
    try:
        build_wheel_to_directory(tmp_path)
    except RuntimeError as exc:
        if str(exc) == "uv not on PATH":
            pytest.skip(str(exc))
        raise
    wheels = sorted(tmp_path.glob("typevet-*.whl"))
    assert len(wheels) == 1
    with zipfile.ZipFile(wheels[0]) as wheel:
        names = wheel.namelist()
    assert "typevet/evaluation/datasets/clinc_domains.json" in names
    assert "typevet/evaluation/datasets/clinc_plus_intent_names.json" in names
