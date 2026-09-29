"""Release metadata checks: one version everywhere, MIT licence, project URLs."""

from __future__ import annotations

import json
import tomllib
import zipfile
from email.parser import Parser
from email.policy import compat32
from pathlib import Path

import pytest

import typevet
from scripts.build_wheel_for_tests import build_wheel_to_directory

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = PROJECT_ROOT / "pyproject.toml"
MANIFEST = PROJECT_ROOT / ".release-please-manifest.json"
RELEASE_CONFIG = PROJECT_ROOT / "release-please-config.json"

EXPECTED_URLS = {
    "Homepage": "https://github.com/Alberto-Codes/typevet",
    "Documentation": "https://alberto-codes.github.io/typevet/",
    "Repository": "https://github.com/Alberto-Codes/typevet",
    "Issues": "https://github.com/Alberto-Codes/typevet/issues",
    "Changelog": "https://github.com/Alberto-Codes/typevet/blob/main/CHANGELOG.md",
}


def _pyproject_version() -> str:
    data = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    project_version = data["project"]["version"]
    assert isinstance(project_version, str)
    return project_version


def test_pyproject_manifest_and_package_versions_agree() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest == {".": _pyproject_version()}
    assert typevet.__version__ == _pyproject_version()


def test_release_please_config_cuts_plain_v_tags_for_the_python_package() -> None:
    config = json.loads(RELEASE_CONFIG.read_text(encoding="utf-8"))
    package = config["packages"]["."]
    assert package["release-type"] == "python"
    assert package["package-name"] == "typevet"
    assert package["include-component-in-tag"] is False
    assert package["bump-minor-pre-major"] is True
    # A forced version may only name the version the manifest already holds.
    assert package.get("release-as", _pyproject_version()) == _pyproject_version()


def test_built_wheel_metadata_carries_mit_licence_and_project_urls(
    tmp_path: Path,
) -> None:
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
        metadata_name = next(n for n in names if n.endswith(".dist-info/METADATA"))
        text = wheel.read(metadata_name).decode("utf-8")
    metadata = Parser(policy=compat32).parsestr(text)
    assert metadata["License-Expression"] == "MIT"
    assert "LICENSE" in (metadata.get_all("License-File") or [])
    assert any(n.endswith(".dist-info/licenses/LICENSE") for n in names)
    urls: dict[str, str] = {}
    for entry in metadata.get_all("Project-URL") or []:
        label, url = entry.split(",", 1)
        urls[label.strip()] = url.strip()
    assert urls == EXPECTED_URLS
