"""The documentation site builds with ``mkdocs build --strict``."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner
from mkdocs.__main__ import cli

REPO_ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.unit


def test_mkdocs_strict_build_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = REPO_ROOT / "mkdocs.yml"
    assert config.is_file(), f"missing site configuration: {config}"
    monkeypatch.chdir(REPO_ROOT)
    site_dir = tmp_path / "site"

    result = CliRunner().invoke(
        cli,
        [
            "build",
            "--strict",
            "--config-file",
            str(config),
            "--site-dir",
            str(site_dir),
        ],
    )

    assert result.exit_code == 0, result.output
    assert (site_dir / "index.html").is_file()
    assert (site_dir / "reference" / "api" / "index.html").is_file()
