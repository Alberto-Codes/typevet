"""The documentation site builds with ``mkdocs build --strict``."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner
from markdown import Markdown
from mkdocs.__main__ import cli
from mkdocs.config import load_config

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
    gemma_page = (
        site_dir / "explanation" / "how-typevet-works-with-gemma-4" / "index.html"
    )
    assert '<pre class="mermaid">' in gemma_page.read_text(encoding="utf-8")


def test_site_markdown_renders_mermaid_fence_as_diagram_block() -> None:
    config = load_config(str(REPO_ROOT / "mkdocs.yml"))
    renderer = Markdown(
        extensions=config.markdown_extensions,
        extension_configs=config.mdx_configs,
    )

    html = renderer.convert("```mermaid\nflowchart TD\n  A --> B\n```\n")

    assert html.startswith('<pre class="mermaid"><code>'), html
    assert "A --&gt; B" in html
