"""The documentation site builds with ``mkdocs build --strict``."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner
from markdown import Markdown
from mkdocs.__main__ import cli
from mkdocs.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[2]
MERMAID_ASSET = "assets/javascripts/mermaid.min.js"
MERMAID_VERSION = "11.17.2"
MERMAID_SHA256 = "581ed7d74bd9048d0e3a91363927d72ef22942d7722546b27f7cc29e35390eb8"
API_PACKAGE_PAGES = {
    "root": "typevet",
    "domain": "typevet.domain",
    "ports": "typevet.ports",
    "runtime": "typevet.runtime",
    "inbound": "typevet.adapters.inbound",
}

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
    api_dir = site_dir / "reference" / "api"
    api_index = (api_dir / "index.html").read_text(encoding="utf-8")
    for package, symbol in API_PACKAGE_PAGES.items():
        page = api_dir / package / "index.html"
        assert page.is_file(), f"missing API package page: {package}"
        assert f'id="{symbol}"' in page.read_text(encoding="utf-8")
        assert f'href="{package}/"' in api_index
    gemma_page = (
        site_dir / "explanation" / "how-typevet-works-with-gemma-4" / "index.html"
    )
    gemma_html = gemma_page.read_text(encoding="utf-8")
    assert '<pre class="mermaid">' in gemma_html
    assert f'<script src="../../{MERMAID_ASSET}"></script>' in gemma_html


def test_docs_workflow_pins_mermaid_to_the_extra_javascript_path() -> None:
    config = load_config(str(REPO_ROOT / "mkdocs.yml"))
    assert MERMAID_ASSET in [str(script) for script in config.extra_javascript]
    workflow = yaml.safe_load(
        (REPO_ROOT / ".github" / "workflows" / "docs.yml").read_text(encoding="utf-8")
    )
    steps = workflow["jobs"]["build"]["steps"]
    names = [step.get("name") for step in steps]
    fetch = steps[names.index("Vendor pinned Mermaid")]

    assert fetch["env"] == {
        "MERMAID_VERSION": MERMAID_VERSION,
        "MERMAID_SHA256": MERMAID_SHA256,
        "MERMAID_PATH": f"docs/{MERMAID_ASSET}",
    }
    assert "sha256sum -c" in fetch["run"]
    assert names.index("Vendor pinned Mermaid") < names.index(
        "Build documentation site"
    )


def test_site_markdown_renders_mermaid_fence_as_diagram_block() -> None:
    config = load_config(str(REPO_ROOT / "mkdocs.yml"))
    renderer = Markdown(
        extensions=config.markdown_extensions,
        extension_configs=config.mdx_configs,
    )

    html = renderer.convert("```mermaid\nflowchart TD\n  A --> B\n```\n")

    assert html.startswith('<pre class="mermaid"><code>'), html
    assert "A --&gt; B" in html
