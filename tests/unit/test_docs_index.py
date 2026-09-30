"""The docs index ``docs/README.md`` links every page under ``docs/``."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCS_DIR = REPO_ROOT / "docs"
INDEX = DOCS_DIR / "README.md"
BLOB_PREFIX = "https://github.com/Alberto-Codes/typevet/blob/main/docs/"
LINK_TARGET = re.compile(r"\]\(([^)\s]+)\)")

pytestmark = pytest.mark.unit


def _linked_pages(text: str) -> set[str]:
    """Return docs-relative page paths that the index text links.

    Args:
        text: The Markdown source of the docs index.

    Returns:
        Each link target as a path relative to ``docs/``, with any anchor
        removed. Links to the GitHub blob view of ``docs/`` count.
    """
    pages: set[str] = set()
    for target in LINK_TARGET.findall(text):
        path = target.split("#", 1)[0]
        if path.startswith(BLOB_PREFIX):
            path = path.removeprefix(BLOB_PREFIX)
        pages.add(path)
    return pages


def _docs_pages() -> set[str]:
    """Return every Markdown page under ``docs/`` except the index.

    Returns:
        Each page path relative to ``docs/`` in POSIX form.
    """
    return {
        page.relative_to(DOCS_DIR).as_posix()
        for page in DOCS_DIR.rglob("*.md")
        if page != INDEX
    }


def test_docs_index_links_every_docs_page() -> None:
    pages = _docs_pages()
    assert pages, "no docs pages found under docs/"
    linked = _linked_pages(INDEX.read_text(encoding="utf-8"))
    missing = sorted(pages - linked)
    assert missing == [], f"docs/README.md does not link: {missing}"


def test_linked_pages_reads_relative_blob_and_anchor_links() -> None:
    text = (
        "- [A](how-to/a.md): x\n"
        f"- [B]({BLOB_PREFIX}maintainers/b.md): y\n"
        "- [C](reference/c.md#part): z\n"
    )
    assert _linked_pages(text) == {
        "how-to/a.md",
        "maintainers/b.md",
        "reference/c.md",
    }
