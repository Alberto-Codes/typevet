"""The judgment text parts page cites source locations that exist (#361)."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PAGE = REPO_ROOT / "docs" / "reference" / "judgment-text-parts.md"
LINKING_PAGES = (
    REPO_ROOT / "docs" / "explanation" / "native-typed-judgments.md",
    REPO_ROOT / "docs" / "explanation" / "wording-evolution-difraud.md",
    REPO_ROOT / "docs" / "how-to" / "use-typevet-as-a-judgevet-provider.md",
)
DEFINED_AT = "Defined at"
CITATION = re.compile(r"`([\w./-]+\.py):(\d+)`")
COMPONENT_NAMES = (
    "instructions",
    "criteria",
    "option_block",
    "context_template",
    "framing_preamble",
    "no_thinking_prefill",
    "control_token_rule",
)
ROW_COUNT = 7

pytestmark = pytest.mark.unit


def _cells(line: str) -> list[str]:
    """Split one Markdown table row into stripped cell texts.

    Args:
        line: A table row that starts and ends with ``|``.

    Returns:
        The cell texts, left to right.
    """
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _defined_at_cells(text: str) -> list[str]:
    """Return the ``Defined at`` cell of each row in the parts table.

    Args:
        text: The Markdown source of the page.

    Returns:
        One cell text per body row of the first table with that column.

    Raises:
        AssertionError: No table has a ``Defined at`` column.
    """
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if not line.startswith("|") or DEFINED_AT not in _cells(line):
            continue
        column = _cells(line).index(DEFINED_AT)
        rows: list[str] = []
        for row in lines[index + 2 :]:
            if not row.startswith("|"):
                break
            rows.append(_cells(row)[column])
        return rows
    msg = f"no table with a {DEFINED_AT!r} column"
    raise AssertionError(msg)


def test_page_lists_seven_parts_with_citations() -> None:
    cells = _defined_at_cells(PAGE.read_text(encoding="utf-8"))
    assert len(cells) == ROW_COUNT
    for cell in cells:
        assert CITATION.search(cell), f"row cites no path:line: {cell!r}"


def test_every_cited_path_exists_and_line_is_in_range() -> None:
    cells = _defined_at_cells(PAGE.read_text(encoding="utf-8"))
    citations = [match for cell in cells for match in CITATION.findall(cell)]
    assert citations
    for path, line in citations:
        source = REPO_ROOT / path
        assert source.is_file(), f"cited path does not exist: {path}"
        length = len(source.read_text(encoding="utf-8").splitlines())
        assert 1 <= int(line) <= length, f"{path}:{line} is past line {length}"


def test_page_names_every_component() -> None:
    text = PAGE.read_text(encoding="utf-8")
    for name in COMPONENT_NAMES:
        assert f"`{name}`" in text, f"component name missing: {name}"


@pytest.mark.parametrize("page", LINKING_PAGES, ids=lambda page: page.name)
def test_linking_pages_link_the_parts_page(page: Path) -> None:
    assert "judgment-text-parts.md" in page.read_text(encoding="utf-8")


def test_defined_at_cells_reads_only_the_named_column() -> None:
    text = (
        "| Part | Defined at |\n"
        "|---|---|\n"
        "| a | `x.py:1` |\n"
        "| b | `y.py:2` |\n"
        "\n"
        "| other |\n"
    )
    assert _defined_at_cells(text) == ["`x.py:1`", "`y.py:2`"]
