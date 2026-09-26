"""Tests for Markdown ownership and exclusions."""

from pathlib import Path

import pytest

from scripts.markdown_prose import paragraphs, patch_selection, sentences

pytestmark = pytest.mark.unit


def test_wrapped_paragraph() -> None:
    assert paragraphs("One line\nand another.\n") == [(1, 2, "One line and another.")]


@pytest.mark.parametrize(
    "excluded",
    [
        "```text\nrobust\n```",
        "~~~\nrobust\n~~~",
        "    robust",
        "\trobust",
        "---\ntitle: robust\n---",
        "<!-- robust\nrobust -->",
        "> robust\n> [Source](https://example.com)",
        "| robust | powerful |\n|---|---|",
    ],
)
def test_exclusions_terminate(excluded: str) -> None:
    found = paragraphs(excluded + "\n\nBlazing prose.\n")
    assert [item[2] for item in found] == ["Blazing prose."]


def test_inline_exclusions_labels_and_table_body() -> None:
    text = (
        "Use `robust` and ``powerful`` at https://example.com/blazing "
        "or docs/robust.md.\n\n"
        "[Visible label](https://example.com) remains.\n\n"
        "| Name | Detail |\n|---|---|\n| Entry | Robust result. |\n"
    )
    assert [item[2] for item in paragraphs(text)] == [
        "Use and at or .",
        "Visible label remains.",
        "Entry",
        "Robust result.",
    ]


def test_uncited_quote_is_prose() -> None:
    assert paragraphs("> Robust prose.\n\nNext sentence.") == [
        (1, 1, "Robust prose."),
        (3, 3, "Next sentence."),
    ]


def test_selection_accepts_an_alternate_scope(tmp_path: Path) -> None:
    path = tmp_path / "other.md"
    path.write_text("Owned prose.\n")
    patch = (
        "diff --git a/other.md b/other.md\n--- a/other.md\n+++ b/other.md\n"
        "@@ -0,0 +1 @@\n+Owned prose.\n"
    )
    assert patch_selection(patch, tmp_path) == {}
    assert patch_selection(patch, tmp_path, lambda path: path.suffix == ".md") == {
        path: {1}
    }


@pytest.mark.parametrize(
    "excluded", ["```text\nrobust\n```", "> robust [Source](https://example.com)"]
)
def test_exclusions_terminate_without_blank_line(excluded: str) -> None:
    found = paragraphs(excluded + "\nBlazing prose.\n")
    assert found == [
        (
            len(excluded.splitlines()) + 1,
            len(excluded.splitlines()) + 1,
            "Blazing prose.",
        )
    ]


@pytest.mark.parametrize(
    "literal", ["```html\n<!--\n```", "~~~html\n<!--\n~~~", "`<!--`"]
)
def test_comment_openers_in_code_are_literals(literal: str) -> None:
    assert [item[2] for item in paragraphs(literal + "\n\nSafe prose.")] == [
        "Safe prose."
    ]


def test_sentences_preserve_wrapped_spans_and_separate_frozen_text() -> None:
    assert sentences("Frozen first.\nWrapped second\nsentence. Third sentence.") == [
        (1, 1, "Frozen first."),
        (2, 3, "Wrapped second sentence."),
        (3, 3, "Third sentence."),
    ]
