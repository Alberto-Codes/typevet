"""Tests for conservative terminology enforcement."""

from pathlib import Path
from subprocess import CompletedProcess

import pytest

from scripts import check_terminology as gate
from scripts import markdown_prose

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("term", ["issue-bus", "Issue-Bus", "ISSUE-BUS"])
def test_forbidden_spelling(tmp_path: Path, term: str) -> None:
    path = tmp_path / "sample.md"
    path.write_text(f"Use the {term}.\n")
    assert gate.main([str(path)]) == 1


@pytest.mark.parametrize(
    "term",
    [
        "issue bus",
        "Issue Bus",
        "issue-buses",
        "xissue-bus",
        "issue-bus2",
        "issue-bus_name",
    ],
)
def test_accepted_terms_and_boundaries(tmp_path: Path, term: str) -> None:
    path = tmp_path / "sample.md"
    path.write_text(f"Use {term}.\n")
    assert gate.main([str(path)]) == 0


@pytest.mark.parametrize(
    "text",
    [
        "```text\nissue-bus\n```",
        "~~~\nissue-bus\n~~~",
        "    issue-bus",
        "`issue-bus`",
        "> issue-bus [Source](https://example.com)",
        "---\nname: issue-bus\n---",
        "<!-- issue-bus -->",
        "https://example.com/issue-bus",
        "docs/issue-bus.md",
        "| issue-bus | Other |\n|---|---|",
    ],
)
def test_excluded_text(tmp_path: Path, text: str) -> None:
    path = tmp_path / "sample.md"
    path.write_text(text + "\n")
    assert gate.main([str(path)]) == 0


@pytest.mark.parametrize(
    "prefix", ["```\nissue-bus\n```\n", "> issue-bus [Source](https://example.com)\n"]
)
def test_immediate_post_exclusion_prose(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], prefix: str
) -> None:
    path = tmp_path / "sample.md"
    path.write_text(prefix + "Use the issue-bus.\n")
    assert gate.main([str(path)]) == 1
    line = prefix.count("\n") + 1
    assert (
        f"{path}:{line}: use 'issue bus' instead of 'issue-bus'"
        in capsys.readouterr().out
    )


@pytest.mark.parametrize(
    "text",
    [
        "> Use the issue-bus.",
        "[issue-bus](https://example.com)",
        "| Term |\n|---|\n| issue-bus |",
    ],
)
def test_visible_prose_is_checked(tmp_path: Path, text: str) -> None:
    path = tmp_path / "sample.md"
    path.write_text(text)
    assert gate.main([str(path)]) == 1


def _patch(path: str, start: int, added: str) -> str:
    return (
        f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n"
        f"@@ -{start} +{start} @@\n-Old text.\n+{added}\n"
    )


def test_owned_sentence_preserves_frozen_text_in_same_paragraph(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "docs/explanation/guide.md"
    path.parent.mkdir(parents=True)
    path.write_text("Frozen issue-bus wording.\nUse the issue bus.\n")
    diff = tmp_path / "owned.diff"
    diff.write_text(_patch("docs/explanation/guide.md", 2, "Use the issue bus."))
    monkeypatch.setattr(gate, "repository_root", lambda: tmp_path)
    assert gate.main(["--diff", str(diff)]) == 0
    assert "1 files, 1 owned paragraphs" in capsys.readouterr().out
    path.write_text("Frozen issue-bus wording.\nUse the issue-bus.\n")
    diff.write_text(_patch("docs/explanation/guide.md", 2, "Use the issue-bus."))
    assert gate.main(["--diff", str(diff)]) == 1
    assert f"{path}:2:" in capsys.readouterr().out


def test_owned_wrapped_sentence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "README.md").write_text("Use the issue-bus\nfor handoffs.\n")
    diff = tmp_path / "owned.diff"
    diff.write_text(_patch("README.md", 2, "for handoffs."))
    monkeypatch.setattr(gate, "repository_root", lambda: tmp_path)
    assert gate.main(["--diff", str(diff)]) == 1


@pytest.mark.parametrize(
    "path,expected",
    [
        ("README.md", True),
        ("CLAUDE.md", True),
        ("AGENTS.md", True),
        ("docs/explanation/guide.md", True),
        ("docs/tutorials/start.md", True),
        ("docs/adr/0001.md", True),
        ("docs/page.md", True),
        ("scratchpad/note.md", False),
        ("tests/fixture.md", False),
        ("docs/source.py", False),
    ],
)
def test_default_scope(path: str, expected: bool) -> None:
    assert gate.in_scope(Path(path)) is expected


def test_default_selection_and_clean_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "README.md").write_text("Frozen issue-bus.\nNew sentence.\n")
    new = tmp_path / "docs/tutorials/new.md"
    new.parent.mkdir(parents=True)
    new.write_text("Use the issue-bus.\n")
    patch = _patch("README.md", 2, "New sentence.")
    untracked = "docs/tutorials/new.md\0scratchpad/missing.md\0"

    def fake_run(command: list[str], **kwargs: object) -> CompletedProcess[str]:
        output = patch if "diff" in command else untracked
        return CompletedProcess(command, 0, stdout=output)

    monkeypatch.setattr(gate, "repository_root", lambda: tmp_path)
    monkeypatch.setattr(markdown_prose.subprocess, "run", fake_run)
    assert gate.main([]) == 1
    assert "2 files, 2 owned paragraphs" in capsys.readouterr().out
    patch = ""
    untracked = ""
    assert gate.main([]) == 0
    assert "0 files, 0 owned paragraphs" in capsys.readouterr().out


@pytest.mark.parametrize(
    "config",
    [
        "",
        "terms = []",
        "terms = 'wrong'",
        "terms = [1]",
        "terms = [{}]",
        "[terms]\nforbidden = 'issue-bus'\npreferred = 'issue bus'",
        "[[terms]]\nforbidden = 'issue-bus'",
        "malformed = [",
        "[[terms]]\nforbidden = ''\npreferred = 'issue bus'",
        "[[terms]]\nforbidden = 5\npreferred = 'issue bus'",
        "[[terms]]\nforbidden = ' issue-bus'\npreferred = 'issue bus'",
        "[[terms]]\nforbidden = 'issue-bus'\npreferred = 'absent term'",
        "[[terms]]\nforbidden = 'issue-bus'\npreferred = 'issue bus'\nextra = true",
        "extra = true\n[[terms]]\nforbidden = 'issue-bus'\npreferred = 'issue bus'",
        "[[terms]]\nforbidden = 'issue bus'\npreferred = 'worker'",
        "[[terms]]\nforbidden = 'bus'\npreferred = 'worker'",
        (
            "[[terms]]\nforbidden = 'issue-bus'\npreferred = 'issue bus'\n"
            "[[terms]]\nforbidden = 'ISSUE-BUS'\npreferred = 'worker'"
        ),
        (
            "[[terms]]\nforbidden = 'bad token'\npreferred = 'worker'\n"
            "[[terms]]\nforbidden = 'token'\npreferred = 'harness'"
        ),
    ],
)
def test_invalid_configuration(tmp_path: Path, config: str) -> None:
    path = tmp_path / "sample.md"
    path.write_text("Accepted prose.")
    config_path = tmp_path / "terms.toml"
    config_path.write_text(config)
    assert gate.main(["--config", str(config_path), str(path)]) == 2


def test_valid_override_and_glossary_headings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    glossary = tmp_path / "glossary.md"
    glossary.write_text(
        "# Glossary\n\n## Canonical heading\n\n**canonical entry.** Definition.\n"
    )
    config = tmp_path / "terms.toml"
    config.write_text(
        "[[terms]]\nforbidden = 'old heading'\npreferred = 'Canonical heading'\n"
        "[[terms]]\nforbidden = 'old entry'\npreferred = 'canonical entry'\n"
    )
    path = tmp_path / "sample.md"
    path.write_text("Old heading and old entry.")
    monkeypatch.setattr(gate, "GLOSSARY", glossary)
    assert gate.main(["--config", str(config), str(path)]) == 1
    path.write_text("Canonical heading and canonical entry.")
    assert gate.main(["--config", str(config), str(path)]) == 0


@pytest.mark.parametrize("kind", ["missing", "directory", "suffix", "encoding"])
def test_invalid_explicit_input(tmp_path: Path, kind: str) -> None:
    path = tmp_path / "file.md"
    if kind == "directory":
        path.mkdir()
    elif kind == "suffix":
        path = tmp_path / "file.txt"
        path.write_text("Text.")
    elif kind == "encoding":
        path.write_bytes(b"\xff")
    assert gate.main([str(path)]) == 2


def test_invalid_patch_and_mixed_modes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "README.md").write_text("Current text.")
    diff = tmp_path / "change.diff"
    diff.write_text(_patch("README.md", 1, "Different text."))
    monkeypatch.setattr(gate, "repository_root", lambda: tmp_path)
    assert gate.main(["--diff", str(diff)]) == 2
    assert gate.main(["--diff", str(diff), "README.md"]) == 2


def test_missing_config_and_glossary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "sample.md"
    path.write_text("Accepted prose.")
    assert gate.main(["--config", str(tmp_path / "missing.toml"), str(path)]) == 2
    monkeypatch.setattr(gate, "GLOSSARY", tmp_path / "missing.md")
    assert gate.main([str(path)]) == 2
