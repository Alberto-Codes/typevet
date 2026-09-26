"""Acceptance tests for owned prose checks."""

from pathlib import Path
from subprocess import CompletedProcess

import pytest

from scripts import check_plain_english as gate
from scripts import markdown_prose

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("count,code", [(20, 0), (21, 1)])
def test_sentence_limit(tmp_path: Path, count: int, code: int) -> None:
    path = tmp_path / "sample.md"
    path.write_text(" ".join(["word"] * count) + ".\n")
    assert gate.main([str(path)]) == code


@pytest.mark.parametrize(
    "word", ["seamless", "Robust", "POWERFUL", "blazing", "cutting-edge"]
)
def test_banned_adjectives(tmp_path: Path, word: str) -> None:
    path = tmp_path / "sample.md"
    path.write_text(f"A {word} tool.\n")
    assert gate.main([str(path)]) == 1


def test_whole_words_and_counting(tmp_path: Path) -> None:
    path = tmp_path / "sample.md"
    path.write_text("Robustness matters! " + " ".join(["don't", "well-known"] * 10))
    assert gate.main([str(path)]) == 0


def _patch(path: str, start: int, added: list[str]) -> str:
    return (
        f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n"
        f"@@ -{start},0 +{start},{len(added)} @@\n"
        + "".join(f"+{line}\n" for line in added)
    )


def test_saved_diff_owns_wrapped_sentence_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "README.md"
    path.write_text("Frozen robust prose.\n\n" + "word " * 19 + "word\nlast.\n")
    patch = tmp_path / "change.diff"
    patch.write_text(_patch("README.md", 4, ["last."]))
    monkeypatch.setattr(gate, "repository_root", lambda: tmp_path)
    assert gate.main(["--diff", str(patch)]) == 1
    output = capsys.readouterr().out
    assert ":3: 21 words" in output
    assert "banned adjective" not in output
    path.write_text("Frozen robust prose.\n\nShort\nlast.\n")
    assert gate.main(["--diff", str(patch)]) == 0


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


@pytest.mark.parametrize(
    "patch",
    [
        "nonsense",
        "diff --git a/README.md b/README.md\n",
        "diff --git a/README.md b/README.md\nBinary files differ\n",
        _patch("README.md", 1, ["Other text."]),
        _patch("README.md", 1, ["Current text."]).replace("+Current", " Current"),
        _patch("../README.md", 1, ["Current text."]),
        _patch("README.md", 1, ["Current text."]).replace(",1 @@", ",2 @@"),
    ],
)
def test_invalid_patch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, patch: str
) -> None:
    (tmp_path / "README.md").write_text("Current text.\n")
    diff = tmp_path / "change.diff"
    diff.write_text(patch)
    monkeypatch.setattr(gate, "repository_root", lambda: tmp_path)
    assert gate.main(["--diff", str(diff)]) == 2


def test_deleted_file_has_no_owned_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    diff = tmp_path / "change.diff"
    diff.write_text(
        "diff --git a/README.md b/README.md\ndeleted file mode 100644\n"
        "--- a/README.md\n+++ /dev/null\n@@ -1 +0,0 @@\n-Robust text.\n"
    )
    monkeypatch.setattr(gate, "repository_root", lambda: tmp_path)
    assert gate.main(["--diff", str(diff)]) == 0


def test_default_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "README.md").write_text("Frozen robust.\n\nNew prose.\n")
    docs = tmp_path / "docs" / "reference"
    docs.mkdir(parents=True)
    (docs / "new.md").write_text("Powerful prose.\n")
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> object:
        calls.append(command)
        output = (
            _patch("README.md", 3, ["New prose."])
            if "diff" in command
            else "docs/reference/new.md\0scratchpad/skip.md\0"
        )
        return CompletedProcess(command, 0, stdout=output)

    monkeypatch.setattr(gate, "repository_root", lambda: tmp_path)
    monkeypatch.setattr(markdown_prose.subprocess, "run", fake_run)
    assert gate.main([]) == 1
    assert "2 files, 2 owned paragraphs" in capsys.readouterr().out
    assert len(calls) == 2


def test_clean_checkout_reports_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(gate, "repository_root", lambda: tmp_path)
    monkeypatch.setattr(gate, "changed_selection", lambda root: {})
    assert gate.main([]) == 0
    assert "0 files, 0 owned paragraphs" in capsys.readouterr().out


def test_modes_cannot_mix(tmp_path: Path) -> None:
    assert gate.main(["--diff", str(tmp_path / "x.diff"), "README.md"]) == 2


def test_unsupported_symlink_suffix(tmp_path: Path) -> None:
    source = tmp_path / "source.md"
    source.write_text("Fine prose.")
    alias = tmp_path / "alias.txt"
    alias.symlink_to(source)
    assert gate.main([str(alias)]) == 2


def test_patch_headers_must_agree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "README.md").write_text("Fine prose.\n")
    patch = tmp_path / "change.diff"
    patch.write_text(
        _patch("README.md", 1, ["Fine prose."]).replace(
            "a/README.md b/README.md", "a/other.md b/other.md"
        )
    )
    monkeypatch.setattr(gate, "repository_root", lambda: tmp_path)
    assert gate.main(["--diff", str(patch)]) == 2


@pytest.mark.parametrize("frozen", ["Robust frozen prose.", "word " * 24 + "word."])
def test_owned_sentence_preserves_frozen_sentence_in_same_paragraph(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, frozen: str
) -> None:
    (tmp_path / "README.md").write_text(f"{frozen}\nEdited short sentence.\n")
    patch = tmp_path / "change.diff"
    patch.write_text(_patch("README.md", 2, ["Edited short sentence."]))
    monkeypatch.setattr(gate, "repository_root", lambda: tmp_path)
    assert gate.main(["--diff", str(patch)]) == 0


def test_findings_locate_sentence_start(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "sample.md"
    path.write_text("First sentence.\nA robust sentence\ncontinues here.\n")
    assert gate.main([str(path)]) == 1
    assert f"{path}:2: banned adjective: robust" in capsys.readouterr().out


def test_heading_does_not_join_following_prose(tmp_path: Path) -> None:
    path = tmp_path / "sample.md"
    path.write_text("# " + "word " * 7 + "\n" + "word " * 14 + "word.\n")
    assert gate.main([str(path)]) == 0


def test_unclosed_comment_is_invalid(tmp_path: Path) -> None:
    path = tmp_path / "sample.md"
    path.write_text("<!-- missing close\nBlazing prose.\n")
    assert gate.main([str(path)]) == 2


@pytest.mark.parametrize(
    "section",
    [
        (
            "diff --git a/image.png b/image.png\nindex 1234567..7654321 100644\n"
            "Binary files a/image.png and b/image.png differ\n"
        ),
        "diff --git a/empty.py b/empty.py\nnew file mode 100644\nindex 0000000..e69de29\n",
        "diff --git a/script.py b/script.py\nold mode 100644\nnew mode 100755\n",
        "diff --git a/scratchpad/note.md b/scratchpad/note.md\nold mode 100644\nnew mode 100755\n",
        "diff --git a/README.md b/README.md\nold mode 100644\nnew mode 100755\n",
    ],
)
def test_normal_metadata_and_outside_scope_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, section: str
) -> None:
    (tmp_path / "README.md").write_text("Robust frozen prose.\n")
    patch = tmp_path / "change.diff"
    patch.write_text(section)
    monkeypatch.setattr(gate, "repository_root", lambda: tmp_path)
    assert gate.main(["--diff", str(patch)]) == 0


def test_owned_empty_file_patch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "README.md"
    path.write_text("")
    patch = tmp_path / "change.diff"
    patch.write_text(
        "diff --git a/README.md b/README.md\nnew file mode 100644\nindex 0000000..e69de29\n"
    )
    monkeypatch.setattr(gate, "repository_root", lambda: tmp_path)
    assert gate.main(["--diff", str(patch)]) == 0
    path.write_text("Actual prose.\n")
    assert gate.main(["--diff", str(patch)]) == 2
