"""Unit tests for scripts/check_commit_msg.py."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from scripts import check_commit_msg as gate


def _msg(subject: str, *body: str) -> str:
    if not body:
        return subject + "\n"
    return subject + "\n\n" + "\n".join(body) + "\n"


@pytest.mark.unit
@pytest.mark.parametrize("commit_type", ["style", "revert"])
def test_documented_types_style_and_revert_accepted(commit_type: str) -> None:
    text = _msg(f"{commit_type}(scope): tidy formatting for #85", "Refs #85")
    found, _ = gate.problems(text)
    assert found == []


@pytest.mark.unit
def test_invented_commit_type_rejected() -> None:
    text = _msg("wip(api): scratch work for #85", "Refs #85")
    found, _ = gate.problems(text)
    assert any("type 'wip'" in problem for problem in found)


@pytest.mark.unit
def test_forbidden_cursor_coauthor_trailer_rejected() -> None:
    text = _msg(
        "feat(api): add helper for #85",
        "Closes #85",
        "Generated-By: composer-2.5-fast (via Cursor Task)",
        "Co-authored-by: Cursor <cursoragent@cursor.com>",
    )
    found, _ = gate.problems(text)
    assert any("Co-Authored-By must not name a model or harness" in p for p in found)


@pytest.mark.unit
def test_forbidden_coauthor_email_case_insensitive() -> None:
    text = _msg(
        "fix(gates): align validation for #85",
        "Refs #85",
        "Co-Authored-By: Cursor <cursoragent@CURSOR.COM>",
    )
    found, _ = gate.problems(text)
    assert found


@pytest.mark.unit
def test_repaired_message_with_evidence_trailers_passes() -> None:
    text = _msg(
        "feat(api): add helper for #85",
        "Closes #85",
        "Specified-By: supervisor (via Cursor Task)",
        "Generated-By: composer-2.5-fast (via Cursor Task)",
    )
    found, _ = gate.problems(text)
    assert found == []


@pytest.mark.unit
def test_legitimate_human_coauthor_passes() -> None:
    text = _msg(
        "docs: note pairing workflow for #85",
        "Refs #85",
        "Co-Authored-By: Alex Example <alex.example@example.com>",
    )
    found, _ = gate.problems(text)
    assert found == []


@pytest.mark.unit
def test_main_file_path_strips_forbidden_coauthor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(gate, "check_author_for_commit_msg", lambda: None)
    path = tmp_path / "COMMIT_EDITMSG"
    path.write_text(
        textwrap.dedent(
            """\
            feat(gates): align commit validation for #85

            Closes #85

            Co-authored-by: Cursor <cursoragent@cursor.com>
            """
        ),
        encoding="utf-8",
    )
    assert gate.main([str(path)]) == 0
    assert "cursoragent" not in path.read_text(encoding="utf-8").casefold()


@pytest.mark.unit
def test_main_range_path_rejects_forbidden_coauthor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bad = textwrap.dedent(
        """\
        feat(gates): align commit validation for #85

        Closes #85

        Co-authored-by: Cursor <cursoragent@cursor.com>
        """
    )

    monkeypatch.setattr(
        gate,
        "messages_in_range",
        lambda _rev: [("deadbeef0", bad, "deadbeef0123456789deadbeef0123456789")],
    )
    monkeypatch.setattr(gate, "check_author_in_range", lambda _rev: [])
    monkeypatch.setattr(gate, "is_coauthor_grandfathered", lambda _sha: False)

    assert gate.main(["--range", "origin/main..HEAD"]) == 1


@pytest.mark.unit
def test_main_range_path_passes_repaired_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    good = textwrap.dedent(
        """\
        feat(gates): align commit validation for #85

        Closes #85

        Generated-By: composer-2.5-fast (via Cursor Task)
        """
    )

    monkeypatch.setattr(
        gate,
        "messages_in_range",
        lambda _rev: [("deadbeef0", good, "deadbeef0123456789deadbeef0123456789")],
    )
    monkeypatch.setattr(gate, "check_author_in_range", lambda _rev: [])

    assert gate.main(["--range", "origin/main..HEAD"]) == 0


@pytest.mark.unit
def test_strip_harness_coauthors_removes_cursor_trailer() -> None:
    text = _msg(
        "fix(gates): align validation for #85",
        "Closes #85",
        "Generated-By: test",
        "Co-authored-by: Cursor <cursoragent@cursor.com>",
    )
    stripped = gate.strip_harness_coauthors(text)
    assert "cursoragent" not in stripped.casefold()
    found, _ = gate.problems(stripped)
    assert found == []


@pytest.mark.unit
def test_main_range_grandfather_warns_on_historical_coauthor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bad = _msg(
        "feat(gates): align commit validation for #85",
        "Closes #85",
        "Co-authored-by: Cursor <cursoragent@cursor.com>",
    )
    monkeypatch.setattr(
        gate,
        "messages_in_range",
        lambda _rev: [
            ("abc123456", bad, "abc1234567890123456789012345678901234567890")
        ],
    )
    monkeypatch.setattr(gate, "check_author_in_range", lambda _rev: [])
    monkeypatch.setattr(gate, "is_coauthor_grandfathered", lambda _sha: True)

    assert gate.main(["--range", "HEAD~8..HEAD"]) == 0
