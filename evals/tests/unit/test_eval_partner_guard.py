"""CI guard: collections NBA partner paths must not enter public typevet."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from typevet_evals.datasets import partner_guard
from typevet_evals.datasets.partner_guard import (
    packaging_line_is_forbidden,
    path_is_forbidden,
    scan_packaging_config,
    scan_tracked_content,
    scan_tracked_paths,
    scan_tree_paths,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.unit
def test_path_is_forbidden_examples() -> None:
    assert path_is_forbidden("data/collections_nba/foo.jsonl")
    assert path_is_forbidden("fixtures/collections_nba/val.jsonl")
    assert not path_is_forbidden("docs/reference/eval-partner-data-policy.md")
    assert not path_is_forbidden("evals/src/typevet_evals/datasets/partner_guard.py")


@pytest.mark.unit
def test_packaging_line_skips_comments_and_blanks() -> None:
    assert not packaging_line_is_forbidden("")
    assert not packaging_line_is_forbidden("# collections_nba ok in comment")
    assert packaging_line_is_forbidden('include = ["data/collections_nba/**"]')


@pytest.mark.unit
def test_scan_tree_returns_empty_when_root_missing(tmp_path: Path) -> None:
    assert scan_tree_paths(tmp_path / "nope") == []


@pytest.mark.unit
def test_scan_packaging_config_flags_bad_line(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[tool.uv_build]\ninclude = ["data/collections_nba"]\n',
        encoding="utf-8",
    )
    hits = scan_packaging_config(tmp_path)
    assert len(hits) == 1
    assert hits[0].startswith("pyproject.toml:")


@pytest.mark.unit
def test_scan_tree_detects_partner_file(tmp_path: Path) -> None:
    bad = tmp_path / "data" / "collections_nba" / "val.jsonl"
    bad.parent.mkdir(parents=True)
    bad.write_text('{"id": "synthetic-only"}\n', encoding="utf-8")
    assert scan_tree_paths(tmp_path) == ["data/collections_nba/val.jsonl"]


@pytest.mark.unit
def test_public_repo_has_no_forbidden_tracked_paths() -> None:
    assert scan_tracked_paths(REPO_ROOT) == []


@pytest.mark.unit
def test_public_repo_has_no_forbidden_tracked_content() -> None:
    assert scan_tracked_content(REPO_ROOT) == []


@pytest.mark.unit
def test_packaging_config_excludes_partner_paths() -> None:
    assert scan_packaging_config(REPO_ROOT) == []


@pytest.mark.unit
def test_scan_packaging_config_flags_the_evals_member(tmp_path: Path) -> None:
    member = tmp_path / "evals"
    member.mkdir()
    (member / "pyproject.toml").write_text(
        '[tool.uv_build]\ninclude = ["data/collections_nba"]\n',
        encoding="utf-8",
    )
    hits = scan_packaging_config(tmp_path)
    assert hits == ['evals/pyproject.toml:2:include = ["data/collections_nba"]']


def _fake_tracked_files(
    monkeypatch: pytest.MonkeyPatch, root: Path, files: dict[str, str]
) -> None:
    """Write ``files`` under ``root`` and make ``git ls-files`` list them."""
    for rel, text in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    listing = b"".join(rel.encode() + b"\0" for rel in files)

    def _run(argv: list[str], **_: object) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(argv, 0, stdout=listing, stderr=b"")

    monkeypatch.setattr(partner_guard.subprocess, "run", _run)


@pytest.mark.unit
def test_import_marker_is_flagged_in_library_and_evals_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Isolate the import rule: with no path markers, only it can report a file.
    monkeypatch.setattr(partner_guard, "FORBIDDEN_PATH_MARKERS", ())
    bad = "import finvet.data.collections_nba\n"
    _fake_tracked_files(
        monkeypatch,
        tmp_path,
        {
            "src/typevet/bad.py": bad,
            "evals/src/typevet_evals/bad.py": bad,
            "evals/tests/unit/bad_test.py": bad,
            "docs/bad.md": bad,
            "evals/src/typevet_evals/clean.py": "import json\n",
        },
    )
    assert scan_tracked_content(tmp_path) == [
        "evals/src/typevet_evals/bad.py",
        "src/typevet/bad.py",
    ]
