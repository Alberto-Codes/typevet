"""CI guard: collections NBA partner paths must not enter public typevet."""

from __future__ import annotations

from pathlib import Path

import pytest

from typevet.eval_partner_guard import (
    packaging_line_is_forbidden,
    path_is_forbidden,
    scan_packaging_config,
    scan_tracked_content,
    scan_tracked_paths,
    scan_tree_paths,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.unit
def test_path_is_forbidden_examples() -> None:
    assert path_is_forbidden("data/collections_nba/foo.jsonl")
    assert path_is_forbidden("fixtures/collections_nba/val.jsonl")
    assert not path_is_forbidden("docs/reference/eval-partner-data-policy.md")
    assert not path_is_forbidden("src/typevet/eval_partner_guard.py")


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
