"""Offline contract: ``read_baseline_commit`` across git on-disk layouts (#209).

Each test builds one layout under ``tmp_path``: a linked worktree whose
``.git`` is a ``gitdir:`` file with a ``commondir``, a branch stored only in
``packed-refs``, and a worktree whose branch is packed in the common dir.
The reader must return the ``HEAD`` commit hex, and ``unknown`` when no
layout resolves.

Examples:
    ```bash
    uv run pytest -q evals/tests/contract/test_experiment_identity_git_layouts.py
    ```

See Also:
    - [typevet_evals.experiment_identity][]: the reader under test
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from typevet_evals.experiment_identity import read_baseline_commit

pytestmark = pytest.mark.contract

_COMMIT = "a" * 40
_OTHER = "b" * 40


def _packed_refs(*lines: str) -> str:
    return "# pack-refs with: peeled fully-peeled sorted\n" + "\n".join(lines) + "\n"


def _linked_worktree(tmp_path: Path, head: str) -> tuple[Path, Path, Path]:
    common = tmp_path / "main" / ".git"
    gitdir = common / "worktrees" / "wt"
    gitdir.mkdir(parents=True)
    (gitdir / "HEAD").write_text(head, encoding="utf-8")
    (gitdir / "commondir").write_text("../..\n", encoding="utf-8")
    worktree = tmp_path / "wt"
    worktree.mkdir()
    (worktree / ".git").write_text(f"gitdir: {gitdir}\n", encoding="utf-8")
    return worktree, gitdir, common


def test_worktree_gitdir_file_resolves_loose_ref_in_common_dir(tmp_path: Path) -> None:
    """A worktree branch stored as a loose ref in the common dir resolves."""
    worktree, _, common = _linked_worktree(tmp_path, "ref: refs/heads/feature\n")
    (common / "refs" / "heads").mkdir(parents=True)
    (common / "refs" / "heads" / "feature").write_text(_COMMIT + "\n", "utf-8")

    assert read_baseline_commit(worktree) == _COMMIT


def test_worktree_gitdir_relative_path_and_detached_head(tmp_path: Path) -> None:
    """A relative ``gitdir:`` path with a detached ``HEAD`` resolves."""
    gitdir = tmp_path / "main" / ".git" / "worktrees" / "wt"
    gitdir.mkdir(parents=True)
    (gitdir / "HEAD").write_text(_COMMIT + "\n", encoding="utf-8")
    worktree = tmp_path / "wt"
    worktree.mkdir()
    (worktree / ".git").write_text(
        "gitdir: ../main/.git/worktrees/wt\n", encoding="utf-8"
    )

    assert read_baseline_commit(worktree) == _COMMIT


def test_branch_only_in_packed_refs(tmp_path: Path) -> None:
    """A branch present only in ``packed-refs`` resolves past peel lines."""
    git = tmp_path / ".git"
    git.mkdir()
    (git / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (git / "packed-refs").write_text(
        _packed_refs(
            f"{_OTHER} refs/heads/other",
            f"{_COMMIT} refs/heads/main",
            f"^{_OTHER}",
        ),
        encoding="utf-8",
    )

    assert read_baseline_commit(tmp_path) == _COMMIT


def test_worktree_branch_packed_in_common_dir(tmp_path: Path) -> None:
    """A worktree branch packed in the common dir ``packed-refs`` resolves."""
    worktree, _, common = _linked_worktree(tmp_path, "ref: refs/heads/feature\n")
    (common / "packed-refs").write_text(
        _packed_refs(f"{_COMMIT} refs/heads/feature"), encoding="utf-8"
    )

    assert read_baseline_commit(worktree) == _COMMIT


def test_loose_ref_in_gitdir_wins_over_common_dir(tmp_path: Path) -> None:
    """A loose ref in the worktree gitdir takes precedence over the common dir."""
    worktree, gitdir, common = _linked_worktree(tmp_path, "ref: refs/heads/feature\n")
    (gitdir / "refs" / "heads").mkdir(parents=True)
    (gitdir / "refs" / "heads" / "feature").write_text(_COMMIT + "\n", "utf-8")
    (common / "refs" / "heads").mkdir(parents=True)
    (common / "refs" / "heads" / "feature").write_text(_OTHER + "\n", "utf-8")

    assert read_baseline_commit(worktree) == _COMMIT


@pytest.mark.parametrize(
    "dot_git",
    ["not a gitdir line\n", "gitdir: missing/dir\n"],
)
def test_unresolvable_gitdir_file_returns_unknown(tmp_path: Path, dot_git: str) -> None:
    """A malformed or dangling ``.git`` file yields ``unknown``."""
    (tmp_path / ".git").write_text(dot_git, encoding="utf-8")

    assert read_baseline_commit(tmp_path) == "unknown"


def test_missing_ref_everywhere_returns_unknown(tmp_path: Path) -> None:
    """A branch absent from loose refs and ``packed-refs`` yields ``unknown``."""
    git = tmp_path / ".git"
    git.mkdir()
    (git / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (git / "packed-refs").write_text(
        _packed_refs(f"{_OTHER} refs/heads/other"), encoding="utf-8"
    )

    assert read_baseline_commit(tmp_path) == "unknown"


def _plain_repo(tmp_path: Path, head: str) -> Path:
    git = tmp_path / ".git"
    git.mkdir()
    (git / "HEAD").write_text(head, encoding="utf-8")
    return git


def test_binary_dot_git_file_returns_unknown(tmp_path: Path) -> None:
    """A ``.git`` file that is not UTF-8 yields ``unknown``, not an error."""
    (tmp_path / ".git").write_bytes(b"\xff\xfe\x00gitdir")

    assert read_baseline_commit(tmp_path) == "unknown"


def test_binary_packed_refs_returns_unknown(tmp_path: Path) -> None:
    """A ``packed-refs`` file that is not UTF-8 yields ``unknown``."""
    git = _plain_repo(tmp_path, "ref: refs/heads/main\n")
    (git / "packed-refs").write_bytes(b"\xff\xfe\x00 refs/heads/main\n")

    assert read_baseline_commit(tmp_path) == "unknown"


def test_binary_commondir_returns_unknown(tmp_path: Path) -> None:
    """A ``commondir`` file that is not UTF-8 yields ``unknown``."""
    worktree, gitdir, _ = _linked_worktree(tmp_path, "ref: refs/heads/feature\n")
    (gitdir / "commondir").write_bytes(b"\xff\xfe\x00")

    assert read_baseline_commit(worktree) == "unknown"


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores file modes")
def test_unreadable_packed_refs_returns_unknown(tmp_path: Path) -> None:
    """A ``packed-refs`` file without read permission yields ``unknown``."""
    git = _plain_repo(tmp_path, "ref: refs/heads/main\n")
    packed = git / "packed-refs"
    packed.write_text(_packed_refs(f"{_COMMIT} refs/heads/main"), encoding="utf-8")
    packed.chmod(0)
    try:
        assert read_baseline_commit(tmp_path) == "unknown"
    finally:
        packed.chmod(stat.S_IRUSR | stat.S_IWUSR)


def _worktree_with_common_packed_ref(tmp_path: Path) -> tuple[Path, Path]:
    worktree, gitdir, common = _linked_worktree(tmp_path, "ref: refs/heads/feature\n")
    (common / "packed-refs").write_text(
        _packed_refs(f"{_COMMIT} refs/heads/feature"), encoding="utf-8"
    )
    return worktree, gitdir / "packed-refs"


def test_binary_gitdir_packed_refs_stops_common_dir_fallback(tmp_path: Path) -> None:
    """A non-UTF-8 gitdir ``packed-refs`` yields ``unknown``, not the common ref."""
    worktree, packed = _worktree_with_common_packed_ref(tmp_path)
    packed.write_bytes(b"\xff\xfe\x00 refs/heads/feature\n")

    assert read_baseline_commit(worktree) == "unknown"


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores file modes")
def test_unreadable_gitdir_packed_refs_stops_common_dir_fallback(
    tmp_path: Path,
) -> None:
    """An unreadable gitdir ``packed-refs`` yields ``unknown``, not the common ref."""
    worktree, packed = _worktree_with_common_packed_ref(tmp_path)
    packed.write_text(_packed_refs(f"{_OTHER} refs/heads/other"), encoding="utf-8")
    packed.chmod(0)
    try:
        assert read_baseline_commit(worktree) == "unknown"
    finally:
        packed.chmod(stat.S_IRUSR | stat.S_IWUSR)


@pytest.mark.parametrize(
    "head",
    ["", "\n", "not-a-commit\n", "a" * 39 + "\n", "A" * 40 + "\n"],
)
def test_head_that_is_not_a_commit_hex_returns_unknown(
    tmp_path: Path, head: str
) -> None:
    """An empty or non-hex detached ``HEAD`` yields ``unknown``."""
    _plain_repo(tmp_path, head)

    assert read_baseline_commit(tmp_path) == "unknown"


def test_sha256_detached_head_resolves(tmp_path: Path) -> None:
    """A 64-hex SHA-256 object name is a valid commit."""
    _plain_repo(tmp_path, "c" * 64 + "\n")

    assert read_baseline_commit(tmp_path) == "c" * 64


def test_loose_ref_that_is_not_a_commit_hex_returns_unknown(tmp_path: Path) -> None:
    """A loose ref whose content is not hex yields ``unknown``."""
    git = _plain_repo(tmp_path, "ref: refs/heads/main\n")
    (git / "refs" / "heads").mkdir(parents=True)
    (git / "refs" / "heads" / "main").write_text("garbage\n", encoding="utf-8")

    assert read_baseline_commit(tmp_path) == "unknown"


@pytest.mark.parametrize(
    "ref",
    ["../secret", "refs/../../secret", "config", "{abs}"],
)
def test_ref_outside_refs_namespace_returns_unknown(tmp_path: Path, ref: str) -> None:
    """A symbolic ref outside ``refs/`` or with ``..`` is never read."""
    secret = tmp_path / "secret"
    git = _plain_repo(tmp_path, f"ref: {ref.replace('{abs}', str(secret))}\n")
    (git / "refs").mkdir()
    secret.write_text(_COMMIT + "\n", encoding="utf-8")
    (git / "config").write_text(_COMMIT + "\n", encoding="utf-8")

    assert read_baseline_commit(tmp_path) == "unknown"


@pytest.mark.parametrize(
    "decoy",
    ["# refs/heads/main", f"^{_OTHER} refs/heads/main"],
)
def test_packed_refs_skips_comment_and_peel_lines(tmp_path: Path, decoy: str) -> None:
    """Comment and peel lines never supply the hex, even when they name the ref."""
    git = _plain_repo(tmp_path, "ref: refs/heads/main\n")
    (git / "packed-refs").write_text(
        _packed_refs(decoy, f"{_COMMIT} refs/heads/main"), encoding="utf-8"
    )

    assert read_baseline_commit(tmp_path) == _COMMIT
