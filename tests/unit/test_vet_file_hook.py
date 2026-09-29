"""Unit tests for scripts/vet_file.sh, the Claude Code PostToolUse hook.

Each test builds a throwaway git project under ``tmp_path`` and runs the
hook as Claude Code does: hook JSON on stdin, ``CLAUDE_PROJECT_DIR`` set.
``UV_PROJECT`` points ``uv run`` at this checkout's environment, so the
hook finds ruff and docvet with no sync and no network.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Mapping
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
UNDOCUMENTED = "def f(x):\n    return x\n"
PYPROJECT = (
    '[project]\nname = "demo"\nversion = "0"\n\n'
    '[tool.docvet]\nfail-on = ["presence"]\nexclude = ["tests", "scripts"]\n'
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(
        shutil.which("uv") is None or shutil.which("jq") is None,
        reason="the hook needs uv and jq",
    ),
]


def _env_without_git(**extra: str) -> dict[str, str]:
    """Copy os.environ without inherited GIT_* variables, then add *extra*.

    A git hook (pre-commit) exports GIT_DIR, GIT_INDEX_FILE and more. Left
    in place, they point the throwaway repos here at the outer repository.
    """
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(extra)
    return env


GIT_ENV = _env_without_git(
    GIT_AUTHOR_NAME="t",
    GIT_AUTHOR_EMAIL="t@example.invalid",
    GIT_COMMITTER_NAME="t",
    GIT_COMMITTER_EMAIL="t@example.invalid",
)


def _project(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["/usr/bin/env", "git", "init", "-q"],
        cwd=root,
        env=GIT_ENV,
        check=True,
        capture_output=True,
    )
    (root / "pyproject.toml").write_text(PYPROJECT, encoding="utf-8")
    return root


def _put(path: Path, text: str = UNDOCUMENTED) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _run_hook(
    payload: Mapping[str, object],
    project_dir: Path,
    git_env: Mapping[str, str] | None = None,
) -> str:
    env = _env_without_git(
        CLAUDE_PROJECT_DIR=str(project_dir),
        UV_PROJECT=str(REPO),
        UV_NO_SYNC="1",
        UV_OFFLINE="1",
    )
    env.update(git_env or {})
    env.pop("VIRTUAL_ENV", None)
    done = subprocess.run(
        ["/usr/bin/env", "bash", "scripts/vet_file.sh"],
        input=json.dumps(payload),
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert done.returncode == 0, done.stderr
    if not done.stdout.strip():
        return ""
    return json.loads(done.stdout)["hookSpecificOutput"]["additionalContext"]


def _write(path: Path) -> dict[str, object]:
    return {"tool_name": "Write", "tool_input": {"file_path": str(path)}}


def test_docvet_skips_a_file_the_docvet_exclude_list_names(tmp_path: Path) -> None:
    root = _project(tmp_path / "main")
    target = _put(root / "tests" / "test_demo.py")

    context = _run_hook(_write(target), root)

    assert "docvet:" not in context


def test_docvet_still_reports_an_included_src_file(tmp_path: Path) -> None:
    root = _project(tmp_path / "main")
    target = _put(root / "src" / "demo" / "mod.py")

    context = _run_hook(_write(target), root)

    assert "== ./src/demo/mod.py" in context
    assert "docvet:" in context
    assert "missing-docstring" in context


def test_hook_ignores_git_variables_inherited_from_an_outer_repo(
    tmp_path: Path,
) -> None:
    outer = _project(tmp_path / "outer")
    root = _project(tmp_path / "main")
    target = _put(root / "src" / "demo" / "mod.py")
    commit_env = {"GIT_DIR": str(outer / ".git"), "GIT_INDEX_FILE": ".git/index"}

    context = _run_hook(_write(target), root, commit_env)

    assert "== ./src/demo/mod.py" in context
    assert "missing-docstring" in context


def test_ruff_still_checks_an_excluded_tests_file(tmp_path: Path) -> None:
    root = _project(tmp_path / "main")
    target = _put(root / "tests" / "test_demo.py", "import os\nx = 1\n")

    context = _run_hook(_write(target), root)

    assert "ruff check:" in context
    assert "F401" in context
    assert "docvet:" not in context


def test_bash_edit_in_worktree_vets_worktree_files_only(tmp_path: Path) -> None:
    root = _project(tmp_path / "main")
    subprocess.run(
        ["/usr/bin/env", "git", "add", "-A"],
        cwd=root,
        env=GIT_ENV,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["/usr/bin/env", "git", "commit", "-qm", "init"],
        cwd=root,
        env=GIT_ENV,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["/usr/bin/env", "git", "worktree", "add", "-q", "-b", "wt", ".claude/wt"],
        cwd=root,
        env=GIT_ENV,
        check=True,
        capture_output=True,
    )
    _put(root / "src" / "demo" / "main_only.py")
    tree = root / ".claude" / "wt"
    target = _put(tree / "src" / "demo" / "worktree_only.py")
    payload = {"tool_name": "Bash", "cwd": str(tree), "duration_ms": 5000}

    context = _run_hook(payload, root)

    assert "worktree_only.py" in context
    assert "docvet:" in context
    assert "main_only.py" not in context
    assert str(target) in context
