"""Wheel helpers keep their uv subprocesses out of the shared uv cache (#268)."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest

from typevet_evals import wheel_isolated

SESSION_CACHE_ENV = "TYPEVET_TEST_UV_CACHE_DIR"


def _default_uv_cache() -> Path:
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg) if xdg else Path.home() / ".cache"
    return (base / "uv").resolve()


def _invoke_each_helper(tmp_path: Path) -> list[Callable[[], object]]:
    wheel = tmp_path / "typevet-0-py3-none-any.whl"
    extra = tmp_path / "typevet_evals-0-py3-none-any.whl"
    return [
        lambda: wheel_isolated.build_wheel_to_directory(tmp_path / "a"),
        lambda: wheel_isolated.build_member_wheel_to_directory(tmp_path / "b"),
        lambda: wheel_isolated.run_isolated_wheel_python(
            wheel=wheel, source="pass", cwd=tmp_path, extra_wheels=(extra,)
        ),
    ]


def _capture_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[list[str], dict[str, Any]]]:
    calls: list[tuple[list[str], dict[str, Any]]] = []

    def _run(argv: Sequence[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append((list(argv), kwargs))
        return subprocess.CompletedProcess(list(argv), 0, stdout="", stderr="")

    monkeypatch.setattr(wheel_isolated.subprocess, "run", _run)
    return calls


def _cache_dir_arg(argv: list[str]) -> str | None:
    if "--cache-dir" not in argv:
        return None
    return argv[argv.index("--cache-dir") + 1]


@pytest.mark.unit
def test_each_helper_names_the_session_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session_cache = tmp_path / "session-uv-cache"
    monkeypatch.setenv(SESSION_CACHE_ENV, str(session_cache))
    calls = _capture_calls(monkeypatch)

    for invoke in _invoke_each_helper(tmp_path):
        invoke()

    assert len(calls) == 3
    for argv, _ in calls:
        assert _cache_dir_arg(argv) == str(session_cache), argv
        assert Path(str(_cache_dir_arg(argv))).resolve() != _default_uv_cache()
        assert "--no-cache" not in argv


@pytest.mark.unit
def test_each_helper_uses_no_cache_without_a_session_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(SESSION_CACHE_ENV, raising=False)
    calls = _capture_calls(monkeypatch)

    for invoke in _invoke_each_helper(tmp_path):
        invoke()

    assert len(calls) == 3
    for argv, _ in calls:
        assert "--no-cache" in argv, argv
        assert _cache_dir_arg(argv) is None


@pytest.mark.unit
def test_helpers_leave_the_caller_uv_cache_dir_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    caller_cache = str(tmp_path / "caller-cache")
    monkeypatch.setenv("UV_CACHE_DIR", caller_cache)
    monkeypatch.setenv(SESSION_CACHE_ENV, str(tmp_path / "session-uv-cache"))
    calls = _capture_calls(monkeypatch)

    for invoke in _invoke_each_helper(tmp_path):
        invoke()

    assert os.environ["UV_CACHE_DIR"] == caller_cache
    for _, kwargs in calls:
        assert "env" not in kwargs


@pytest.mark.unit
def test_pytest_session_sets_a_private_existing_uv_cache() -> None:
    session_cache = os.environ.get(SESSION_CACHE_ENV)

    assert session_cache, f"{SESSION_CACHE_ENV} is not set for this pytest session"
    path = Path(session_cache)
    assert path.is_dir()
    assert path.resolve() != _default_uv_cache()
    assert _default_uv_cache() not in path.resolve().parents


@pytest.mark.unit
def test_private_uv_cache_creates_sets_and_removes_its_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(SESSION_CACHE_ENV, raising=False)

    with wheel_isolated.private_uv_cache(tmp_path) as cache:
        assert cache.is_dir()
        assert cache.parent == tmp_path
        assert os.environ[SESSION_CACHE_ENV] == str(cache)
        assert wheel_isolated.uv_cache_args() == ("--cache-dir", str(cache))
        locked = cache / "archive-v0" / "entry"
        locked.mkdir(parents=True)
        (locked / "file.py").write_text("x = 1\n", encoding="utf-8")
        (locked / "file.py").chmod(0o444)
        locked.chmod(0o555)

    assert not cache.exists()
    assert SESSION_CACHE_ENV not in os.environ


@pytest.mark.unit
def test_private_uv_cache_reuses_a_cache_that_is_already_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    outer = tmp_path / "outer-cache"
    outer.mkdir()
    monkeypatch.setenv(SESSION_CACHE_ENV, str(outer))

    with wheel_isolated.private_uv_cache(tmp_path) as cache:
        assert cache == outer

    assert outer.is_dir()
    assert os.environ[SESSION_CACHE_ENV] == str(outer)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["outer-cache"]
