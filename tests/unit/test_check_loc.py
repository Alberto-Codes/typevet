"""Unit tests for scripts/check_loc.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts import check_loc as gate


def _module_with_function_body(*, code_lines: int) -> str:
    body = "\n".join(f"    _ = {i}" for i in range(code_lines))
    return f"def chunky():\n{body}\n"


@pytest.mark.unit
def test_main_fails_when_function_body_exceeds_limit(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "chunky.py").write_text(
        _module_with_function_body(code_lines=51), encoding="utf-8"
    )
    assert gate.main([str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert "FAIL" in out
    assert "chunky" in out
    assert "function limit 50" in out


@pytest.mark.unit
def test_main_passes_when_function_body_at_limit(tmp_path: Path) -> None:
    (tmp_path / "ok.py").write_text(
        _module_with_function_body(code_lines=50), encoding="utf-8"
    )
    assert gate.main([str(tmp_path)]) == 0


@pytest.mark.unit
def test_count_function_lines_matches_body_only(tmp_path: Path) -> None:
    source = '''"""Module doc."""

def outer():
    """Not counted as body code."""
    x = 1
    y = 2
'''
    path = tmp_path / "sample.py"
    path.write_text(source, encoding="utf-8")
    assert gate.count_function_lines(path) == [("outer", 2)]
