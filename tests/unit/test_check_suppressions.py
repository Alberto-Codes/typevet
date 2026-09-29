"""Tests for scripts/check_suppressions.py."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

from scripts.check_suppressions import (
    ALLOWED_PER_FILE_IGNORE_CODES,
    count_per_file_ignores,
    main,
    scan,
)


class TestCountPerFileIgnores:
    """Tests for the count_per_file_ignores function."""

    def test_empty_file_returns_zero(self) -> None:
        """An empty pyproject.toml has no per-file-ignores codes."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".toml", delete=False) as f:
            f.write("")
            f.flush()
            total, per_pattern = count_per_file_ignores(Path(f.name))
        assert total == 0
        assert per_pattern == {}

    def test_current_pyproject_toml_budget(self) -> None:
        """The actual pyproject.toml matches the deliberate ignore budget."""
        total, per_pattern = count_per_file_ignores(Path("pyproject.toml"))
        assert total == ALLOWED_PER_FILE_IGNORE_CODES == 11
        assert len(per_pattern) == 4
        assert len(per_pattern["**/tests/**/*.py"]) == 7
        assert len(per_pattern["evals/src/typevet_evals/wheel_isolated.py"]) == 1
        assert len(per_pattern["scripts/check_commit_msg.py"]) == 1
        assert len(per_pattern["src/typevet/evaluation/datasets/partner_guard.py"]) == 2

    def test_main_fails_when_budget_exceeded(self) -> None:
        """main() returns 1 when per-file-ignores exceeds the budget."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            pyproject = tmpdir_path / "pyproject.toml"

            over = ALLOWED_PER_FILE_IGNORE_CODES + 1
            codes = "".join(f'    "CODE{i:03d}",\n' for i in range(over))
            pyproject.write_text(
                "[tool.ruff.lint.per-file-ignores]\n"
                '"tests/**/*.py" = [\n' + codes + "]\n"
            )

            total, _ = count_per_file_ignores(pyproject)
            assert total == ALLOWED_PER_FILE_IGNORE_CODES + 1

            original_pyproject = Path("pyproject.toml")
            backup = tmpdir_path / "pyproject.toml.bak"
            if original_pyproject.exists():
                shutil.move(str(original_pyproject), str(backup))

            try:
                shutil.copy(pyproject, "pyproject.toml")
                result = main([])
                assert result == 1
            finally:
                if backup.exists():
                    shutil.move(str(backup), str(original_pyproject))


class TestScan:
    """Tests for the scan function."""

    def test_planted_suppression_in_code_line_is_flagged(self) -> None:
        """A # noqa comment on a code line is flagged."""
        with tempfile.TemporaryDirectory() as tmpdir:
            script_dir = Path(tmpdir) / "scripts"
            script_dir.mkdir()
            probe_file = script_dir / "_probe_tmp.py"
            probe_file.write_text("import os  # noqa: E501\n")

            findings, files_scanned = scan([script_dir])

            assert files_scanned == 1
            assert len(findings) == 1
            assert "_probe_tmp.py" in findings[0]

    def test_suppression_in_string_literal_is_not_flagged(self) -> None:
        """A # noqa text in a string literal is not flagged."""
        with tempfile.TemporaryDirectory() as tmpdir:
            script_dir = Path(tmpdir) / "scripts"
            script_dir.mkdir()
            probe_file = script_dir / "_probe_tmp.py"
            probe_file.write_text('S = "this mentions # noqa: E501 in prose"\n')

            findings, files_scanned = scan([script_dir])

            assert files_scanned == 1
            assert len(findings) == 0

    def test_main_on_clean_dir_returns_zero(self) -> None:
        """main() returns 0 over a file with no suppression."""
        with tempfile.TemporaryDirectory() as tmpdir:
            script_dir = Path(tmpdir)
            (script_dir / "_probe_tmp.py").write_text("import os\n")
            assert main([tmpdir]) == 0


@pytest.mark.parametrize(
    "directive",
    [
        "# ty: ignore",
        "# ty: ignore[invalid-assignment]",
    ],
)
def test_ty_comment_rejected(directive: str, tmp_path: Path) -> None:
    """Reject real ty comments."""
    path = tmp_path / "canary.py"
    path.write_text(f"value: str = 1  {directive}\n")
    findings, count = scan([path])
    assert count == 1
    assert len(findings) == 1
    assert main([str(path)]) == 1
