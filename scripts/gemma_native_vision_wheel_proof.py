r"""Wheel-isolated smoke for ``open_gemma_native_vision_judgment`` ([#177][i177]).

Examples:
    ```console
    $ uv run python scripts/gemma_native_vision_wheel_proof.py
    ```
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from typevet.testing.wheel_isolated import (
    build_wheel_to_directory,
    run_isolated_wheel_python,
)


def run_offline_wheel_smoke(*, work_dir: Path | None = None) -> tuple[int, Path]:
    """Build wheel and run factory smoke inside an isolated environment.

    Returns:
        ``(exit_code, wheel_path)``.
    """
    base = work_dir or Path(tempfile.mkdtemp(prefix="typevet-gemma-wheel-"))
    dist = base / "dist"
    isolated_cwd = base / "isolated_cwd"
    isolated_cwd.mkdir(parents=True, exist_ok=True)
    build_wheel_to_directory(dist)
    wheels = sorted(dist.glob("typevet-*.whl"))
    if len(wheels) != 1:
        print(f"FAIL_CLOSED: expected one wheel under {dist}", file=sys.stderr)
        return (2, dist)
    wheel = wheels[0]
    source = """
from typevet.evaluation.gemma_native_vision_wheel_smoke import run_wheel_smoke
raise SystemExit(run_wheel_smoke())
"""
    completed = run_isolated_wheel_python(
        wheel=wheel,
        source=source,
        cwd=isolated_cwd,
    )
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    return (int(completed.returncode), wheel)


def main() -> int:
    """CLI entry for isolated factory wheel smoke.

    Returns:
        Process exit code from the smoke harness.
    """
    code, _wheel = run_offline_wheel_smoke()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
