"""Build the typevet wheel into a directory (packaging tests only)."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]


def build_wheel_to_directory(out_dir: Path) -> None:
    """Run ``uv build`` with wheel output under ``out_dir``.

    Args:
        out_dir: Absolute directory for ``typevet-*.whl`` artifacts.
    """
    uv_bin = shutil.which("uv")
    if uv_bin is None:
        msg = "uv not on PATH"
        raise RuntimeError(msg)
    resolved = out_dir.resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [uv_bin, "build", "--out-dir", str(resolved)],
        check=True,
        cwd=_REPO_ROOT,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry for pre-commit or manual wheel builds into one directory."""
    args = list(argv if argv is not None else sys.argv[1:])
    if len(args) != 1:
        print("usage: build_wheel_for_tests.py OUT_DIR", file=sys.stderr)
        return 2
    build_wheel_to_directory(Path(args[0]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
