"""Build and run isolated wheel environments for packaging proofs.

Examples:
    ```python
    from pathlib import Path

    from typevet_evals.wheel_isolated import (
        build_wheel_to_directory,
        run_isolated_wheel_python,
    )

    build_wheel_to_directory(Path("/tmp/dist"))
    ```

See Also:
    - [scripts.build_wheel_for_tests][]: CLI shim for this module
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_UV_RUN_ISOLATED_TAIL: tuple[str, ...] = (
    "run",
    "--isolated",
    "--no-project",
    "--with",
)
_MIN_UV_HEAD_LEN = 2
_MIN_ISOLATED_WHEEL_ARGV_LEN = 9


def _uv_executable() -> str:
    """Return the ``uv`` binary path.

    Returns:
        Absolute path to the ``uv`` executable on ``PATH``.

    Raises:
        RuntimeError: When ``uv`` is not on ``PATH``.
    """
    uv_bin = shutil.which("uv")
    if uv_bin is None:
        msg = "uv not on PATH"
        raise RuntimeError(msg)
    return uv_bin


def _assert_uv_argv(argv: Sequence[str]) -> None:
    """Reject subprocess argv that does not start with the resolved ``uv`` binary.

    Raises:
        ValueError: When ``argv`` is not a validated isolated ``uv run --with`` invoke.
    """
    if len(argv) < _MIN_UV_HEAD_LEN:
        msg = "uv argv too short"
        raise ValueError(msg)
    expected_uv = _uv_executable()
    if argv[0] != expected_uv:
        msg = f"unexpected executable {argv[0]!r}; expected {expected_uv!r}"
        raise ValueError(msg)
    tail = tuple(argv[1 : 1 + len(_UV_RUN_ISOLATED_TAIL)])
    if tail != _UV_RUN_ISOLATED_TAIL:
        msg = f"unexpected uv tail {tail!r}"
        raise ValueError(msg)
    if len(argv) < _MIN_ISOLATED_WHEEL_ARGV_LEN:
        msg = "isolated wheel argv missing wheel path or python -c body"
        raise ValueError(msg)


def build_wheel_to_directory(out_dir: Path) -> None:
    """Run ``uv build`` with wheel output under ``out_dir``.

    Args:
        out_dir: Absolute directory for ``typevet-*.whl`` artifacts.

    Raises:
        RuntimeError: When ``uv`` is not on ``PATH``.
        subprocess.CalledProcessError: When ``uv build`` fails.
    """
    _uv_build(out_dir)


def build_member_wheel_to_directory(out_dir: Path) -> None:
    """Run ``uv build`` for the ``typevet-evals`` member wheel under ``out_dir``.

    An isolated environment needs this wheel next to the ``typevet`` wheel
    when its code imports ``typevet_evals`` (#256 E2).

    Args:
        out_dir: Absolute directory for the ``typevet_evals-*.whl`` artifact.

    Raises:
        RuntimeError: When ``uv`` is not on ``PATH``.
        subprocess.CalledProcessError: When ``uv build`` fails.
    """
    _uv_build(out_dir, "--package", "typevet-evals", "--wheel")


def _uv_build(out_dir: Path, *extra: str) -> None:
    """Run ``uv build`` from the repository root into ``out_dir``.

    Raises:
        RuntimeError: When ``uv`` is not on ``PATH``.
        subprocess.CalledProcessError: When ``uv build`` fails.
    """
    uv_bin = _uv_executable()
    resolved = out_dir.resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [uv_bin, "build", *extra, "--out-dir", str(resolved)],
        check=True,
        cwd=_REPO_ROOT,
    )


def run_isolated_wheel_python(
    *,
    wheel: Path,
    source: str,
    cwd: Path,
    extra_wheels: Sequence[Path] = (),
) -> subprocess.CompletedProcess[str]:
    """Run ``python -c`` in an isolated env with only the given wheels installed.

    Args:
        wheel: Built ``typevet`` wheel path passed to ``uv run --with``.
        source: Python statements for ``python -c``.
        cwd: Working directory outside the checkout when possible.
        extra_wheels: More wheel paths, each passed as one more ``--with``
            (for example the ``typevet-evals`` member wheel).

    Returns:
        Completed process with captured stdout and stderr.

    Raises:
        RuntimeError: When ``uv`` is not on ``PATH``.
        ValueError: When the constructed argv fails the allowlist check.
    """
    argv = [
        _uv_executable(),
        *_UV_RUN_ISOLATED_TAIL,
        str(wheel.resolve()),
        *(arg for extra in extra_wheels for arg in ("--with", str(extra.resolve()))),
        "python",
        "-c",
        source,
    ]
    _assert_uv_argv(argv)
    return subprocess.run(
        argv,
        check=False,
        capture_output=True,
        text=True,
        cwd=cwd,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry for wheel builds into one directory.

    Returns:
        Exit code ``0`` on success, ``2`` when usage is wrong.
    """
    args = list(argv if argv is not None else sys.argv[1:])
    if len(args) != 1:
        print("usage: build_wheel_for_tests.py OUT_DIR", file=sys.stderr)
        return 2
    build_wheel_to_directory(Path(args[0]))
    return 0
