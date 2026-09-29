r"""Wheel-isolated instruction-variant consumer proof ([#177][i177]).

Examples:
    ```console
    $ uv run python scripts/consumer_instruction_variant_proof.py
    ```

See Also:
    - [scripts.build_wheel_for_tests][]: wheel build helpers
    - [typevet.evaluation.instruction_variant_consumer_proof][]: in-tree CLI
"""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

from typevet.adapters.inbound.settings import load_llama_settings
from typevet.evaluation.runner.live_gate import (
    TYPEVET_REQUIRE_LIVE_ENV,
    live_gate_action,
    live_skip_reason,
    require_live_enabled,
)
from typevet_evals.wheel_isolated import (
    build_wheel_to_directory,
    run_isolated_wheel_python,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]

TYPEVET_WHEEL_SHA256_ENV = "TYPEVET_WHEEL_SHA256"
_EXIT_INVALID = 2
_FIXTURE_CLI_ARG_COUNT = 2
_DEFAULT_FIXTURE = _REPO_ROOT / "tests" / "fixtures" / "psai" / "vision_smoke"


def sha256_hex(path: Path) -> str:
    """Return the SHA-256 hex digest of ``path`` bytes.

    Returns:
        Lowercase hex digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_wheel_sha256(measured: str, expected: str | None) -> None:
    """Raise when ``expected`` is set and does not match ``measured``.

    Raises:
        ValueError: On digest mismatch.
    """
    if expected is None or expected.strip() == "":
        return
    if measured.lower() != expected.strip().lower():
        msg = f"wheel sha256 mismatch: measured {measured} != env {expected}"
        raise ValueError(msg)


def _isolated_offline_source(
    *,
    fixture_root: Path,
    wheel_sha256: str,
    out_dir: Path,
) -> str:
    fixture = str(fixture_root.resolve())
    receipt_dir = str(out_dir.resolve())
    return f"""
import importlib.metadata
from pathlib import Path

import typevet
from typevet.evaluation.instruction_variant_consumer_proof import proof_main

install_path = Path(typevet.__file__).resolve()
version = importlib.metadata.version("typevet")
print("typevet_install_path", install_path)
print("typevet_version", version)
print("wheel_sha256_measured", {wheel_sha256!r})
argv = [
    "--fixture-root",
    {fixture!r},
    "--out-dir",
    {receipt_dir!r},
    "--wheel-sha256",
    {wheel_sha256!r},
    "--typevet-install-path",
    str(install_path),
]
raise SystemExit(proof_main(argv))
"""


def _isolated_live_source(
    *,
    fixture_root: Path,
    wheel_sha256: str,
    out_dir: Path,
) -> str:
    fixture = str(fixture_root.resolve())
    receipt_dir = str(out_dir.resolve())
    return f"""
import importlib.metadata
from pathlib import Path

import typevet
from typevet.evaluation.instruction_variant_consumer_live import (
    live_instruction_variant_proof_main,
)

install_path = Path(typevet.__file__).resolve()
print("typevet_install_path", install_path)
print("wheel_sha256_measured", {wheel_sha256!r})
argv = [
    "--fixture-root",
    {fixture!r},
    "--out-dir",
    {receipt_dir!r},
    "--wheel-sha256",
    {wheel_sha256!r},
    "--typevet-install-path",
    str(install_path),
]
raise SystemExit(live_instruction_variant_proof_main(argv))
"""


def run_offline_wheel_proof(
    *,
    fixture_root: Path,
    work_dir: Path | None = None,
    out_dir: Path | None = None,
) -> tuple[int, Path, str]:
    """Build wheel and run offline instruction-variant proof in isolation.

    Returns:
        ``(exit_code, wheel_path, wheel_sha256)``.
    """
    base = work_dir or Path(tempfile.mkdtemp(prefix="typevet-variant-wheel-"))
    dist = base / "dist"
    isolated_cwd = base / "isolated_cwd"
    isolated_cwd.mkdir(parents=True, exist_ok=True)
    build_wheel_to_directory(dist)
    wheels = sorted(dist.glob("typevet-*.whl"))
    if len(wheels) != 1:
        print(f"FAIL_CLOSED: expected one wheel under {dist}", file=sys.stderr)
        return (_EXIT_INVALID, dist, "")
    wheel = wheels[0]
    measured = sha256_hex(wheel)
    try:
        verify_wheel_sha256(measured, os.environ.get(TYPEVET_WHEEL_SHA256_ENV))
    except ValueError as exc:
        print(f"FAIL_CLOSED: {exc}", file=sys.stderr)
        return (_EXIT_INVALID, wheel, measured)
    receipt_dir = out_dir or (base / "receipts")
    receipt_dir.mkdir(parents=True, exist_ok=True)
    completed = run_isolated_wheel_python(
        wheel=wheel,
        source=_isolated_offline_source(
            fixture_root=fixture_root,
            wheel_sha256=measured,
            out_dir=receipt_dir,
        ),
        cwd=isolated_cwd,
    )
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    return (int(completed.returncode), wheel, measured)


def run_live_wheel_proof(
    *,
    fixture_root: Path,
    wheel: Path,
    wheel_sha256: str,
    work_dir: Path | None = None,
    out_dir: Path | None = None,
) -> int:
    """Run live instruction-variant proof inside an isolated wheel environment.

    Returns:
        Process exit code from the isolated live harness.
    """
    base = work_dir or Path(tempfile.mkdtemp(prefix="typevet-variant-live-"))
    isolated_cwd = base / "isolated_live_cwd"
    isolated_cwd.mkdir(parents=True, exist_ok=True)
    receipt_dir = out_dir or (base / "receipts")
    receipt_dir.mkdir(parents=True, exist_ok=True)
    completed = run_isolated_wheel_python(
        wheel=wheel,
        source=_isolated_live_source(
            fixture_root=fixture_root,
            wheel_sha256=wheel_sha256,
            out_dir=receipt_dir,
        ),
        cwd=isolated_cwd,
    )
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    return int(completed.returncode)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI for wheel-isolated instruction-variant proof with optional live gate.

    Returns:
        Exit code ``0`` on offline pass (and live pass or skip when not required).
    """
    args = list(argv if argv is not None else sys.argv[1:])
    fixture = _DEFAULT_FIXTURE
    exit_code = 0
    if args:
        if args[0] in {"-h", "--help"}:
            print(
                "usage: consumer_instruction_variant_proof.py [--fixture-root PATH]",
                file=sys.stderr,
            )
            return 0
        if args[0] == "--fixture-root" and len(args) == _FIXTURE_CLI_ARG_COUNT:
            fixture = Path(args[1])
        else:
            print(
                "usage: consumer_instruction_variant_proof.py [--fixture-root PATH]",
                file=sys.stderr,
            )
            exit_code = _EXIT_INVALID
    if exit_code == 0:
        try:
            offline_code, wheel, measured = run_offline_wheel_proof(
                fixture_root=fixture.resolve(),
            )
        except (RuntimeError, ValueError) as exc:
            print(f"FAIL_CLOSED: {exc}", file=sys.stderr)
            exit_code = _EXIT_INVALID
        else:
            exit_code = offline_code
            if exit_code == 0 and require_live_enabled():
                settings = load_llama_settings()
                reason = live_skip_reason(settings)
                action = live_gate_action(reason)
                if action.name == "FAIL":
                    print(
                        f"FAIL_CLOSED {TYPEVET_REQUIRE_LIVE_ENV}: {reason}",
                        file=sys.stderr,
                    )
                    exit_code = 1
                elif action.name == "RUN":
                    exit_code = run_live_wheel_proof(
                        fixture_root=fixture.resolve(),
                        wheel=wheel,
                        wheel_sha256=measured,
                    )
                else:
                    print(f"live skipped: {reason}", file=sys.stderr)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
