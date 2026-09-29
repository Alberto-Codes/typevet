r"""Wheel-isolated PSAI consumer proof runner ([#177][i177]).

Examples:
    ```console
    $ uv run python scripts/psai_vision_consumer_wheel_proof.py \
        --fixture-root tests/fixtures/psai/vision_smoke
    ```

See Also:
    - [scripts.build_wheel_for_tests][]: wheel build helpers
    - [typevet_evals.psai_vision_consumer.harness][]: matrix harness
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
    build_member_wheel_to_directory,
    build_wheel_to_directory,
    run_isolated_wheel_python,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]

TYPEVET_WHEEL_SHA256_ENV = "TYPEVET_WHEEL_SHA256"
_EXIT_INVALID = 2
_FIXTURE_CLI_ARG_COUNT = 2


def sha256_hex(path: Path) -> str:
    """Return the SHA-256 hex digest of ``path`` bytes.

    Args:
        path: File to hash.

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

    Args:
        measured: Digest from the built wheel on disk.
        expected: Optional pin from the environment.

    Raises:
        ValueError: On digest mismatch.
    """
    if expected is None or expected.strip() == "":
        return
    if measured.lower() != expected.strip().lower():
        msg = f"wheel sha256 mismatch: measured {measured} != env {expected}"
        raise ValueError(msg)


def _member_wheels(base: Path) -> list[Path]:
    """Build the ``typevet-evals`` member wheel under ``base``.

    The harness lives in the member, which the ``typevet`` wheel does not
    hold (#256), so each isolated run installs this wheel next to it.

    Returns:
        Sorted ``typevet_evals-*.whl`` paths for ``extra_wheels``.
    """
    member_dist = base / "evals-dist"
    build_member_wheel_to_directory(member_dist)
    return sorted(member_dist.glob("typevet_evals-*.whl"))


def _isolated_offline_source(
    *,
    fixture_root: Path,
    wheel_sha256: str,
) -> str:
    fixture = str(fixture_root.resolve())
    return f"""
import importlib.metadata
from pathlib import Path

import typevet
from typevet_evals.psai_vision_consumer.harness import consumer_proof_main

install_path = Path(typevet.__file__).resolve()
version = importlib.metadata.version("typevet")
print("typevet_install_path", install_path)
print("typevet_version", version)
print("wheel_sha256_measured", {wheel_sha256!r})
argv = [
    "--fixture-root",
    {fixture!r},
    "--wheel-sha256",
    {wheel_sha256!r},
]
raise SystemExit(consumer_proof_main(argv))
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
from typevet_evals.psai_vision_consumer.live import live_consumer_proof_main

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
]
raise SystemExit(live_consumer_proof_main(argv))
"""


def _run_isolated_phase(
    *,
    wheel: Path,
    source: str,
    isolated_cwd: Path,
    extra_wheels: Sequence[Path],
) -> int:
    completed = run_isolated_wheel_python(
        wheel=wheel,
        source=source,
        cwd=isolated_cwd,
        extra_wheels=extra_wheels,
    )
    if completed.stdout:
        print(completed.stdout, end="")
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    return int(completed.returncode)


def run_offline_wheel_proof(
    *,
    fixture_root: Path,
    work_dir: Path | None = None,
    wheel: Path | None = None,
    measured_sha256: str | None = None,
) -> tuple[int, Path, str]:
    """Build a wheel, verify optional digest pin, run offline consumer proof.

    Args:
        fixture_root: Committed ``vision_smoke`` directory (absolute path allowed).
        work_dir: Optional parent for dist and isolated cwd (outside checkout).
        wheel: Pre-built wheel (skips build when set with ``measured_sha256``).
        measured_sha256: Digest for ``wheel`` when build is skipped.

    Returns:
        ``(exit_code, wheel_path, wheel_sha256)``.
    """
    base = work_dir or Path(tempfile.mkdtemp(prefix="typevet-consumer-wheel-"))
    dist = base / "dist"
    isolated_cwd = base / "isolated_cwd"
    isolated_cwd.mkdir(parents=True, exist_ok=True)
    if wheel is None:
        build_wheel_to_directory(dist)
        wheels = sorted(dist.glob("typevet-*.whl"))
        if len(wheels) != 1:
            print(f"FAIL_CLOSED: expected one wheel under {dist}", file=sys.stderr)
            return (_EXIT_INVALID, dist, "")
        wheel = wheels[0]
        measured = sha256_hex(wheel)
    else:
        measured = measured_sha256 or sha256_hex(wheel)
    try:
        verify_wheel_sha256(measured, os.environ.get(TYPEVET_WHEEL_SHA256_ENV))
    except ValueError as exc:
        print(f"FAIL_CLOSED: {exc}", file=sys.stderr)
        return (_EXIT_INVALID, wheel, measured)
    code = _run_isolated_phase(
        wheel=wheel,
        source=_isolated_offline_source(
            fixture_root=fixture_root,
            wheel_sha256=measured,
        ),
        isolated_cwd=isolated_cwd,
        extra_wheels=_member_wheels(base),
    )
    return (code, wheel, measured)


def run_live_wheel_proof(
    *,
    fixture_root: Path,
    wheel: Path,
    wheel_sha256: str,
    work_dir: Path | None = None,
    out_dir: Path | None = None,
) -> int:
    """Run live consumer proof inside an isolated wheel environment.

    Args:
        fixture_root: Committed ``vision_smoke`` directory.
        wheel: Built ``typevet`` wheel installed in isolation.
        wheel_sha256: Measured digest of ``wheel``.
        work_dir: Optional parent for isolated cwd.
        out_dir: Receipt directory (defaults to a temp dir under ``work_dir``).

    Returns:
        Process exit code from the isolated live harness.
    """
    base = work_dir or Path(tempfile.mkdtemp(prefix="typevet-consumer-live-"))
    isolated_cwd = base / "isolated_live_cwd"
    isolated_cwd.mkdir(parents=True, exist_ok=True)
    receipt_dir = out_dir or (base / "live_receipts")
    receipt_dir.mkdir(parents=True, exist_ok=True)
    return _run_isolated_phase(
        wheel=wheel,
        source=_isolated_live_source(
            fixture_root=fixture_root,
            wheel_sha256=wheel_sha256,
            out_dir=receipt_dir,
        ),
        isolated_cwd=isolated_cwd,
        extra_wheels=_member_wheels(base),
    )


def live_wheel_branch_status() -> tuple[int, str]:
    """Report live consumer proof status for ``TYPEVET_REQUIRE_LIVE`` runs.

    Returns:
        ``(exit_code, message)`` where ``0`` means live branch skipped cleanly.
    """
    settings = load_llama_settings()
    reason = live_skip_reason(settings)
    action = live_gate_action(reason)
    if action.name == "FAIL":
        return (1, f"FAIL_CLOSED {TYPEVET_REQUIRE_LIVE_ENV}: {reason}")
    if action.name == "SKIP":
        return (0, f"live skipped: {reason}")
    return (0, "live gate open")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI for wheel-isolated consumer proof with optional live gate.

    Returns:
        Exit code ``0`` on offline pass (and live pass or skip when not required).
    """
    args = list(argv if argv is not None else sys.argv[1:])
    fixture = _REPO_ROOT / "tests/fixtures/psai/vision_smoke"
    exit_code = 0
    if args:
        if args[0] in {"-h", "--help"}:
            print(
                "usage: psai_vision_consumer_wheel_proof.py [--fixture-root PATH]",
                file=sys.stderr,
            )
            return 0
        if args[0] == "--fixture-root" and len(args) == _FIXTURE_CLI_ARG_COUNT:
            fixture = Path(args[1])
        else:
            print("usage: psai_vision_consumer_wheel_proof.py [--fixture-root PATH]")
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
                gate_code, message = live_wheel_branch_status()
                print(message, file=sys.stderr)
                exit_code = gate_code
                settings = load_llama_settings()
                if (
                    exit_code == 0
                    and live_gate_action(live_skip_reason(settings)).name == "RUN"
                ):
                    exit_code = run_live_wheel_proof(
                        fixture_root=fixture.resolve(),
                        wheel=wheel,
                        wheel_sha256=measured,
                    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
