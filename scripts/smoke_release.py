"""Smoke-test a built typevet wheel before it goes to an index.

The script reads the wheel metadata, then installs the wheel alone in an
isolated ``uv`` environment outside the checkout. The child process imports
``typevet`` and checks ``__version__``, the ``py.typed`` marker and that the
import did not come from ``src/``.

Examples:
    ```console
    $ uv build --out-dir /tmp/typevet-dist
    $ uv run python scripts/smoke_release.py \
        --wheel /tmp/typevet-dist/typevet-0.1.0-py3-none-any.whl
    ```

See Also:
    - [typevet_evals.wheel_isolated][]: isolated ``uv run --with`` helper
    - [tests.unit.test_release_metadata][]: the same metadata checks in the suite
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import zipfile
from email.parser import Parser
from email.policy import compat32
from pathlib import Path

from typevet_evals.wheel_isolated import run_isolated_wheel_python

__all__ = ["check_wheel_metadata", "main", "wheel_version"]

_DOCUMENTATION_URL = "Documentation, https://alberto-codes.github.io/typevet/"
_WHEEL_NAME_MIN_PARTS = 2

_CHILD_SOURCE = """
from importlib.metadata import version
from pathlib import Path

import typevet

package_dir = Path(typevet.__file__).resolve().parent
problems = []
if typevet.__version__ != expected:
    problems.append(f"__version__ {typevet.__version__!r} != {expected!r}")
if version("typevet") != expected:
    problems.append(f"metadata version {version('typevet')!r} != {expected!r}")
if not (package_dir / "py.typed").is_file():
    problems.append("py.typed marker missing")
if "site-packages" not in package_dir.parts:
    problems.append(f"typevet imported from {package_dir}, not site-packages")
for problem in problems:
    print(f"FAIL: {problem}")
if problems:
    raise SystemExit(1)
print(f"OK: typevet {typevet.__version__} from {package_dir}")
"""


def wheel_version(wheel: Path) -> str:
    """Return the version field of a wheel file name.

    Args:
        wheel: Path to a ``typevet-<version>-...whl`` file.

    Returns:
        The version segment of the file name.

    Raises:
        ValueError: When the file name is not a typevet wheel name.
    """
    parts = wheel.name.split("-")
    if (
        len(parts) < _WHEEL_NAME_MIN_PARTS
        or parts[0] != "typevet"
        or wheel.suffix != ".whl"
    ):
        msg = f"not a typevet wheel: {wheel.name}"
        raise ValueError(msg)
    return parts[1]


def check_wheel_metadata(wheel: Path) -> list[str]:
    """Check the licence and project URL fields in the wheel ``METADATA``.

    Args:
        wheel: Path to the built wheel.

    Returns:
        One message for each problem. An empty list means the metadata is good.
    """
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        metadata_name = next(n for n in names if n.endswith(".dist-info/METADATA"))
        text = archive.read(metadata_name).decode("utf-8")
    metadata = Parser(policy=compat32).parsestr(text)
    problems: list[str] = []
    if metadata["License-Expression"] != "MIT":
        problems.append(f"License-Expression is {metadata['License-Expression']!r}")
    if not any(n.endswith(".dist-info/licenses/LICENSE") for n in names):
        problems.append("LICENSE file missing from .dist-info/licenses/")
    if _DOCUMENTATION_URL not in (metadata.get_all("Project-URL") or []):
        problems.append("Documentation Project-URL missing")
    return problems


def main(argv: list[str] | None = None) -> int:
    """Run the metadata checks and the isolated import check.

    Args:
        argv: Command-line arguments. ``None`` reads ``sys.argv``.

    Returns:
        ``0`` when every check passes, ``1`` otherwise.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument(
        "--expect-version",
        help="Version the wheel must carry (default: the wheel file name).",
    )
    args = parser.parse_args(argv)
    wheel: Path = args.wheel.resolve()
    expected = args.expect_version or wheel_version(wheel)
    if wheel_version(wheel) != expected:
        print(f"FAIL: wheel {wheel.name} is not version {expected}")
        return 1
    problems = check_wheel_metadata(wheel)
    for problem in problems:
        print(f"FAIL: {problem}")
    with tempfile.TemporaryDirectory(prefix="typevet-smoke-") as scratch:
        result = run_isolated_wheel_python(
            wheel=wheel,
            source=f"expected = {expected!r}\n{_CHILD_SOURCE}",
            cwd=Path(scratch),
        )
    print(result.stdout, end="")
    print(result.stderr, end="", file=sys.stderr)
    if problems or result.returncode != 0:
        return 1
    print(f"OK: wheel metadata for {wheel.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
