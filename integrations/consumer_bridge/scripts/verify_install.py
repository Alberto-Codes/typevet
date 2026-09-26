"""Verify private wheel identities and isolated companion installation."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import shutil
import tempfile
import zipfile
from email.parser import Parser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {
    "typevet": {"==0.1.0"},
    "judgevet": {"==0.13.0"},
    "httpx": {">=0.28", "<0.29"},
}
IMPORT_GUARD = """
import importlib
import importlib.metadata
from contextlib import ExitStack
from unittest.mock import patch
import httpx
import socket
import urllib.request


def guard_import(name):
    with ExitStack() as guards:
        for target in (
            'httpx.Client', 'httpx.AsyncClient',
            'httpx.request', 'httpx.get', 'httpx.post',
            'socket.create_connection', 'socket.socket.connect',
            'urllib.request.urlopen',
            'importlib.metadata.version', 'importlib.metadata.distribution',
            'importlib.metadata.distributions',
        ):
            guards.enter_context(patch(
                target, side_effect=RuntimeError('import-time I/O is forbidden')
            ))
        return importlib.import_module(name)
"""
PROBE = (
    IMPORT_GUARD
    + """
import importlib
import importlib.metadata as metadata
import importlib.util
import json
from pathlib import Path
import sys

root = Path(sys.prefix).resolve()
if root == Path(sys.base_prefix).resolve():
    raise AssertionError("missing isolated environment")
for name in sys.argv[1:]:
    if name == "typevet_consumer_bridge":
        module = guard_import(name)
    else:
        module = importlib.import_module(name)
    path = Path(module.__file__).resolve()
    distribution = metadata.distribution(name.replace("_", "-"))
    files = distribution.files or ()
    recorded = {Path(distribution.locate_file(item)).resolve() for item in files}
    if not path.is_relative_to(root) or path not in recorded:
        raise AssertionError("import does not match installed RECORD")
if "typevet_consumer_bridge" not in sys.argv:
    if importlib.util.find_spec("judgevet") is not None:
        raise AssertionError("consumer leaked into base installation")
    if importlib.util.find_spec("typevet_consumer_bridge") is not None:
        raise AssertionError("bridge leaked into base installation")
print(json.dumps({d.metadata["Name"]: d.version for d in metadata.distributions()},
                 sort_keys=True))
"""
)


def validate_dependencies(requirements: list[str]) -> None:
    """Reject missing, extra or changed bridge dependency declarations."""
    actual = {}
    for requirement in requirements:
        match = re.fullmatch(r"([a-z-]+)([<>=0-9.,]+)", requirement.replace(" ", ""))
        if match is None or match[1] in actual:
            raise ValueError("invalid dependency declaration")
        actual[match[1]] = set(match[2].split(","))
    if actual != EXPECTED:
        raise ValueError("declared dependency set differs from accepted artifacts")


def verify_artifact(path: Path, artifact: dict[str, str]) -> None:
    """Reject a dependency artifact before any installation starts."""
    if path.name != artifact["filename"]:
        raise ValueError("dependency artifact filename mismatch")
    if hashlib.sha256(path.read_bytes()).hexdigest() != artifact["sha256"]:
        raise ValueError("dependency artifact SHA256 mismatch")


def verify_metadata(wheel: Path) -> None:
    """Check bridge wheel metadata and its typed import surface."""
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        metadata_name = next(name for name in names if name.endswith("/METADATA"))
        metadata = Parser().parsestr(archive.read(metadata_name).decode())
        if metadata["Name"] != "typevet-consumer-bridge":
            raise ValueError("bridge distribution identity mismatch")
        validate_dependencies(metadata.get_all("Requires-Dist", []))
        if "typevet_consumer_bridge/py.typed" not in names:
            raise ValueError("missing typing marker")


async def run_uv(*arguments: str, cwd: Path) -> None:
    """Run the resolved uv executable without a shell or inherited Python paths."""
    executable = shutil.which("uv")
    if executable is None:
        raise RuntimeError("uv is required")
    environment = {
        key: value
        for key, value in os.environ.items()
        if key
        not in {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT"}
    }
    process = await asyncio.create_subprocess_exec(
        executable, *arguments, cwd=cwd, env=environment, stderr=asyncio.subprocess.PIPE
    )
    _, errors = await process.communicate()
    if process.returncode:
        raise RuntimeError("uv verification command failed: " + errors.decode())


async def verify(typevet_wheel: Path, consumer_wheel: Path) -> None:
    """Build and install only after checking both frozen dependency hashes."""
    manifest = json.loads((ROOT / "dependency-artifacts.json").read_text())
    verify_artifact(typevet_wheel, manifest["typevet"])
    verify_artifact(consumer_wheel, manifest["judgevet"])
    with tempfile.TemporaryDirectory(prefix="typevet-bridge-package-") as directory:
        scratch = Path(directory)
        await run_uv("build", str(ROOT), "--wheel", "--out-dir", directory, cwd=scratch)
        bridge = next(scratch.glob("typevet_consumer_bridge-*.whl"))
        verify_metadata(bridge)
        print(
            "bridge SHA256:",
            hashlib.sha256(bridge.read_bytes()).hexdigest(),
            flush=True,
        )
        for label, wheels, imports in (
            ("base", [typevet_wheel], ["typevet"]),
            (
                "bridge",
                [typevet_wheel, consumer_wheel, bridge],
                ["typevet", "judgevet", "typevet_consumer_bridge"],
            ),
        ):
            environment = scratch / label
            await run_uv("venv", str(environment), cwd=scratch)
            python = str(environment / "bin/python")
            await run_uv(
                "pip",
                "install",
                "--python",
                python,
                *(str(wheel) for wheel in wheels),
                cwd=scratch,
            )
            await run_uv(
                "run",
                "--no-project",
                "--python",
                python,
                "python",
                "-I",
                "-B",
                "-c",
                PROBE,
                *imports,
                cwd=scratch,
            )


def main() -> None:
    """Parse operator-supplied paths and run the installation proof."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--typevet-wheel", type=Path, required=True)
    parser.add_argument("--consumer-wheel", type=Path, required=True)
    arguments = parser.parse_args()
    asyncio.run(
        verify(arguments.typevet_wheel.resolve(), arguments.consumer_wheel.resolve())
    )


if __name__ == "__main__":
    main()
