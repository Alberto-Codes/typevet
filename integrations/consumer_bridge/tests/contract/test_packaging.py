"""Check the private distribution's packaging contract."""

import asyncio
import json
import sys
import tomllib
from pathlib import Path

import pytest

from integrations.consumer_bridge.scripts.verify_install import (
    IMPORT_GUARD,
    run_uv,
    validate_dependencies,
    verify_artifact,
)

SHA256_LENGTH = 64
SOURCE_SHA_LENGTH = 40
ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.contract


def require(condition: bool, message: str) -> None:
    """Fail with a named packaging condition."""
    if not condition:
        raise AssertionError(message)


def test_separate_distribution_metadata() -> None:
    """Require exact dependencies without source-checkout references."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    require(project["name"] == "typevet-consumer-bridge", "distribution identity")
    require(project["requires-python"] == ">=3.12", "Python support floor")
    require(
        set(project["dependencies"])
        == {"typevet==0.1.0", "judgevet==0.13.0", "httpx>=0.28,<0.29"},
        "declared dependency set",
    )


@pytest.mark.parametrize(
    "operation",
    [
        "import httpx; httpx.Client()",
        "import socket; socket.create_connection(('127.0.0.1', 1))",
        "import importlib.metadata; importlib.metadata.version('httpx')",
    ],
)
def test_import_guard_rejects_io(tmp_path: Path, operation: str) -> None:
    """Reject executable import side effects before they reach external resources."""
    (tmp_path / "unsafe_bridge.py").write_text(operation)
    probe = tmp_path / "probe.py"
    probe.write_text(
        IMPORT_GUARD
        + "\nimport sys\nsys.path.insert(0, "
        + repr(str(tmp_path))
        + ")\n"
        + "guard_import('unsafe_bridge')\n"
    )
    with pytest.raises(RuntimeError, match="import-time I/O is forbidden"):
        asyncio.run(
            run_uv(
                "run",
                "--no-project",
                "--python",
                sys.executable,
                "python",
                "-I",
                "-B",
                str(probe),
                cwd=tmp_path,
            )
        )


def test_artifact_manifest_has_frozen_hashes() -> None:
    """Require source and wheel identities instead of checkout paths."""
    manifest = json.loads((ROOT / "dependency-artifacts.json").read_text())
    require(set(manifest) == {"typevet", "judgevet"}, "artifact set")
    for artifact in manifest.values():
        require(
            set(artifact) == {"filename", "version", "source_sha", "sha256"},
            "artifact fields",
        )
        require(len(artifact["sha256"]) == SHA256_LENGTH, "SHA256 length")
        require(len(artifact["source_sha"]) == SOURCE_SHA_LENGTH, "source SHA length")
        require(Path(artifact["filename"]).name == artifact["filename"], "no paths")


def test_missing_dependency_declaration_fails() -> None:
    """Prove the wheel metadata oracle rejects a missing consumer dependency."""
    valid = ["typevet==0.1.0", "judgevet==0.13.0", "httpx<0.29,>=0.28"]
    validate_dependencies(valid)
    with pytest.raises(ValueError, match="declared dependency set"):
        validate_dependencies(
            [item for item in valid if not item.startswith("judgevet")]
        )


def test_changed_artifact_is_rejected(tmp_path: Path) -> None:
    """Reject changed wheel bytes before an installer can run."""
    artifact = json.loads((ROOT / "dependency-artifacts.json").read_text())["judgevet"]
    wheel = tmp_path / artifact["filename"]
    wheel.write_bytes(b"wrong artifact")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        verify_artifact(wheel, artifact)
