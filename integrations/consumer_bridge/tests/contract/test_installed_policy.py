"""Exercise the installed policy proof as an external command."""

import asyncio
import base64
import csv
import hashlib
import io
import json
import os
import sys
import zipfile
from pathlib import Path

import pytest

from integrations.consumer_bridge.scripts.verify_install import run_uv

ROOT = Path(__file__).resolve().parents[2]
PROOF = ROOT / "scripts/installed_policy_proof.py"
pytestmark = pytest.mark.contract


def test_installed_proof_command_exists(tmp_path: Path) -> None:
    """Require an executable installed proof entry point before implementation."""
    asyncio.run(
        run_uv(
            "run",
            "--no-project",
            "--python",
            sys.executable,
            "python",
            str(PROOF),
            "--help",
            cwd=tmp_path,
        )
    )


def require(condition: bool, message: str) -> None:
    """Reject a named installed proof assertion."""
    if not condition:
        raise AssertionError(message)


@pytest.fixture(scope="module")
def wheels(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, Path]:
    """Build current bridge bytes and use operator-supplied frozen dependencies."""
    paths = [os.environ.get(name) for name in ("TYPEVET_WHEEL", "CONSUMER_WHEEL")]
    if not all(paths):
        pytest.skip("Set TYPEVET_WHEEL and CONSUMER_WHEEL to frozen artifact paths")
    directory = tmp_path_factory.mktemp("installed-wheels")
    asyncio.run(
        run_uv(
            "build", str(ROOT), "--wheel", "--out-dir", str(directory), cwd=directory
        )
    )
    return Path(str(paths[0])), Path(str(paths[1])), next(directory.glob("*.whl"))


def command(
    wheels: tuple[Path, Path, Path], receipt: Path, *, sha256: str | None = None
) -> None:
    """Execute the public proof CLI from outside the checkout."""
    asyncio.run(
        run_uv(
            "run",
            "--no-project",
            "--python",
            sys.executable,
            "python",
            str(PROOF),
            "--typevet-wheel",
            str(wheels[0]),
            "--consumer-wheel",
            str(wheels[1]),
            "--bridge-wheel",
            str(wheels[2]),
            "--bridge-sha256",
            sha256 or hashlib.sha256(wheels[2].read_bytes()).hexdigest(),
            "--receipt",
            str(receipt),
            cwd=receipt.parent,
        )
    )


def test_installed_real_policy(wheels: tuple[Path, Path, Path], tmp_path: Path) -> None:
    """Accept both known outcomes with installed byte and lifecycle evidence."""
    receipt = tmp_path / "proof.json"
    command(wheels, receipt)
    result = json.loads(receipt.read_text())
    require(result["status"] == "passed", "successful proof receipt")
    require(
        [case["passed"] for case in result["cases"]] == [True, False],
        "both policy outcomes",
    )
    for case in result["cases"]:
        require(
            [rule["passed"] for rule in case["policy"]["rules"]]
            == [case["positive"]] * 3,
            "all rule outcomes",
        )
        require(all(case["cleanup"].values()), "lifecycle checks")
    require(
        set(result["artifacts"]) == {"typevet", "judgevet", "typevet-consumer-bridge"},
        "exact artifacts",
    )
    require(
        result["installation"]["versions"]["judgevet"] == "0.13.0", "consumer version"
    )
    require(
        result["inputs"]["text_cases.json"]["sha256"]
        == "a6d40930345351e64fb65b669982655cfa012b19d4a0efb83ba0b00380a1d94a",
        "frozen fixture",
    )
    before = receipt.read_bytes()
    with pytest.raises(RuntimeError, match="FileExistsError"):
        command(wheels, receipt)
    require(receipt.read_bytes() == before, "exclusive receipt remains unchanged")


def mutate_wheel(source: Path, target: Path, member: str, old: str, new: str) -> None:
    """Repack a semantic mutation with valid wheel RECORD hashes and sizes."""
    with zipfile.ZipFile(source) as archive:
        content = {name: archive.read(name) for name in archive.namelist()}
    text = content[member].decode()
    require(old in text, "mutation precondition")
    content[member] = text.replace(old, new).encode()
    record = next(name for name in content if name.endswith("/RECORD"))
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    for name, data in content.items():
        if name != record and not name.endswith("/"):
            encoded = (
                base64.urlsafe_b64encode(hashlib.sha256(data).digest())
                .rstrip(b"=")
                .decode()
            )
            writer.writerow([name, "sha256=" + encoded, len(data)])
    writer.writerow([record, "", ""])
    content[record] = buffer.getvalue().encode()
    with zipfile.ZipFile(target, "w") as archive:
        for name, data in content.items():
            archive.writestr(name, data)


@pytest.mark.parametrize(
    ("member", "old", "new", "failure"),
    [
        (
            "questions.py",
            "elif isinstance(question, Mapping):",
            "elif isinstance(question, Mapping) and False:",
            "Unsupported question form",
        ),
        (
            "questions.py",
            "instructions=instructions",
            'instructions="generic"',
            "exact caller instructions",
        ),
        (
            "adapter.py",
            "finally:\n            adapter.close()",
            "finally:\n            pass",
            "owned wrapper must invalidate adapter",
        ),
    ],
)
def test_semantic_mutations_fail(
    wheels: tuple[Path, Path, Path],
    tmp_path: Path,
    member: str,
    old: str,
    new: str,
    failure: str,
) -> None:
    """Reject each behavior defect through a newly hashed installed wheel."""
    changed = tmp_path / wheels[2].name
    mutate_wheel(wheels[2], changed, "typevet_consumer_bridge/" + member, old, new)
    receipt = tmp_path / "mutation.json"
    with pytest.raises(RuntimeError, match=failure):
        command((wheels[0], wheels[1], changed), receipt)
    result = json.loads(receipt.read_text())
    require(
        result["status"] == "failed" and failure in result["error"],
        "retained semantic failure",
    )
    require("RECORD bytes" not in result["error"], "semantic probe reached")


def test_wrong_hash_fails_with_receipt(
    wheels: tuple[Path, Path, Path], tmp_path: Path
) -> None:
    """Retain hash failure evidence before any package installation."""
    receipt = tmp_path / "hash.json"
    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        command(wheels, receipt, sha256="0" * 64)
    require(
        json.loads(receipt.read_text())["status"] == "failed", "hash failure receipt"
    )


def test_modified_installed_bytes_fail(
    wheels: tuple[Path, Path, Path],
    tmp_path: Path,
) -> None:
    """Reject installed source changes even when its RECORD membership remains."""
    receipt = tmp_path / "intact.json"
    command(wheels, receipt)
    result = json.loads(receipt.read_text())
    scratch = Path(result["scratch"])
    module = Path(result["installation"]["modules"]["typevet_consumer_bridge"])
    replacement = module.with_suffix(".changed")
    replacement.write_bytes(module.read_bytes() + b"\n# Changed installed bytes.\n")
    replacement.replace(module)
    bootstrap = (
        "import runpy,sys; sys.path.insert(0,sys.argv[1]); "
        "sys.argv=sys.argv[2:]; runpy.run_path(sys.argv[0],run_name='__main__')"
    )
    with pytest.raises(RuntimeError, match="installed wheel bytes"):
        asyncio.run(
            run_uv(
                "run",
                "--no-project",
                "--python",
                str(scratch / "environment/bin/python"),
                "python",
                "-I",
                "-B",
                "-c",
                bootstrap,
                str(scratch),
                str(scratch / "installed_policy_proof.py"),
                "--installed-config",
                str(scratch / "config.json"),
                cwd=scratch,
            )
        )
