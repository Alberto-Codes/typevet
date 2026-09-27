"""Exercise installed proof commands, usage evidence, and saved responses without service calls.

Examples:
    Run ``uv run pytest integrations/consumer_bridge/tests/contract``.

See Also:
    - [live_text_proof][]: The bounded installed live command.
"""

import argparse
import asyncio
import base64
import csv
import hashlib
import io
import json
import os
import shutil
import sys
import zipfile
from http import HTTPStatus
from pathlib import Path

import httpx
import pytest

from integrations.consumer_bridge.scripts import live_text_proof
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
    """Reject a named installed proof assertion.

    Raises:
        AssertionError: When the named condition fails.
    """
    if not condition:
        raise AssertionError(message)


@pytest.fixture(scope="module")
def wheels(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path, Path]:
    """Build current bridge bytes and use operator-supplied frozen dependencies.

    Returns:
        Paths to the engine, consumer, and bridge wheels.
    """
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


def test_live_proof_command_exists(tmp_path: Path) -> None:
    """Require the bounded live command before its implementation."""
    asyncio.run(
        run_uv(
            "run",
            "--no-project",
            "--python",
            sys.executable,
            "python",
            str(ROOT / "scripts/live_text_proof.py"),
            "--help",
            cwd=tmp_path,
        )
    )


@pytest.fixture
def live():
    """Import the live command without contacting any service.

    Returns:
        The imported live command module.
    """
    return live_text_proof


@pytest.mark.parametrize("positive", [True, False])
def test_live_real_factory(live, router, positive: bool) -> None:
    """Check actual scoring entries, policy rules, complete wire and cleanup."""
    router.positive = positive
    receipt = {}
    case = json.loads((ROOT / "tests/fixtures/text_cases.json").read_text())
    arguments = argparse.Namespace(
        endpoint="http://offline.invalid", timeout=1.0, model="requested-model"
    )
    live.run_case(case, arguments, receipt, httpx.MockTransport(router.handle))
    require(receipt["counts"] == live.LIMITS, "all actual attempt counts")
    require(
        [rule["passed"] for rule in receipt["policy"]["rules"]] == [positive] * 3,
        "actual policy rules",
    )
    require(
        receipt["status"] == ("passed" if positive else "policy_failed"),
        "policy status",
    )
    require(all(receipt["cleanup"].values()), "real cleanup")
    require(
        len(router.requests)
        == sum(live.LIMITS[key] for key in ("metadata", "tokenizer", "completion")),
        "no extra cleanup HTTP",
    )
    require(all(call["state"] == "responded" for call in receipt["calls"]), "full wire")


@pytest.mark.parametrize(
    ("route", "counts"),
    [
        (
            "/props",
            {
                "judgment": 0,
                "scoring": 0,
                "metadata": 1,
                "tokenizer": 0,
                "completion": 0,
            },
        ),
        (
            "/tokenize",
            {
                "judgment": 1,
                "scoring": 0,
                "metadata": 2,
                "tokenizer": 1,
                "completion": 0,
            },
        ),
        (
            "/completion",
            {
                "judgment": 1,
                "scoring": 1,
                "metadata": 2,
                "tokenizer": 6,
                "completion": 1,
            },
        ),
    ],
)
def test_live_failed_attempts(live, router, route: str, counts: dict) -> None:
    """Count failed dispatches before setup, tokenization or scoring returns."""

    def handle(request):
        """Fail at the selected endpoint.

        Returns:
            The deterministic response for other endpoints.

        Raises:
            httpx.ConnectError: At the selected failure endpoint.
        """
        if request.url.path == route:
            raise httpx.ConnectError("synthetic unavailable service", request=request)
        return router.handle(request)

    receipt = {}
    case = json.loads((ROOT / "tests/fixtures/text_cases.json").read_text())
    arguments = argparse.Namespace(
        endpoint="http://offline.invalid", timeout=1.0, model="requested-model"
    )
    with pytest.raises(Exception) as failure:
        live.run_case(case, arguments, receipt, httpx.MockTransport(handle))
    live.record_failure(failure.value, receipt)
    require(receipt["counts"] == counts, "failed attempt charged")
    require(receipt["calls"][-1]["state"] == "admitted", "failed call retained")
    require(receipt["status"] == "service_unavailable", "unavailable classification")
    require(receipt["cleanup"]["client_closed"], "failure client cleanup")
    require(
        receipt["cleanup"]["adapter_close_called"] is (route != "/props"),
        "failure adapter cleanup",
    )


def test_live_budget_refuses_later_io(live) -> None:
    """Reject exhausted and unknown routes without dispatch or count increments."""
    receipt = {}
    calls = []
    budget = live.Budget(receipt, "http://offline.invalid")
    with httpx.Client(
        transport=httpx.MockTransport(lambda r: calls.append(r) or httpx.Response(200)),
        event_hooks={"request": [budget.request]},
    ) as client:
        for _ in range(live.LIMITS["completion"]):
            client.post("http://offline.invalid/completion", json={})
        before = receipt["counts"].copy()
        for url in (
            "http://offline.invalid/completion",
            "http://offline.invalid/health",
            "http://elsewhere.invalid/props",
        ):
            with pytest.raises(AssertionError):
                client.post(url, json={})
        require(receipt["counts"] == before, "refused attempts do not dispatch")
        require(len(calls) == live.LIMITS["completion"], "zero later IO")
    for category in ("judgment", "scoring"):
        for _ in range(live.LIMITS[category]):
            budget.reserve(category)
        with pytest.raises(AssertionError, match="budget exhausted"):
            budget.reserve(category)


@pytest.fixture(scope="module")
def offline_receipt(wheels, tmp_path_factory):
    """Keep one accepted installed environment for live CLI socket-guarded probes.

    Returns:
        The accepted offline receipt path.
    """
    path = tmp_path_factory.mktemp("live-installed") / "offline.json"
    command(wheels, path)
    return path


LIVE_PROBE = """
import sys
from unittest.mock import patch
sys.path.insert(0, sys.argv.pop(1))
scenario = sys.argv.pop(1)
import live_text_proof as live
import httpx
original = live.run_case

def handle(request):
    if scenario == "runtime" and request.url.path == "/completion":
        return httpx.Response(503, text="synthetic backend failure")
    if request.url.path == scenario:
        raise httpx.ConnectError("synthetic service failure", request=request)
    import installed_policy_proof
    return installed_policy_proof.handler(request, scenario != "negative", [])

def run(case, arguments, receipt):
    return original(case, arguments, receipt, httpx.MockTransport(handle))

live.run_case = run
with (
    patch("socket.socket.connect", side_effect=AssertionError("real socket forbidden")),
    patch("socket.create_connection", side_effect=AssertionError("real socket forbidden")),
):
    live.main()
"""


def live_command(
    offline: Path,
    receipt: Path,
    scenario: str,
    *,
    wrong_hash: bool = False,
    preflight: bool = False,
) -> None:
    """Execute staged production CLI in the accepted isolated interpreter."""
    data = json.loads(offline.read_text())
    script = receipt.parent / "live_text_proof.py"
    script.write_bytes((ROOT / "scripts/live_text_proof.py").read_bytes())
    config = Path(data["scratch"]) / "config.json"
    asyncio.run(
        run_uv(
            "run",
            "--no-project",
            "--python",
            data["installation"]["prefix"] + "/bin/python",
            "python",
            "-I",
            "-B",
            "-c",
            PREFLIGHT_PROBE if preflight else LIVE_PROBE,
            str(receipt.parent),
            scenario,
            "--offline-receipt",
            str(offline),
            "--offline-receipt-sha256",
            "0" * 64
            if wrong_hash
            else hashlib.sha256(offline.read_bytes()).hexdigest(),
            "--installed-config-sha256",
            "0" * 64
            if scenario == "bad-config"
            else hashlib.sha256(config.read_bytes()).hexdigest(),
            "--script-sha256",
            hashlib.sha256(script.read_bytes()).hexdigest(),
            "--endpoint",
            "http://offline.invalid",
            "--model",
            "private-model",
            "--receipt",
            str(receipt),
            cwd=receipt.parent,
        )
    )


@pytest.mark.parametrize(
    "scenario",
    ["positive", "negative", "runtime", "/props", "/tokenize", "/completion"],
)
def test_live_installed_receipt(offline_receipt, tmp_path: Path, scenario: str) -> None:
    """Retain installed identity, full results or failed attempts through the CLI."""
    receipt = tmp_path / "live.json"
    if scenario == "positive":
        live_command(offline_receipt, receipt, scenario)
    else:
        with pytest.raises(RuntimeError):
            live_command(offline_receipt, receipt, scenario)
    data = json.loads(receipt.read_text())
    require(
        bool(data["installation_before"]["modules"]), "before installed attestation"
    )
    require(bool(data["installation_after"]["modules"]), "after installed attestation")
    require(data["cleanup"]["client_closed"], "CLI cleanup")
    if scenario == "runtime":
        require(data["status"] == "runtime_failed", "runtime failure receipt")
        require(
            data["calls"][-1]["status"] == HTTPStatus.SERVICE_UNAVAILABLE,
            "failed status retained",
        )
    elif scenario.startswith("/"):
        require(data["status"] == "service_unavailable", "service failure receipt")
        require(data["calls"][-1]["path"] == scenario, "failed endpoint retained")
    else:
        require(
            data["status"] == ("passed" if scenario == "positive" else "policy_failed"),
            "policy classification",
        )
        require(
            len(data["response"]["answers"]) == len(data["policy"]["rules"]),
            "all typed results and rules",
        )
    before = receipt.read_bytes()
    with pytest.raises(RuntimeError, match="FileExistsError"):
        live_command(offline_receipt, receipt, scenario)
    require(receipt.read_bytes() == before, "exclusive live receipt")


def test_live_preflight_failure_receipt(offline_receipt, tmp_path: Path) -> None:
    """Reject changed acceptance bytes before any IO and retain the failure."""
    receipt = tmp_path / "bad.json"
    with pytest.raises(RuntimeError, match="offline receipt bytes"):
        live_command(offline_receipt, receipt, "positive", wrong_hash=True)
    data = json.loads(receipt.read_text())
    require(data["status"] == "runtime_failed", "proof failure classification")
    require(not data.get("calls"), "preflight zero IO")


def test_live_cleanup_calls_once(live, router, monkeypatch) -> None:
    """Observe one judgment entry and adapter close without a second probe."""
    bridge = __import__("typevet_consumer_bridge")
    original_close = bridge.TypevetSystemOneAdapter.close
    original_judge = bridge.TypevetSystemOneAdapter.system_one
    calls = []

    def close(adapter):
        """Record the actual adapter close invocation."""
        calls.append("close")
        original_close(adapter)

    def judge(adapter, *args):
        """Record the actual judgment invocation.

        Returns:
            The real consumer response.
        """
        calls.append("judge")
        return original_judge(adapter, *args)

    monkeypatch.setattr(bridge.TypevetSystemOneAdapter, "close", close)
    monkeypatch.setattr(bridge.TypevetSystemOneAdapter, "system_one", judge)
    receipt = {}
    case = json.loads((ROOT / "tests/fixtures/text_cases.json").read_text())
    arguments = argparse.Namespace(
        endpoint="http://offline.invalid", timeout=1.0, model="requested-model"
    )
    live.run_case(case, arguments, receipt, httpx.MockTransport(router.handle))
    require(calls == ["judge", "close"], "one judgment and close")


@pytest.mark.parametrize("defect", ["instructions", "multimodal_data"])
def test_live_wire_defect_refuses_transport(live, defect: str) -> None:
    """Charge and retain a bad completion request before refusing transport IO."""
    case = json.loads((ROOT / "tests/fixtures/text_cases.json").read_text())
    receipt = {}
    calls = []
    budget = live.Budget(receipt, "http://offline.invalid", "requested-model", case)
    body = {"model": "requested-model", "prompt": "generic prompt"}
    if defect == "multimodal_data":
        body["multimodal_data"] = "forbidden"
    with (
        httpx.Client(
            transport=httpx.MockTransport(
                lambda r: calls.append(r) or httpx.Response(200)
            ),
            event_hooks={"request": [budget.request]},
        ) as client,
        pytest.raises(AssertionError),
    ):
        client.post("http://offline.invalid/completion", json=body)
    require(not calls, "wire defect causes zero transport IO")
    require(receipt["counts"]["completion"] == 1, "failed request hook charged")
    require(receipt["calls"][0]["body"] == body, "failed body retained")


PREFLIGHT_PROBE = """
import json, sys
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, sys.argv.pop(1))
scenario = sys.argv.pop(1)
import live_text_proof as live
calls = []
def forbidden(*args, **kwargs):
    calls.append("socket")
    raise AssertionError("preimport socket attempted")
try:
    with (
        patch("socket.create_connection", side_effect=forbidden),
        patch("socket.socket.connect", side_effect=forbidden),
        patch("socket.socket.connect_ex", side_effect=forbidden),
    ):
        live.main()
finally:
    Path("observed.json").write_text(json.dumps({
        "calls": calls,
        "loaded": [name for name in (
            "installed_policy_proof", "verify_install", "typevet",
            "judgevet", "typevet_consumer_bridge",
        ) if name in sys.modules],
    }))
"""


@pytest.mark.parametrize("defect", ["bad-config", "installed-import"])
def test_live_rejects_before_import(
    offline_receipt, tmp_path: Path, defect: str
) -> None:
    """Reject corrupt inputs before helper or public imports can execute IO."""
    original = json.loads(offline_receipt.read_text())
    source = Path(original["scratch"])
    copied = tmp_path / "copied"
    shutil.copytree(source, copied, symlinks=True)
    offline = tmp_path / "offline.json"
    offline.write_text(offline_receipt.read_text().replace(str(source), str(copied)))
    config = copied / "config.json"
    config.write_text(config.read_text().replace(str(source), str(copied)))
    target = next(
        (copied / "environment/lib").glob(
            "python*/site-packages/typevet_consumer_bridge/__init__.py"
        )
    )
    target.write_bytes(
        target.read_bytes()
        + b'\nimport socket\nsocket.create_connection(("127.0.0.1", 9))\n'
    )
    receipt = tmp_path / "refused.json"
    with pytest.raises(RuntimeError):
        live_command(offline, receipt, defect, preflight=True)
    observed = json.loads((tmp_path / "observed.json").read_text())
    require(observed["calls"] == [], "zero pre-import socket attempts")
    expected_loaded = (
        [] if defect == "bad-config" else ["installed_policy_proof", "verify_install"]
    )
    require(observed["loaded"] == expected_loaded, "preflight import boundary")
    result = json.loads(receipt.read_text())
    require(result["status"] == "runtime_failed", "preflight failure retained")
    require(
        result["counts"] == dict.fromkeys(live_text_proof.LIMITS, 0),
        "zero attempt counters",
    )
    require(result["calls"] == [], "zero HTTP attempts")


@pytest.mark.parametrize(
    ("usage", "expected"),
    [
        ([(82, 1), (80, 1), (80, 1)], {"input_tokens": 242, "output_tokens": 3}),
        (
            [(82, None), (None, None), (80, None)],
            {"input_tokens": 162, "output_tokens": None},
        ),
        (
            [(None, 1), (None, None), (None, 1)],
            {"input_tokens": None, "output_tokens": 2},
        ),
        (
            [(True, -1), (1.5, "1"), (None, None)],
            {"input_tokens": None, "output_tokens": None},
        ),
        ([(0, 0), (None, None), (None, None)], {"input_tokens": 0, "output_tokens": 0}),
    ],
)
def test_live_usage_evidence(live, router, usage: list, expected: dict) -> None:
    """Preserve available per-field usage with the real runtime aggregation."""
    completion = iter(usage)

    def handle(request):
        """Attach recorded or partially available usage to deterministic answers.

        Returns:
            A real protocol response with the selected usage fields.
        """
        response = router.handle(request)
        if request.url.path == "/completion":
            input_tokens, output_tokens = next(completion)
            payload = response.json()
            payload.update(
                tokens_evaluated=input_tokens, tokens_predicted=output_tokens
            )
            return httpx.Response(200, json=payload)
        return response

    receipt = {}
    case = json.loads((ROOT / "tests/fixtures/text_cases.json").read_text())
    arguments = argparse.Namespace(
        endpoint="http://offline.invalid", timeout=1.0, model="requested-model"
    )
    live.run_case(case, arguments, receipt, httpx.MockTransport(handle))
    require(receipt["response"]["usage"] == expected, "actual engine usage")
    require(
        receipt["usage"] == expected, "receipt usage preserves known and unknown fields"
    )
    require(receipt["status"] == "passed", "usage does not reject policy success")
    receipt["response"]["usage"]["input_tokens"] = 999
    with pytest.raises(AssertionError, match="usage matches completion evidence"):
        live.validate_usage(receipt)


def test_saved_live_evidence_replay(live) -> None:
    """Replay frozen HTTP evidence without claiming a new live script execution."""
    path = os.environ.get("TYPEVET_LIVE_RECEIPT")
    if path is None:
        pytest.skip("Set TYPEVET_LIVE_RECEIPT for the retained one-shot evidence")
    raw = Path(path).read_bytes()
    require(
        hashlib.sha256(raw).hexdigest()
        == "ccb61825233c4bcbd27b2e9e5c08ada2107c71bca79717c8befb019e9691bda5",
        "original live evidence bytes",
    )
    saved = json.loads(raw)
    calls = iter(saved["calls"])

    def handle(request):
        """Serve recorded responses only for byte-equivalent request bodies.

        Returns:
            A saved response through an offline MockTransport.
        """
        call = next(calls)
        require(str(request.url) == call["url"], "recorded request URL")
        require(request.method == call["method"], "recorded request method")
        body = json.loads(request.content) if request.content else {}
        require(body == call["body"], "recorded request body")
        return httpx.Response(call["status"], text=call["response"])

    receipt = {}
    arguments = argparse.Namespace(
        endpoint=saved["endpoint"], timeout=1.0, model=saved["requested_model"]
    )
    live.run_case(saved["fixture"], arguments, receipt, httpx.MockTransport(handle))
    require(next(calls, None) is None, "all recorded calls consumed once")
    require(
        receipt["counts"] == saved["counts"] == live.LIMITS, "recorded attempt counts"
    )
    require(
        json.loads(json.dumps(receipt["response"])) == saved["response"],
        "exact saved typed result",
    )
    require(
        json.loads(json.dumps(receipt["policy"])) == saved["policy"],
        "exact saved policy outcome",
    )
    require(
        receipt["usage"] == {"input_tokens": 242, "output_tokens": 3},
        "recorded known usage",
    )
    require(receipt["status"] == "passed", "repaired oracle accepts offline replay")
    require(saved["status"] == "runtime_failed", "original process failure preserved")
    require(Path(path).read_bytes() == raw, "original receipt unchanged")
