"""Prove private installed bridge policy behavior without service calls."""

from __future__ import annotations

import argparse
import asyncio
import base64
import csv
import hashlib
import importlib
import io
import json
import math
import shutil
import sys
import tempfile
import zipfile
from collections import Counter
from dataclasses import asdict
from importlib import metadata
from pathlib import Path
from unittest.mock import patch

import verify_install

ROOT = Path(__file__).resolve().parents[1]
MODEL = "private-model"
EXPECTED_CALLS = {"/props": 1, "/apply-template": 1, "/tokenize": 6, "/completion": 3}
PACKAGES = ("typevet", "judgevet", "typevet_consumer_bridge")


def require(condition: bool, message: str) -> None:
    """Reject a failed proof assertion."""
    if not condition:
        raise AssertionError(message)


def digest(path: Path) -> str:
    """Return the SHA256 of exact file bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_record(data: bytes, encoded: str, size: str) -> None:
    """Verify one wheel RECORD digest and byte count."""
    algorithm, value = encoded.split("=", 1)
    require(algorithm == "sha256", "RECORD algorithm")
    actual = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
    require(actual.decode() == value and len(data) == int(size), "RECORD bytes")


def attest(config: dict) -> dict:
    """Check wheel bytes, installed RECORD bytes and every loaded package module."""
    prefix = Path(sys.prefix).resolve()
    checkout = Path(config["checkout"])
    require(prefix != Path(sys.base_prefix).resolve(), "isolated environment")
    require(
        not Path.cwd().resolve().is_relative_to(checkout), "external working directory"
    )
    require(
        not any(Path(p).resolve().is_relative_to(checkout) for p in sys.path),
        "no checkout import path",
    )
    verified = set()
    for name, artifact in config["artifacts"].items():
        wheel = Path(artifact["path"])
        require(digest(wheel) == artifact["sha256"], "frozen wheel bytes")
        distribution = metadata.distribution(name)
        installed = {str(item): item for item in distribution.files or ()}
        with zipfile.ZipFile(wheel) as archive:
            record = next(n for n in archive.namelist() if n.endswith("/RECORD"))
            rows = csv.reader(io.StringIO(archive.read(record).decode()))
            for member, encoded, size in rows:
                if member == record:
                    continue
                data = archive.read(member)
                check_record(data, encoded, size)
                require(member in installed, "installed RECORD membership")
                item = installed[member]
                path = Path(str(distribution.locate_file(item))).resolve()
                require(path.is_relative_to(prefix), "installed wheel path")
                require(path.read_bytes() == data, "installed wheel bytes")
                if item.hash is None:
                    raise AssertionError("installed RECORD hash")
                check_record(
                    path.read_bytes(),
                    f"{item.hash.mode}={item.hash.value}",
                    str(item.size),
                )
                verified.add(path)
            require(
                {name for name in archive.namelist() if not name.endswith("/")}
                == {
                    row[0]
                    for row in csv.reader(io.StringIO(archive.read(record).decode()))
                },
                "complete wheel RECORD",
            )
    modules = {}
    for name, module in tuple(sys.modules.items()):
        if name.split(".")[0] in PACKAGES:
            if module.__file__ is None:
                raise AssertionError("loaded module has no source")
            path = Path(module.__file__).resolve()
            require(path in verified, "loaded module matches verified wheel bytes")
            modules[name] = str(path)
    require(all(name in modules for name in PACKAGES), "all installed public imports")
    return {
        "prefix": str(prefix),
        "cwd": str(Path.cwd()),
        "modules": modules,
        "versions": {d.metadata["Name"]: d.version for d in metadata.distributions()},
    }


def handler(request, positive: bool, calls: list):
    """Return deterministic protocol payloads behind the real runtime factory."""
    httpx = importlib.import_module("httpx")
    path = request.url.path
    body = json.loads(request.content) if request.content else {}
    calls.append({"path": path, "body": body})
    if path == "/props":
        payload = {"modalities": {"vision": True}}
    elif path == "/apply-template":
        payload = {"prompt": "<|turn>user\nhello<turn|>\n<|turn>model\n"}
    elif path == "/tokenize":
        payload = {"tokens": [101 if body["content"].endswith("1") else 100]}
    elif path == "/completion":
        preferred = 101 if positive else 100
        payload = {
            "completion_probabilities": [
                {
                    "top_logprobs": [
                        {"id": token, "logprob": 0.0 if token == preferred else -5.0}
                        for token in (100, 101)
                    ]
                }
            ]
        }
    else:
        raise AssertionError("unexpected HTTP endpoint")
    return httpx.Response(200, json=payload)


def verify_answers(response, questions: dict, positive: bool) -> None:
    """Assert consumer ownership, exact IDs, distributions and scalar values."""
    consumer = importlib.import_module("judgevet")
    require(type(response) is consumer.SystemOneResponse, "consumer response")
    require(type(response.usage) is consumer.Usage, "consumer usage")
    require(response.model == MODEL, "exact requested model")
    require(
        asdict(response.usage) == {"input_tokens": None, "output_tokens": None},
        "unknown usage",
    )
    require(list(response.answers) == list(questions), "exact answer IDs")
    true = (
        1 / (1 + math.exp(-5.0)) if positive else math.exp(-5.0) / (1 + math.exp(-5.0))
    )
    false = (
        math.exp(-5.0) / (1 + math.exp(-5.0)) if positive else 1 / (1 + math.exp(-5.0))
    )
    noul, choice, score = response.answers.values()
    require(type(noul) is consumer.NoulAnswer, "consumer NoulAnswer")
    require(type(choice) is consumer.ChoiceAnswer, "consumer ChoiceAnswer")
    require(type(score) is consumer.ScoreAnswer, "consumer ScoreAnswer")
    require(noul.noul == true, "Noul value")
    require(
        choice.probabilities == {"seven": false, "forty_two": true},
        "Choice distribution",
    )
    require(
        list(choice.probabilities) == ["seven", "forty_two"], "ordered Choice labels"
    )
    require(choice.choice == ("forty_two" if positive else "seven"), "Choice value")
    require(choice.confidence == max(false, true), "Choice confidence")
    require(score.probabilities == {0: false, 1: true}, "Score distribution")
    require(
        score.legend == {0: "seven dollars", 1: "forty-two dollars"}, "Score legend"
    )
    require(
        score.score == true and score.confidence == max(false, true),
        "weighted Score and confidence",
    )


def verify_wire(case: dict, calls: list) -> None:
    """Check real dispatches retain exact caller text and contain no media."""
    require(
        Counter(call["path"] for call in calls) == EXPECTED_CALLS, "HTTP attempt counts"
    )
    completions = [c["body"] for c in calls if c["path"] == "/completion"]
    for question, body in zip(case["questions"].values(), completions, strict=True):
        require(case["state"] in body["prompt"], "exact text state")
        require(question["instructions"] in body["prompt"], "exact caller instructions")
        criteria = question["criteria"]
        descriptions = criteria.values() if isinstance(criteria, dict) else criteria
        require(
            all(value in body["prompt"] for value in descriptions),
            "exact caller criteria",
        )
    require(all("image_data" not in c["body"] for c in calls), "no media")
    require(
        all(c["body"].get("model", MODEL) == MODEL for c in calls),
        "wire model identity",
    )


def prove_case(case: dict, positive: bool) -> dict:
    """Exercise raw mappings, consumer policy and meaningful wrapper cleanup."""
    httpx = importlib.import_module("httpx")
    consumer = importlib.import_module("judgevet")
    policy = importlib.import_module("judgevet.policy")
    bridge = importlib.import_module("typevet_consumer_bridge")
    forms = {"noul": consumer.Noul, "choice": consumer.Choice, "score": consumer.Score}
    questions = {
        key: forms[value["type"]](
            instructions=value["instructions"], criteria=value["criteria"]
        )
        for key, value in case["questions"].items()
    }
    validated = policy.validate_policy(
        policy.Policy(
            rules=(
                policy.NoulRule("total_is_42", **case["policy"]["total_is_42"]),
                policy.ChoiceRule("amount", **case["policy"]["amount"]),
                policy.ScoreRule("amount_score", **case["policy"]["amount_score"]),
            )
        ),
        questions,
    )
    calls = []
    settings = bridge.BridgeSettings("http://offline.invalid", 5.0, MODEL)
    with httpx.Client(
        base_url=settings.base_url,
        transport=httpx.MockTransport(lambda r: handler(r, positive, calls)),
    ) as client:
        with bridge.open_typevet_system_one(
            settings=settings, http_client=client
        ) as adapter:
            response = adapter.system_one(case["state"], case["questions"], MODEL)
        require(not client.is_closed, "caller client remains open")
        before = len(calls)
        try:
            adapter.system_one(case["state"], case["questions"], MODEL)
        except bridge.BridgeUnavailableError:
            pass
        else:
            raise AssertionError("owned wrapper must invalidate adapter")
        require(len(calls) == before, "closed adapter causes zero IO")
        verify_wire(case, calls)
        client.get("http://offline.invalid/props")
        require(len(calls) == before + 1, "caller client survives context exit")
    verify_answers(response, questions, positive)
    report = policy.evaluate_policy(validated, response.answers)
    require(
        report.passed is case["expected"]["positive" if positive else "negative"],
        "policy outcome",
    )
    require(all(rule.passed is positive for rule in report.rules), "per-rule outcomes")
    require(
        [rule.question for rule in report.rules] == list(questions), "policy rule IDs"
    )
    return {
        "positive": positive,
        "response": asdict(response),
        "policy": asdict(report),
        "passed": report.passed,
        "calls": calls,
        "cleanup": {"adapter_invalidated": True, "caller_client_survived": True},
    }


def child(config_path: Path) -> None:
    """Run offline proof under socket guards and retain installed evidence."""
    config = json.loads(config_path.read_text())
    case = json.loads(Path(config["fixture"]).read_text())
    with (
        patch(
            "socket.socket.connect", side_effect=AssertionError("real socket forbidden")
        ),
        patch(
            "socket.socket.connect_ex",
            side_effect=AssertionError("real socket forbidden"),
        ),
        patch(
            "socket.create_connection",
            side_effect=AssertionError("real socket forbidden"),
        ),
    ):
        cases = [prove_case(case, positive) for positive in (True, False)]
        identity = attest(config)
    Path(config["result"]).write_text(
        json.dumps({"cases": cases, "installation": identity}, indent=2)
    )


async def install_and_prove(arguments: argparse.Namespace, receipt: dict) -> None:
    """Install copied hash-verified artifacts and run the frozen child externally."""
    checkout = ROOT.parents[1]
    scratch = Path(tempfile.mkdtemp(prefix="typevet-bridge-installed-")).resolve()
    require(not scratch.is_relative_to(checkout), "scratch must be outside checkout")
    receipt["scratch"] = str(scratch)
    manifest_path = ROOT / "dependency-artifacts.json"
    manifest = json.loads(manifest_path.read_text())
    artifacts = {}
    receipt["artifacts"] = artifacts
    for name, source, expected in (
        ("typevet", arguments.typevet_wheel, manifest["typevet"]),
        ("judgevet", arguments.consumer_wheel, manifest["judgevet"]),
        (
            "typevet-consumer-bridge",
            arguments.bridge_wheel,
            {
                "filename": arguments.bridge_wheel.name,
                "sha256": arguments.bridge_sha256,
            },
        ),
    ):
        target = scratch / source.name
        shutil.copyfile(source, target)
        artifacts[name] = {
            "path": str(target),
            "sha256": digest(target),
            "expected_sha256": expected["sha256"],
        }
        verify_install.verify_artifact(target, expected)
    verify_install.verify_metadata(Path(artifacts["typevet-consumer-bridge"]["path"]))
    inputs = {}
    for source in (
        Path(__file__),
        Path(verify_install.__file__),
        ROOT / "tests/fixtures/text_cases.json",
        manifest_path,
    ):
        target = scratch / source.name
        shutil.copyfile(source, target)
        inputs[source.name] = {"path": str(target), "sha256": digest(target)}
    receipt.update({"artifacts": artifacts, "inputs": inputs})
    environment = scratch / "environment"
    await verify_install.run_uv("venv", str(environment), cwd=scratch)
    python = str(environment / "bin/python")
    await verify_install.run_uv(
        "pip",
        "install",
        "--python",
        python,
        *(v["path"] for v in artifacts.values()),
        cwd=scratch,
    )
    config = {
        "artifacts": artifacts,
        "checkout": str(checkout),
        "fixture": inputs["text_cases.json"]["path"],
        "result": str(scratch / "result.json"),
    }
    config_path = scratch / "config.json"
    config_path.write_text(json.dumps(config))
    command = (
        "import runpy,sys; sys.path.insert(0,sys.argv[1]); "
        "sys.argv=sys.argv[2:]; runpy.run_path(sys.argv[0],run_name='__main__')"
    )
    await verify_install.run_uv(
        "run",
        "--no-project",
        "--python",
        python,
        "python",
        "-I",
        "-B",
        "-c",
        command,
        str(scratch),
        inputs["installed_policy_proof.py"]["path"],
        "--installed-config",
        str(config_path),
        cwd=scratch,
    )
    receipt.update(json.loads(Path(config["result"]).read_text()))


def main() -> None:
    """Create an exclusive receipt before installing or exercising proof inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--installed-config", type=Path, help=argparse.SUPPRESS)
    for name in ("typevet-wheel", "consumer-wheel", "bridge-wheel", "receipt"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--bridge-sha256")
    arguments = parser.parse_args()
    if arguments.installed_config:
        child(arguments.installed_config)
        return
    if not all(
        (
            arguments.typevet_wheel,
            arguments.consumer_wheel,
            arguments.bridge_wheel,
            arguments.bridge_sha256,
            arguments.receipt,
        )
    ):
        parser.error("all wheel paths, --bridge-sha256 and --receipt are required")
    receipt = {"status": "started", "offline": True}
    with arguments.receipt.open("x") as output:
        output.write(json.dumps(receipt))
        output.flush()
        try:
            asyncio.run(install_and_prove(arguments, receipt))
        except BaseException as error:
            receipt.update(
                {"status": "failed", "error": f"{type(error).__name__}: {error}"}
            )
            raise
        else:
            receipt["status"] = "passed"
        finally:
            output.seek(0)
            output.truncate()
            output.write(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
