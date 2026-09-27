"""Run one installed text judgment with admitted-attempt budgets and a receipt.

Examples:
    Run ``uv run python live_text_proof.py --help`` for required frozen inputs.

See Also:
    - [installed_policy_proof][]: Offline wheel and installation attestation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

LIMITS = {"judgment": 1, "scoring": 3, "metadata": 2, "tokenizer": 6, "completion": 3}
ROUTES = {
    ("GET", "/props"): "metadata",
    ("POST", "/apply-template"): "metadata",
    ("POST", "/tokenize"): "tokenizer",
    ("POST", "/completion"): "completion",
}
INPUTS = {
    "installed_policy_proof.py": "7380aab4cc537dabb6457dc033652aa79731c6720043c6a2146f69bce994a5ca",
    "verify_install.py": "21563c0b391569ebb27f9f032510d3675016644fa1f491d05cbd4f71e03e1d86",
    "text_cases.json": "a6d40930345351e64fb65b669982655cfa012b19d4a0efb83ba0b00380a1d94a",
}


def require(condition: bool, message: str) -> None:
    """Reject a failed proof condition.

    Raises:
        AssertionError: When the condition fails.
    """
    if not condition:
        raise AssertionError(message)


def digest(path: Path) -> str:
    """Hash exact file bytes.

    Returns:
        The hexadecimal SHA256 digest.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Budget:
    """Reserve each attempt before dispatch and retain failed attempts.

    Attributes:
        receipt (dict): Mutable attempt ledger.
        base_url (str): Frozen HTTP origin.
        model (str): Requested model identity.
        sealed (bool): Whether admission has ended.
        case (dict | None): Frozen caller inputs.

    Examples:
        ``Budget({}, "http://offline.invalid").reserve("judgment")``
    """

    def __init__(
        self, receipt: dict, base_url: str, model: str = "", case: dict | None = None
    ) -> None:
        """Create counters and append-only request evidence."""
        self.receipt = receipt
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.sealed = False
        self.case = case
        receipt.update(counts=dict.fromkeys(LIMITS, 0), limits=LIMITS.copy(), calls=[])

    def reserve(self, category: str) -> None:
        """Refuse exhausted counters before their next operation."""
        require(not self.sealed, "attempt budget sealed")
        require(category in LIMITS, "unknown attempt category")
        counts = self.receipt["counts"]
        require(counts[category] < LIMITS[category], "attempt budget exhausted")
        counts[category] += 1

    def request(self, request) -> None:
        """Count only approved same-origin routes before transport dispatch."""
        category = ROUTES.get((request.method, request.url.path))
        require(category is not None, "unknown HTTP route")
        require(
            str(request.url.copy_with(query=None)) == self.base_url + request.url.path,
            "exact endpoint",
        )
        require(
            dict(request.url.params)
            == ({"model": self.model} if request.url.path == "/props" else {}),
            "exact request parameters",
        )
        self.reserve(str(category))
        self.receipt["calls"].append(
            {
                "method": request.method,
                "url": str(request.url),
                "path": request.url.path,
                "body": json.loads(request.content) if request.content else {},
                "state": "admitted",
            }
        )
        body = self.receipt["calls"][-1]["body"]
        require(
            not any(
                key in json.dumps(body)
                for key in ("image_data", "multimodal_data", "<__media__>")
            ),
            "text-only request",
        )
        require(body.get("model", self.model) == self.model, "wire requested model")
        if request.url.path == "/completion" and self.case is not None:
            index = self.receipt["counts"]["completion"] - 1
            question = list(self.case["questions"].values())[index]
            check_prompt(self.case, question, body)

    def response(self, response) -> None:
        """Retain complete response text before runtime decoding can fail."""
        response.read()
        self.receipt["calls"][-1].update(
            state="responded", status=response.status_code, response=response.text
        )

    def wrap(self, port):
        """Wrap the actual scoring port supplied by the public factory.

        Returns:
            A forwarding port with admission accounting.
        """
        budget = self

        class CountedScoring:
            """Reserve scoring entries before the actual port runs.

            Examples:
                The public factory uses ``score_candidates(request)``.
            """

            def score_candidates(self, request):
                """Count failed and successful scoring attempts equally.

                Returns:
                    The actual scoring response without modification.
                """
                budget.reserve("scoring")
                return port.score_candidates(request)

        return CountedScoring()


def verified_inputs(arguments: argparse.Namespace, receipt: dict):
    """Bind an accepted offline receipt to its installed bytes and frozen inputs.

    Returns:
        Verified helper, configuration, and frozen case.
    """
    require(digest(Path(__file__)) == arguments.script_sha256, "live script bytes")
    require(
        digest(arguments.offline_receipt) == arguments.offline_receipt_sha256,
        "offline receipt bytes",
    )
    offline = json.loads(arguments.offline_receipt.read_text())
    require(offline["status"] == "passed" and offline["offline"], "offline acceptance")
    require(
        Path(sys.prefix).resolve() == Path(offline["installation"]["prefix"]).resolve(),
        "accepted installed interpreter",
    )
    for name, expected in INPUTS.items():
        require(digest(Path(offline["inputs"][name]["path"])) == expected, name)
    helper_dir = Path(offline["inputs"]["installed_policy_proof.py"]["path"]).parent
    require(
        Path(offline["inputs"]["verify_install.py"]["path"]).parent == helper_dir,
        "helper directory",
    )
    config_path = Path(offline["scratch"]) / "config.json"
    require(
        digest(config_path) == arguments.installed_config_sha256,
        "installed config bytes",
    )
    config = json.loads(config_path.read_text())
    require(
        not Path(__file__).resolve().is_relative_to(Path(config["checkout"])),
        "externally staged live script",
    )
    require(config["artifacts"] == offline["artifacts"], "artifact manifest")
    require(
        config["fixture"] == offline["inputs"]["text_cases.json"]["path"], "fixture"
    )
    sys.path.insert(0, str(helper_dir))
    helper = importlib.import_module("installed_policy_proof")
    for name in ("installed_policy_proof", "verify_install"):
        require(
            Path(str(sys.modules[name].__file__)).resolve()
            == helper_dir / (name + ".py"),
            "loaded helper identity",
        )
    helper.attest_bytes(config)
    for name in helper.PACKAGES:
        importlib.import_module(name)
    receipt.update(
        artifacts=offline["artifacts"],
        inputs=offline["inputs"],
        script_sha256=arguments.script_sha256,
        installed_config_sha256=arguments.installed_config_sha256,
        offline_receipt_sha256=arguments.offline_receipt_sha256,
        installation_before=helper.attest(config),
    )
    return helper, config, json.loads(Path(config["fixture"]).read_text())


def validate_wire(case: dict, receipt: dict, model: str) -> None:
    """Check exact caller text, instructions, criteria and requested model."""
    require(receipt["counts"] == LIMITS, "complete attempt counts")
    calls = receipt["calls"]
    completions = [call["body"] for call in calls if call["path"] == "/completion"]
    for question, body in zip(case["questions"].values(), completions, strict=True):
        check_prompt(case, question, body)
    require(
        not any(
            key in json.dumps([call["body"] for call in calls])
            for key in ("image_data", "multimodal_data", "<__media__>")
        ),
        "text only",
    )
    require(all(call["body"].get("model", model) == model for call in calls), "model")


def check_prompt(case: dict, question: dict, body: dict) -> None:
    """Reject modified caller inputs before the completion transport runs."""
    require(isinstance(body["prompt"], str), "text prompt")
    require(case["state"] in body["prompt"], "exact caller state")
    require(question["instructions"] in body["prompt"], "exact caller instructions")
    criteria = question["criteria"]
    labels = criteria.values() if isinstance(criteria, dict) else criteria
    require(all(label in body["prompt"] for label in labels), "exact caller criteria")


def run_case(
    case: dict, arguments: argparse.Namespace, receipt: dict, transport=None
) -> None:
    """Exercise the real factory and bridge once, with caller-owned HTTP cleanup."""
    httpx = importlib.import_module("httpx")
    consumer = importlib.import_module("judgevet")
    policy = importlib.import_module("judgevet.policy")
    bridge = importlib.import_module("typevet_consumer_bridge")
    runtime = importlib.import_module("typevet.runtime")
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
    budget = Budget(receipt, arguments.endpoint, arguments.model, case)
    settings = bridge.BridgeSettings(
        arguments.endpoint, arguments.timeout, arguments.model
    )
    client = httpx.Client(
        base_url=settings.base_url,
        timeout=settings.timeout,
        transport=transport,
        follow_redirects=False,
        trust_env=False,
        event_hooks={"request": [budget.request], "response": [budget.response]},
    )
    adapter = None
    receipt["cleanup"] = {"client_closed": False, "adapter_close_called": False}
    try:
        with (
            client,
            runtime.open_gemma_native_vision_judgment(
                settings=settings,
                model=arguments.model,
                http_client=client,
                scoring_port_wrapper=budget.wrap,
            ) as session,
        ):
            adapter = bridge.TypevetSystemOneAdapter(session)
            try:
                budget.reserve("judgment")
                response = adapter.system_one(
                    case["state"], case["questions"], arguments.model
                )
            finally:
                adapter.close()
                receipt["cleanup"]["adapter_close_called"] = True
            receipt["runtime_exercised"] = True
            receipt["response"] = asdict(response)
            report = policy.evaluate_policy(validated, response.answers)
            receipt.update(policy=asdict(report), passed=report.passed)
            require(type(response) is consumer.SystemOneResponse, "consumer response")
            require(response.model == arguments.model, "requested model")
            require(list(response.answers) == list(questions), "answer identities")
            require(
                [type(answer) for answer in response.answers.values()]
                == [consumer.NoulAnswer, consumer.ChoiceAnswer, consumer.ScoreAnswer],
                "consumer answer types",
            )
            require(
                asdict(response.usage) == {"input_tokens": None, "output_tokens": None},
                "unknown usage",
            )
            validate_wire(case, receipt, arguments.model)
            receipt["status"] = "passed" if report.passed else "policy_failed"
    finally:
        budget.sealed = True
        receipt["cleanup"]["client_closed"] = client.is_closed
        require(client.is_closed, "caller client must close")


def execute(arguments: argparse.Namespace, receipt: dict) -> None:
    """Retain failures and recheck installed bytes after every attempted judgment."""
    helper, config, case = verified_inputs(arguments, receipt)
    receipt.update(
        endpoint=arguments.endpoint,
        requested_model=arguments.model,
        served_weights="unknown",
        usage="unknown",
        fixture=case,
    )
    try:
        run_case(case, arguments, receipt)
    finally:
        receipt["installation_after"] = helper.attest(config)
        require(digest(Path(__file__)) == arguments.script_sha256, "final script bytes")
        require(
            digest(arguments.offline_receipt) == arguments.offline_receipt_sha256,
            "final offline receipt bytes",
        )
        offline = json.loads(arguments.offline_receipt.read_text())
        require(
            digest(Path(offline["scratch"]) / "config.json")
            == arguments.installed_config_sha256,
            "final config bytes",
        )
        for name, expected in INPUTS.items():
            require(
                digest(Path(receipt["inputs"][name]["path"])) == expected,
                "final input bytes",
            )


def record_failure(error: BaseException, receipt: dict) -> None:
    """Distinguish transport connection failures from runtime and policy failures."""
    httpx = sys.modules.get("httpx")
    errors = []
    current = error
    while current is not None:
        errors.append(current)
        current = current.__context__
    receipt.update(
        status="service_unavailable"
        if httpx is not None and any(isinstance(e, httpx.ConnectError) for e in errors)
        else "runtime_failed",
        error=f"{type(error).__name__}: {error}",
        exception_chain=[{"type": type(e).__name__, "message": str(e)} for e in errors],
    )


def main() -> None:
    """Write an exclusive receipt even when setup, tokenization or scoring fails.

    Raises:
        BaseException: After preserving a failed proof receipt.
        SystemExit: When the consumer policy fails.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("offline-receipt", "receipt"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in (
        "offline-receipt-sha256",
        "installed-config-sha256",
        "script-sha256",
        "endpoint",
        "model",
    ):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--timeout", type=float, default=60.0)
    arguments = parser.parse_args()
    receipt = {
        "status": "started",
        "live": True,
        "started_at": datetime.now(UTC).isoformat(),
        "counts": dict.fromkeys(LIMITS, 0),
        "limits": LIMITS.copy(),
        "calls": [],
    }
    with arguments.receipt.open("x") as output:
        output.write(json.dumps(receipt))
        output.flush()
        try:
            execute(arguments, receipt)
        except BaseException as error:
            record_failure(error, receipt)
            raise
        finally:
            receipt["finished_at"] = datetime.now(UTC).isoformat()
            output.seek(0)
            output.truncate()
            output.write(json.dumps(receipt, indent=2) + "\n")
    if receipt["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
