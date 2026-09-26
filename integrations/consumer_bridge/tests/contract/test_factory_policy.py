"""Exercise installed conversion through the real runtime and consumer policy.

Examples:
    ```bash
    uv run pytest -q integrations/consumer_bridge/tests/contract/test_factory_policy.py
    ```

See Also:
    - [typevet_consumer_bridge.adapter][]: Public conversion and ownership.
"""

import json

import httpx
import pytest
from conftest import Router
from judgevet import Noul, NoulAnswer
from judgevet.policy import NoulRule, Policy, evaluate_policy, validate_policy
from judgevet.ports import SystemOnePort
from typevet_consumer_bridge import (
    BridgeRequestError,
    BridgeSettings,
    open_typevet_system_one,
)

pytestmark = pytest.mark.contract
MODEL = "private-model"
INSTRUCTIONS = "  Does the invoice total equal 42 dollars?\n"
STATE = "The invoice total is 42 dollars."
SETTINGS = BridgeSettings(
    base_url="http://offline", timeout=5.0, multimodal_model=MODEL
)


def require(condition: bool, message: str) -> None:
    """Fail a named integration assertion.

    Raises:
        AssertionError: If the named contract assertion fails.
    """
    if not condition:
        raise AssertionError(message)


@pytest.mark.parametrize("positive", [True, False])
def test_real_factory_policy(router: Router, positive: bool) -> None:
    """Both policy outcomes require real engine scoring and owned answers."""
    router.positive = positive
    questions = {
        " total_is_42 ": Noul(
            instructions=INSTRUCTIONS,
            criteria={"true": "Exactly forty-two", "false": "Another total"},
        )
    }
    with httpx.Client(
        transport=httpx.MockTransport(router.handle), base_url="http://offline"
    ) as client:
        with open_typevet_system_one(settings=SETTINGS, http_client=client) as adapter:
            port: SystemOnePort = adapter
            response = port.system_one(STATE, questions, MODEL)
        require(not client.is_closed, "caller client remains open")
    require(
        type(response.answers[" total_is_42 "]) is NoulAnswer, "consumer-owned answer"
    )
    require(response.model == MODEL, "exact requested model")
    require(
        response.usage.input_tokens is None and response.usage.output_tokens is None,
        "unknown usage",
    )
    policy = validate_policy(
        Policy(rules=(NoulRule(name=" total_is_42 ", minimum=0.8),)), questions
    )
    report = evaluate_policy(policy, response.answers)
    require(report.passed is positive, "actual consumer policy outcome")
    completions = [body for path, body in router.requests if path == "/completion"]
    require(len(completions) == 1, "one scoring call")
    prompt = completions[0]["prompt"]
    require(INSTRUCTIONS in prompt, "exact caller instructions in dispatched prompt")
    require(STATE in prompt, "exact state in prompt")
    require(
        "Exactly forty-two" in prompt and "Another total" in prompt, "caller criteria"
    )
    require("image_data" not in completions[0], "text-only request")
    require(
        all(body.get("model", MODEL) == MODEL for _, body in router.requests),
        "routing identity",
    )


@pytest.mark.parametrize("invalid", ["later-question", "model", "state"])
def test_reject_before_judgment_io(router: Router, invalid: str) -> None:
    """Validate later questions and model before any request tokenization."""
    with (
        httpx.Client(
            transport=httpx.MockTransport(router.handle), base_url="http://offline"
        ) as client,
        open_typevet_system_one(settings=SETTINGS, http_client=client) as adapter,
    ):
        before = len(router.requests)
        questions = {
            "valid": Noul(),
            "later": Noul(instructions={} if invalid == "later-question" else None),
        }
        with pytest.raises(BridgeRequestError):
            adapter.system_one(
                {1: "invalid"} if invalid == "state" else STATE,
                questions,
                "other" if invalid == "model" else MODEL,
            )
        require(len(router.requests) == before, "zero judgment IO")


def test_json_state_fidelity(router: Router) -> None:
    """JSON serialization retains caller ordering and string values."""
    state = {"last": ["  preserved\n", None], "first": True}
    with (
        httpx.Client(
            transport=httpx.MockTransport(router.handle), base_url="http://offline"
        ) as client,
        open_typevet_system_one(settings=SETTINGS, http_client=client) as adapter,
    ):
        adapter.system_one(state, {"q": {"type": "noul"}}, MODEL)
    prompt = next(
        body["prompt"] for path, body in router.requests if path == "/completion"
    )
    require(json.dumps(state) in prompt, "JSON state fidelity")
