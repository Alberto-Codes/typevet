"""Exercise installed conversion through the real runtime and consumer policy.

Examples:
    ```bash
    uv run pytest -q integrations/consumer_bridge/tests/contract/test_factory_policy.py
    ```

See Also:
    - [typevet_consumer_bridge.adapter][]: Public conversion and ownership.
"""

import json
from pathlib import Path

import httpx
import pytest
from conftest import Router
from judgevet import Choice, ChoiceAnswer, Noul, NoulAnswer, Score, ScoreAnswer
from judgevet.policy import (
    ChoiceRule,
    NoulRule,
    Policy,
    ScoreRule,
    evaluate_policy,
    validate_policy,
)
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


@pytest.mark.parametrize("positive", [True, False])
@pytest.mark.parametrize("raw", [True, False])
def test_three_question_policy(router: Router, positive: bool, raw: bool) -> None:
    """Use frozen text through the real factory and all consumer policy rules."""
    case = json.loads(
        (Path(__file__).parents[1] / "fixtures/text_cases.json").read_text()
    )
    forms = {"noul": Noul, "choice": Choice, "score": Score}
    questions = (
        case["questions"]
        if raw
        else {
            key: forms[value["type"]](
                instructions=value["instructions"], criteria=value["criteria"]
            )
            for key, value in case["questions"].items()
        }
    )
    policy_questions = {
        key: forms[value["type"]](
            instructions=value["instructions"], criteria=value["criteria"]
        )
        for key, value in case["questions"].items()
    }
    router.positive = positive
    with (
        httpx.Client(
            transport=httpx.MockTransport(router.handle), base_url="http://offline"
        ) as client,
        open_typevet_system_one(settings=SETTINGS, http_client=client) as adapter,
    ):
        response = adapter.system_one(case["state"], questions, MODEL)
    answers = response.answers
    require(list(answers) == list(questions), "exact answer IDs")
    for key, expected in zip(
        questions, (NoulAnswer, ChoiceAnswer, ScoreAnswer), strict=True
    ):
        require(type(answers[key]) is expected, "consumer answer variant")
    choice, score = answers["amount"], answers["amount_score"]
    if not isinstance(choice, ChoiceAnswer) or not isinstance(score, ScoreAnswer):
        raise TypeError("Incorrect consumer answer variants")
    require(list(choice.probabilities) == ["seven", "forty_two"], "ordered labels")
    require(
        score.legend == {0: "seven dollars", 1: "forty-two dollars"},
        "zero-based legend",
    )
    require(score.score == score.probabilities[1], "probability-weighted score")
    require(score.confidence == max(score.probabilities.values()), "maximum confidence")
    require(0 < score.score < 1, "score is not modal level")
    policy = validate_policy(
        Policy(
            rules=(
                NoulRule("total_is_42", **case["policy"]["total_is_42"]),
                ChoiceRule("amount", **case["policy"]["amount"]),
                ScoreRule("amount_score", **case["policy"]["amount_score"]),
            )
        ),
        policy_questions,
    )
    require(
        evaluate_policy(policy, answers).passed is positive, "three-rule policy outcome"
    )
    completions = [body for path, body in router.requests if path == "/completion"]
    require(len(completions) == len(questions), "three scoring calls")
    for question, body in zip(case["questions"].values(), completions, strict=True):
        prompt = body["prompt"]
        require(question["instructions"] in prompt, "exact caller instruction")
        require(case["state"] in prompt, "exact text state")
        criteria = question["criteria"]
        descriptions = criteria.values() if isinstance(criteria, dict) else criteria
        require(
            all(value in prompt for value in descriptions), "all rubric descriptions"
        )
        require("image_data" not in body, "no media")


@pytest.mark.parametrize(
    "later",
    [
        {"type": "choice", "criteria": {"a": {}, "b": None}},
        {"type": "choice", "criteria": {str(i): None for i in range(25)}},
        {"type": "score", "criteria": ["a", []]},
        {"type": "score", "criteria": ["a"] * 25},
        {"type": "choice", "instructions": [], "criteria": {"a": None, "b": None}},
    ],
)
def test_later_unsupported_zero_io(router: Router, later: object) -> None:
    """Validate every option before any judgment IO."""
    with (
        httpx.Client(
            transport=httpx.MockTransport(router.handle), base_url="http://offline"
        ) as client,
        open_typevet_system_one(settings=SETTINGS, http_client=client) as adapter,
    ):
        before = len(router.requests)
        with pytest.raises(BridgeRequestError):
            adapter.system_one(STATE, {"valid": Noul(), "later": later}, MODEL)
        require(len(router.requests) == before, "whole request zero IO")
