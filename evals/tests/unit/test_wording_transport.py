"""Unit checks for the wording transport's helpers and construction (#306).

Examples:
    ```bash
    uv run pytest -q evals/tests/unit/test_wording_transport.py
    ```

See Also:
    - [typevet_evals.wording.transport][]: the transport module
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import pytest
from google.adk.models import LlmRequest, LlmResponse
from google.genai import types
from judgevet import NoulAnswer, SystemOneResponse, Usage
from judgevet.domain.questions import Choice, Noul, Score

from typevet_evals.wording import WordingTransport, last_user_text, usage_metadata
from typevet_evals.wording.parts import PART_NAMES, seed_mapping

pytestmark = pytest.mark.unit

KEY = "is_scam"
SENTINEL = "SENTINEL-WORDING-TEXT"
SEED = Noul(instructions="Is this message a scam?")
FULL_SEED = Noul(
    instructions="Is this message a scam?",
    criteria={"true": "It is a scam", "false": "It is legitimate"},
)


@dataclass
class RecordingPort:
    """Record each call and answer 0.7.

    Attributes:
        calls (list[tuple[str, Mapping[str, Any], str]]): Each call's arguments.
    """

    calls: list[tuple[str, Mapping[str, Any], str]] = field(default_factory=list)

    def system_one(
        self, state: str, questions: Mapping[str, Any], model: str
    ) -> SystemOneResponse:
        """Record the call.

        Returns:
            One Noul answer under ``KEY``.
        """
        self.calls.append((state, dict(questions), model))
        return SystemOneResponse(
            model=model, usage=Usage(), answers={KEY: NoulAnswer(noul=0.7)}
        )


async def _turn(transport: WordingTransport) -> list[LlmResponse]:
    """Drive one model turn about ``Win cash now``."""
    request = LlmRequest(contents=[_content("user", "Win cash now")])
    return [r async for r in transport.generate_content_async(request)]


def _content(role: str, text: str) -> types.Content:
    """Return one content with one text part."""
    return types.Content(role=role, parts=[types.Part.from_text(text=text)])


def test_last_user_text_reads_the_last_user_turn() -> None:
    request = LlmRequest(
        contents=[_content("user", "a"), _content("user", "b"), _content("model", "c")]
    )

    assert last_user_text(request) == "b"


def test_a_request_without_user_text_is_refused() -> None:
    request = LlmRequest(contents=[_content("model", "x")])

    with pytest.raises(ValueError, match="no user text"):
        last_user_text(request)


def test_usage_metadata_sums_the_counts() -> None:
    usage = usage_metadata(Usage(input_tokens=7, output_tokens=1))

    assert usage is not None
    assert (usage.prompt_token_count, usage.candidates_token_count) == (7, 1)
    assert usage.total_token_count == 8


@pytest.mark.parametrize("usage", [None, Usage()])
def test_no_counts_give_no_usage_metadata(usage: Usage | None) -> None:
    assert usage_metadata(usage) is None


def _transport(mapping: dict[str, str], seed: Any = FULL_SEED) -> WordingTransport:
    """Return a transport over the recording port."""
    return WordingTransport(
        port=RecordingPort(),
        mapping=mapping,
        question_name=KEY,
        seed=seed,
        judge_model="m",
    )


def test_seed_mapping_names_each_part_of_a_noul() -> None:
    assert seed_mapping(FULL_SEED) == {
        "instructions": "Is this message a scam?",
        "criteria_true": "It is a scam",
        "criteria_false": "It is legitimate",
    }
    assert seed_mapping(SEED) == {"instructions": "Is this message a scam?"}
    assert PART_NAMES == ("instructions", "criteria_true", "criteria_false")


def test_each_call_renders_every_part_from_the_mapping() -> None:
    mapping = seed_mapping(FULL_SEED)
    transport = _transport(mapping)
    mapping.update(
        instructions="Does it ask for money?",
        criteria_true="It asks for money",
        criteria_false="It asks for nothing",
    )

    [response] = asyncio.run(_turn(transport))

    port = transport.port
    assert isinstance(port, RecordingPort)
    [(state, questions, model)] = port.calls
    assert (state, model, list(questions)) == ("Win cash now", "m", [KEY])
    noul = questions[KEY]
    assert isinstance(noul, Noul)
    assert noul.instructions == "Does it ask for money?"
    assert noul.criteria == {
        "true": "It asks for money",
        "false": "It asks for nothing",
    }
    assert response.content is not None


def test_a_seed_without_criteria_sends_no_criteria() -> None:
    transport = _transport(seed_mapping(SEED), seed=SEED)

    asyncio.run(_turn(transport))

    port = transport.port
    assert isinstance(port, RecordingPort)
    assert port.calls[0][1][KEY].criteria is None


@pytest.mark.parametrize(
    ("change", "name"),
    [
        ({"criteria_maybe": SENTINEL}, "criteria_maybe"),
        ({"is_scam": SENTINEL}, "is_scam"),
    ],
)
def test_an_unknown_key_is_refused_by_name_not_text(
    change: dict[str, str], name: str
) -> None:
    mapping = seed_mapping(FULL_SEED) | change

    with pytest.raises(ValueError, match=f"'{name}'") as caught:
        _transport(mapping)
    assert SENTINEL not in str(caught.value)


def test_a_criteria_part_of_a_seed_without_criteria_is_refused() -> None:
    mapping = seed_mapping(SEED) | {"criteria_true": SENTINEL}

    with pytest.raises(ValueError, match="'criteria_true'") as caught:
        _transport(mapping, seed=SEED)
    assert SENTINEL not in str(caught.value)


@pytest.mark.parametrize("name", ["instructions", "criteria_true", "criteria_false"])
def test_a_missing_part_is_refused_by_name_not_text(name: str) -> None:
    mapping = {k: SENTINEL for k in seed_mapping(FULL_SEED) if k != name}

    with pytest.raises(ValueError, match=f"'{name}'") as caught:
        _transport(mapping)
    assert SENTINEL not in str(caught.value)


def test_a_part_that_is_not_text_is_refused_by_name() -> None:
    mapping: dict[str, Any] = seed_mapping(FULL_SEED) | {"criteria_false": 3}

    with pytest.raises((TypeError, ValueError), match="'criteria_false'"):
        _transport(mapping)


@pytest.mark.parametrize(
    "seed",
    [Score(instructions=SENTINEL, criteria=[SENTINEL, "high"])],
    ids=["score"],
)
def test_a_score_seed_is_refused_by_type_not_text(seed: Any) -> None:
    name = type(seed).__name__

    with pytest.raises(ValueError, match=name) as caught:
        seed_mapping(seed)
    assert SENTINEL not in str(caught.value)
    with pytest.raises(ValueError, match=name):
        _transport({"instructions": "x"}, seed=seed)


def test_a_choice_seed_maps_to_its_parts() -> None:
    seed = Choice(instructions="Which?", criteria={"a": "A", "b": None})

    assert seed_mapping(seed) == {"instructions": "Which?", "criteria_a": "A"}


@pytest.mark.parametrize(
    "criteria",
    [
        {"true": "yes"},
        {"true": "yes", "false": "no", "maybe": SENTINEL},
        {"true": 1, "false": "no"},
    ],
)
def test_a_noul_seed_with_odd_criteria_is_refused(criteria: dict[str, Any]) -> None:
    with pytest.raises((TypeError, ValueError), match="criteria") as caught:
        seed_mapping(Noul(instructions="q", criteria=criteria))
    assert SENTINEL not in str(caught.value)


def test_a_construction_error_does_not_echo_the_input() -> None:
    seed = Noul(instructions="q")

    # pydantic shortens a long input in its message; this input is short.
    with pytest.raises(ValueError, match="'b'") as caught:
        WordingTransport(
            mapping={"b": "LEAK"}, port=0, question_name=KEY, seed=seed, judge_model="m"
        )
    assert "LEAK" not in str(caught.value)


def test_the_mapping_is_held_by_reference() -> None:
    mapping = seed_mapping(FULL_SEED)
    transport = _transport(mapping)

    assert transport.mapping is mapping
    assert transport.capabilities.output_schema_and_tools is False


def test_the_transport_asks_under_the_question_name() -> None:
    port = RecordingPort()
    transport = WordingTransport(
        port=port,
        mapping=seed_mapping(SEED),
        question_name=KEY,
        seed=SEED,
        judge_model="m",
    )

    asyncio.run(_turn(transport))

    assert transport.question_name == KEY
    assert list(port.calls[0][1]) == [KEY]
