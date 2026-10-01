"""Contract: the wording transport over judgevet's fake and typevet's bridge (#306).

The same assertions run against two ``SystemOnePort`` implementations on one
shared fixture: judgevet's ``FakeSystemOnePort`` and typevet's
``TypevetSystemOnePort`` over ``ScoringJudgmentAdapter`` and
``ScriptedScoringFake``. A recording wrapper keeps the questions each port
receives. The tests prove that the transport reads every part of the
caller-owned mapping (#363) at call time, returns the port's probability and
lets concurrent evaluations overlap. They say nothing about model quality.
"""

from __future__ import annotations

import asyncio
import json
import math
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import pytest
from google.adk.models import LlmRequest, LlmResponse
from google.genai import types
from judgevet import NoulAnswer, SystemOneResponse
from judgevet.domain.questions import Noul
from judgevet.testing import FakeSystemOnePort

from tests.fixtures.judgevet_bridge import judgment_port
from typevet.adapters.inbound.judgevet import TypevetSystemOnePort
from typevet.testing import ScriptedScoringFake
from typevet_evals.wording import WordingTransport

pytestmark = pytest.mark.contract

KEY = "is_scam"
MODEL = "fake-judge"
STATE = "Win cash now"
CRITERIA = {"true": "It is a scam", "false": "It is legitimate"}
SEED = Noul(instructions="Is this message a scam?", criteria=CRITERIA)
WORDINGS = (
    "Is this message a scam?",
    "Does this text try to trick the reader out of money?",
)
PROBABILITIES = (0.8, 0.3)
BARRIER_TIMEOUT = 5.0


@dataclass
class RecordingPort:
    """Keep each call's questions, then pass the call to the wrapped port.

    Attributes:
        inner (Any): The wrapped ``SystemOnePort``.
        barrier (threading.Barrier | None): A barrier every call waits on first.
        calls (list[tuple[str, dict[str, Any], str]]): Each call's arguments.
    """

    inner: Any
    barrier: threading.Barrier | None = None
    calls: list[tuple[str, dict[str, Any], str]] = field(default_factory=list)

    def system_one(
        self, state: str, questions: Mapping[str, Any], model: str
    ) -> SystemOneResponse:
        """Record the call, wait on the barrier and delegate.

        Returns:
            The wrapped port's response.
        """
        self.calls.append((state, dict(questions), model))
        if self.barrier is not None:
            self.barrier.wait()
        return self.inner.system_one(state, questions, model)


def _judgevet_fake(p: float) -> tuple[Any, Callable[[int], str] | None]:
    """Return judgevet's fake port scripted to answer ``p``."""
    return FakeSystemOnePort(answers={KEY: NoulAnswer(noul=p)}), None


def _typevet_bridge(p: float) -> tuple[Any, Callable[[int], str] | None]:
    """Return typevet's bridge over scripted logprobs that give ``p``.

    The second item reads the prompt prefix of scoring call ``i``, so the
    test sees what reached the typevet backend.
    """
    scorer = ScriptedScoringFake(
        logprobs={"True": math.log(p), "False": math.log(1 - p)}
    )
    port = TypevetSystemOnePort(judgment_port(scorer=scorer))
    return port, lambda i: scorer.calls[i].prefix


PORTS = pytest.mark.parametrize(
    "make_port", [_judgevet_fake, _typevet_bridge], ids=["judgevet-fake", "bridge"]
)


def _request(text: str) -> LlmRequest:
    """Return an ADK request whose last user turn is ``text``."""
    return LlmRequest(
        contents=[types.Content(role="user", parts=[types.Part.from_text(text=text)])]
    )


async def _turn(transport: WordingTransport) -> list[LlmResponse]:
    """Drive one model turn and collect every response."""
    return [r async for r in transport.generate_content_async(_request(STATE))]


def _parts(instructions: str) -> dict[str, str]:
    """Return the full part mapping with ``instructions`` and the fixture criteria."""
    return {
        "instructions": instructions,
        "criteria_true": CRITERIA["true"],
        "criteria_false": CRITERIA["false"],
    }


def _transport(port: RecordingPort, mapping: dict[str, str]) -> WordingTransport:
    """Build the transport over the shared fixture."""
    return WordingTransport(
        port=port, mapping=mapping, question_name=KEY, seed=SEED, judge_model=MODEL
    )


@PORTS
@pytest.mark.parametrize("p", PROBABILITIES)
def test_each_call_sends_the_current_mapping_value(
    make_port: Callable[[float], tuple[Any, Callable[[int], str] | None]], p: float
) -> None:
    inner, prefix = make_port(p)
    port = RecordingPort(inner)
    mapping = _parts(WORDINGS[0])
    transport = _transport(port, mapping)

    first = asyncio.run(_turn(transport))
    mapping["instructions"] = WORDINGS[1]
    second = asyncio.run(_turn(transport))

    sent = [call[1][KEY] for call in port.calls]
    assert all(isinstance(q, Noul) and q.criteria == CRITERIA for q in sent)
    assert [q.instructions for q in sent] == list(WORDINGS)
    assert all(call[0] == STATE and call[2] == MODEL for call in port.calls)
    if prefix is not None:
        assert WORDINGS[0] in prefix(0) and WORDINGS[1] not in prefix(0)
        assert WORDINGS[1] in prefix(1)
    for [response] in (first, second):
        assert response.content is not None and response.content.parts
        body = json.loads(response.content.parts[0].text or "")
        assert set(body) == {"probability"}
        assert body["probability"] == pytest.approx(p)


@PORTS
def test_concurrent_evaluations_overlap(
    make_port: Callable[[float], tuple[Any, Callable[[int], str] | None]],
) -> None:
    calls = 3
    inner, _ = make_port(PROBABILITIES[0])
    port = RecordingPort(inner, threading.Barrier(calls, timeout=BARRIER_TIMEOUT))
    transport = _transport(port, _parts(WORDINGS[0]))

    async def gather() -> list[list[LlmResponse]]:
        return list(await asyncio.gather(*(_turn(transport) for _ in range(calls))))

    assert [len(turn) for turn in asyncio.run(gather())] == [1] * calls
