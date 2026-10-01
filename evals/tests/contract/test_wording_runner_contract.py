"""Contract: the wording runner over judgevet's fake and typevet's bridge (#308).

The same assertions run against two ``SystemOnePort`` implementations on one
shared fixture: judgevet's ``FakeSystemOnePort`` and typevet's
``TypevetSystemOnePort`` over ``ScoringJudgmentAdapter`` and
``ScriptedScoringFake``. Both answer one fixed probability, and a scripted
reflector proposes a too-long wording and then a short one. The tests prove
that the Brier reward, the length cap and the split boundary hold for both
ports. They say nothing about model quality.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import AsyncGenerator, Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import pytest
from google.adk.models import BaseLlm, LlmRequest, LlmResponse
from google.genai import types
from judgevet import NoulAnswer, SystemOneResponse
from judgevet.domain.questions import Noul
from judgevet.testing import FakeSystemOnePort

from tests.fixtures.judgevet_bridge import judgment_port
from typevet.adapters.inbound.judgevet import TypevetSystemOnePort
from typevet.testing import ScriptedScoringFake
from typevet_evals.datasets.difraud import DIFrauDRecord, map_row, record_id
from typevet_evals.wording import WordingRun, WordingRunConfig, evolve_wording

pytestmark = pytest.mark.contract

KEY = "is_scam"
P = 0.8
SEED = Noul(
    instructions="Is this message a scam?",
    criteria={"true": "It is a scam", "false": "It is legitimate"},
)
TOO_LONG = "Is this message a scam? " * 3
SHORT = "Does it ask for money?"


def _record(text: str, scam: bool, split: str) -> DIFrauDRecord:
    """Return one record whose example names ``split``."""
    return DIFrauDRecord(
        record_id(text), replace(map_row(text, int(scam)), split=split)
    )


TRAIN = (_record("Win cash now", True, "train"), _record("See you", False, "train"))
VALIDATION = (
    _record("Claim a prize", True, "validation"),
    _record("Lunch at noon", False, "validation"),
)
HELD_OUT = (_record("Send gift cards", True, "test"),)


@dataclass
class RecordingPort:
    """Keep each call's state and wording, then pass the call on.

    Attributes:
        inner (Any): The wrapped ``SystemOnePort``.
        calls (list[tuple[str, str]]): Each call's state and wording.
    """

    inner: Any
    calls: list[tuple[str, str]] = field(default_factory=list)

    def system_one(
        self, state: str, questions: Mapping[str, Any], model: str
    ) -> SystemOneResponse:
        """Record the call and delegate.

        Returns:
            The wrapped port's response.
        """
        self.calls.append((state, str(questions[KEY].instructions)))
        return self.inner.system_one(state, questions, model)


class ScriptedReflector(BaseLlm):
    """A reflection model that answers the scripted proposals in order.

    Attributes:
        proposals (list[str]): The texts still to propose; the last one repeats.
    """

    model: str = "scripted-reflector"
    proposals: list[str]

    async def generate_content_async(
        self, llm_request: LlmRequest, stream: bool = False
    ) -> AsyncGenerator[LlmResponse, None]:
        """Yield the next scripted proposal.

        Yields:
            One text response.
        """
        text = self.proposals.pop(0) if len(self.proposals) > 1 else self.proposals[0]
        part = types.Part.from_text(text=text)
        yield LlmResponse(content=types.Content(role="model", parts=[part]))


def _judgevet_fake() -> Any:
    """Return judgevet's fake port scripted to answer ``P``."""
    return FakeSystemOnePort(answers={KEY: NoulAnswer(noul=P)})


def _typevet_bridge() -> Any:
    """Return typevet's bridge over scripted logprobs that give ``P``."""
    scorer = ScriptedScoringFake(
        logprobs={"True": math.log(P), "False": math.log(1 - P)}
    )
    return TypevetSystemOnePort(judgment_port(scorer=scorer))


PORTS = pytest.mark.parametrize(
    "make_port", [_judgevet_fake, _typevet_bridge], ids=["judgevet-fake", "bridge"]
)


def _run(port: RecordingPort, tmp_path: Path) -> WordingRun:
    """Run two iterations: a too-long proposal, then a short one."""
    config = WordingRunConfig(
        reflector=ScriptedReflector(proposals=[TOO_LONG, SHORT]),
        judge_model="fake-judge",
        max_iterations=2,
        patience=2,
        checkpoint_path=tmp_path / "checkpoint.json",
    )
    return asyncio.run(
        evolve_wording(
            port=port,
            seed=SEED,
            question_name=KEY,
            train=TRAIN,
            validation=VALIDATION,
            config=config,
        )
    )


@PORTS
def test_the_brier_reward_matches_hand_computed_values(
    make_port: Callable[[], Any], tmp_path: Path
) -> None:
    run = _run(RecordingPort(make_port()), tmp_path)

    per_row = [1 - (P - 1) ** 2, 1 - (P - 0) ** 2]
    assert run.result.original_score == pytest.approx(sum(per_row))
    assert run.result.valset_score == pytest.approx(sum(per_row) / 2)
    assert run.evolved_text == SEED.instructions


@PORTS
def test_a_too_long_proposal_never_reaches_the_port(
    make_port: Callable[[], Any], tmp_path: Path
) -> None:
    port = RecordingPort(make_port())

    run = _run(port, tmp_path)

    skips = [r.skip_reason for r in run.result.iteration_history]
    assert skips[0] == "proposal_rejected"
    wordings = {wording for _, wording in port.calls}
    assert TOO_LONG not in wordings
    assert SHORT in wordings


@PORTS
def test_only_train_and_validation_texts_reach_the_port(
    make_port: Callable[[], Any], tmp_path: Path
) -> None:
    port = RecordingPort(make_port())

    _run(port, tmp_path)

    sent = {state for state, _ in port.calls}
    assert sent == {r.example.text for r in (*TRAIN, *VALIDATION)}
    assert not sent & {r.example.text for r in HELD_OUT}
