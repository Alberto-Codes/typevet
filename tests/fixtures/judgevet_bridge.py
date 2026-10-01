"""Offline typevet judgment ports for the judgevet bridge tests (#284).

Every port here runs ``ScoringJudgmentAdapter`` over ``ScriptedScoringFake``,
so the bridge tests exercise the real typevet judgment path with no network.
The questions match the questions of the judgevet provider conformance kit
(``judgevet.testing.conformance``, judgevet 0.17.0).
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass

from judgevet.domain.questions import Choice, Noul, Question, Score

from typevet.adapters.outbound.gemma import ServedTemplateClass
from typevet.domain.errors import ScoringError
from typevet.runtime import ScoringJudgmentAdapter
from typevet.testing import ScriptedScoringFake

FAKE_MODEL = "fake-gemma"
"""The model id the scripted scorer reports back."""

STATE = "A customer reports that one invoice was charged twice."
"""The synthetic state every bridge judgment receives."""

QUESTIONS: Mapping[str, Question] = {
    "conformance_noul": Noul(instructions="Does the state report a duplicate charge?"),
    "conformance_choice": Choice(
        criteria={"billing": "A payment problem", "technical": "A product fault"},
        instructions="Which team should handle the state?",
    ),
    "conformance_score": Score(
        criteria=["calm", "concerned", "angry"],
        instructions="How upset is the customer?",
    ),
}
"""One judgevet question of each kind, keyed by question name."""

LOGPROBS: Mapping[str, float] = {
    "True": math.log(0.8),
    "False": math.log(0.2),
    "billing": math.log(0.75),
    "technical": math.log(0.25),
    "0": math.log(0.2),
    "1": math.log(0.5),
    "2": math.log(0.3),
}
"""Scripted logprobs for every candidate label the questions bind."""


def single_token(text: str) -> tuple[int, ...]:
    """Encode each character as one token, so ``"10"`` is two tokens.

    Args:
        text: A control string such as ``"0"``.

    Returns:
        One token id per character.
    """
    return tuple(ord(char) for char in text)


def judgment_port(
    *,
    fail: ScoringError | None = None,
    media: bool = False,
    scorer: ScriptedScoringFake | None = None,
) -> ScoringJudgmentAdapter:
    """Build a typevet ``JudgmentPort`` over scripted logprobs.

    Args:
        fail: A scoring error every call raises, or None to answer.
        media: Use the native Gemma 4 served template, which accepts images.
        scorer: The scripted scorer to wrap, or None for a new one.

    Returns:
        A judgment port that makes no network call.
    """
    template = ServedTemplateClass.NATIVE_GEMMA4_TURN if media else None
    if scorer is None:
        scorer = ScriptedScoringFake(logprobs=LOGPROBS, fail=fail)
    return ScoringJudgmentAdapter(
        scorer,
        tokenize_content=single_token,
        served_template=template,
    )


@dataclass
class FakeSession:
    """A stand-in for a typevet judgment session with ``port`` and ``model``.

    Attributes:
        port (ScoringJudgmentAdapter): The offline judgment port.
        model (str): The model id the session serves.
        closed (bool): Whether the owning context has exited.
    """

    port: ScoringJudgmentAdapter
    model: str = FAKE_MODEL
    closed: bool = False


@contextmanager
def open_fake_session(
    sessions: list[FakeSession] | None = None,
    *,
    media: bool = False,
) -> Iterator[FakeSession]:
    """Open one offline session, recording it for lifetime checks.

    Args:
        sessions: A list that receives each opened session, or None.
        media: Serve the native Gemma 4 template, which accepts images.

    Yields:
        A new session over a fresh offline judgment port.
    """
    session = FakeSession(judgment_port(media=media))
    if sessions is not None:
        sessions.append(session)
    try:
        yield session
    finally:
        session.closed = True
