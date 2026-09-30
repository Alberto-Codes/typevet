"""Provider conformance rules for the typevet judgevet bridge (#284).

judgevet 0.14.0 does not publish ``judgevet.testing.conformance``. The kit is
on judgevet main (3b36650) and is unreleased. This module checks the same five
rules with the public judgevet 0.14 API, one test per rule, over offline typevet
fakes. Replace it with a ``BaseProviderConformance`` subclass when a judgevet
release in the supported range ships the kit.
"""

from __future__ import annotations

import inspect
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager

import pytest
from judgevet.domain.answers import ChoiceAnswer, NoulAnswer, ScoreAnswer
from judgevet.domain.media import ImageAttachment, ImageEvidence, MediaCapabilities
from judgevet.domain.response import SystemOneResponse
from judgevet.media import judge_with_images
from judgevet.policy import (
    ChoiceRule,
    NoulRule,
    Policy,
    ScoreRule,
    evaluate_policy,
    validate_policy,
)
from judgevet.ports import SystemOnePort
from judgevet.providers import (
    ProviderCapabilityError,
    ProviderError,
    ProviderFactory,
    provider_scope,
)

from tests.fixtures.judgevet_bridge import (
    FAKE_MODEL,
    LOGPROBS,
    QUESTIONS,
    STATE,
    FakeSession,
    judgment_port,
    open_fake_session,
)
from typevet.adapters.inbound.judgevet import (
    TypevetMediaSystemOnePort,
    TypevetSystemOnePort,
    provider_factory,
)
from typevet.domain.errors import ScoringError
from typevet.testing import ScriptedScoringFake

pytestmark = pytest.mark.contract

POLICY = validate_policy(
    Policy(
        (
            NoulRule("conformance_noul", minimum=0.0),
            ChoiceRule("conformance_choice", "billing"),
            ScoreRule("conformance_score", minimum=0.0),
        )
    ),
    QUESTIONS,
)
"""A policy that accepts any well-typed answer to the questions."""


def _judge(port: SystemOnePort) -> SystemOneResponse:
    return port.system_one(state=STATE, questions=QUESTIONS, model=FAKE_MODEL)


class _Recorder:
    """Wrap a factory and record each context's entries and exits."""

    def __init__(self, factory: ProviderFactory) -> None:
        self.factory = factory
        self.exits: list[list[BaseException | None]] = []
        self.contexts: list[AbstractContextManager[SystemOnePort]] = []

    def __call__(self) -> AbstractContextManager[SystemOnePort]:
        inner = self.factory()
        self.contexts.append(inner)
        exits: list[BaseException | None] = []
        self.exits.append(exits)

        @contextmanager
        def recorded() -> Iterator[SystemOnePort]:
            port = inner.__enter__()
            try:
                yield port
            except BaseException as exc:
                exits.append(exc)
                inner.__exit__(type(exc), exc, exc.__traceback__)
                raise
            exits.append(None)
            inner.__exit__(None, None, None)

        return recorded()


@pytest.fixture
def sessions() -> list[FakeSession]:
    return []


@pytest.fixture
def factory(sessions: list[FakeSession]) -> ProviderFactory:
    return provider_factory(lambda: open_fake_session(sessions))


def test_port_shape(factory: ProviderFactory) -> None:
    with provider_scope(factory=factory) as port:
        signature = inspect.signature(port.system_one)
    signature.bind(STATE, QUESTIONS, FAKE_MODEL)
    signature.bind(state=STATE, questions=QUESTIONS, model=FAKE_MODEL)


def test_typed_answers_pass_policy(factory: ProviderFactory) -> None:
    with provider_scope(factory=factory) as port:
        response = _judge(port)
    assert isinstance(response, SystemOneResponse)
    assert set(response.answers) == set(QUESTIONS)
    assert isinstance(response.answers["conformance_noul"], NoulAnswer)
    assert isinstance(response.answers["conformance_choice"], ChoiceAnswer)
    assert isinstance(response.answers["conformance_score"], ScoreAnswer)
    assert evaluate_policy(POLICY, response.answers).passed
    assert response.model == FAKE_MODEL


def test_failure_raises_provider_error() -> None:
    port = TypevetSystemOnePort(judgment_port(fail=ScoringError("offline")))
    with pytest.raises(ProviderError):
        _judge(port)


def test_scope_entry_yields_usable_port(
    factory: ProviderFactory, sessions: list[FakeSession]
) -> None:
    recorder = _Recorder(factory)
    with provider_scope(factory=recorder) as port:
        assert isinstance(_judge(port), SystemOneResponse)
    assert recorder.exits == [[None]]
    assert [session.closed for session in sessions] == [True]


def test_scope_exit_runs_once(
    factory: ProviderFactory, sessions: list[FakeSession]
) -> None:
    recorder = _Recorder(factory)
    for _ in range(2):
        with provider_scope(factory=recorder):
            pass
    first, second = recorder.contexts
    assert first is not second
    assert recorder.exits == [[None], [None]]
    assert [session.closed for session in sessions] == [True, True]


class _BodyError(Exception):
    pass


def test_scope_propagates_body_exception(factory: ProviderFactory) -> None:
    body = _BodyError("raised inside the provider scope")
    recorder = _Recorder(factory)
    with pytest.raises(_BodyError) as caught, provider_scope(factory=recorder):
        raise body
    assert caught.value is body
    assert recorder.exits == [[body]]
    direct = factory()
    direct.__enter__()
    assert not direct.__exit__(_BodyError, body, None)


def _evidence(media_type: str) -> ImageEvidence:
    image = ImageAttachment("scan", b"\x89PNG\r\n\x1a\nconformance", media_type)
    return ImageEvidence([image], {name: ["scan"] for name in QUESTIONS})


def test_media_refused_without_media_support(factory: ProviderFactory) -> None:
    with provider_scope(factory=factory) as port:
        assert not hasattr(port, "capabilities")
        assert not hasattr(port, "system_one_media")
        with pytest.raises(ProviderCapabilityError):
            judge_with_images(
                port, STATE, QUESTIONS, FAKE_MODEL, evidence=_evidence("image/png")
            )


def test_media_port_refuses_undeclared_media() -> None:
    scorer = ScriptedScoringFake(logprobs=LOGPROBS)
    judgment = judgment_port(media=True, scorer=scorer)
    port = TypevetMediaSystemOnePort(judgment, media=MediaCapabilities({"image/png"}))
    assert isinstance(port.capabilities(FAKE_MODEL), MediaCapabilities)
    with pytest.raises(ProviderCapabilityError):
        judge_with_images(
            port, STATE, QUESTIONS, FAKE_MODEL, evidence=_evidence("image/jpeg")
        )
    assert scorer.calls == []


def test_media_factory_yields_a_media_port(sessions: list[FakeSession]) -> None:
    media = MediaCapabilities({"image/png"})
    factory = provider_factory(lambda: open_fake_session(sessions), media=media)
    with provider_scope(factory=factory) as port:
        assert isinstance(port, TypevetMediaSystemOnePort)
        assert port.capabilities(FAKE_MODEL) == media
    assert [session.closed for session in sessions] == [True]
