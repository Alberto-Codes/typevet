"""judgevet provider conformance kit over the typevet judgevet bridge (#290).

judgevet 0.17.0 publishes ``judgevet.testing.conformance``. Each class below
subclasses a kit base class for one bridge port over offline typevet fakes:
the sync text port, the media port and the async port. The module tests that
follow check typevet behaviour the kit does not cover: each provider scope
closes its typevet session, and a media factory yields a media port.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager

import pytest
from judgevet.domain.media import MediaCapabilities
from judgevet.providers import ProviderFactory, provider_scope
from judgevet.testing.conformance import (
    BaseAsyncProviderConformance,
    BaseProviderConformance,
)

from tests.fixtures.judgevet_bridge import (
    FakeSession,
    judgment_port,
    open_fake_session,
)
from typevet.adapters.inbound.judgevet import (
    AsyncTypevetSystemOnePort,
    TypevetMediaSystemOnePort,
    TypevetSystemOnePort,
    provider_factory,
)
from typevet.domain.errors import ScoringError

pytestmark = pytest.mark.contract

MEDIA = MediaCapabilities({"image/png"})
"""The media declaration the media port conformance class uses."""

AsyncFactory = Callable[[], AbstractAsyncContextManager[AsyncTypevetSystemOnePort]]
"""A zero-argument callable whose async context yields the async bridge port."""


class TestTypevetTextProviderConformance(BaseProviderConformance):
    """The kit rules for the sync text port, ``TypevetSystemOnePort``."""

    @pytest.fixture
    def provider_factory(self) -> ProviderFactory:
        return provider_factory(open_fake_session)

    @pytest.fixture
    def failing_port(self) -> TypevetSystemOnePort:
        return TypevetSystemOnePort(judgment_port(fail=ScoringError("offline")))


class TestTypevetMediaProviderConformance(BaseProviderConformance):
    """The kit rules for the media port, ``TypevetMediaSystemOnePort``."""

    @pytest.fixture
    def provider_factory(self) -> ProviderFactory:
        return provider_factory(lambda: open_fake_session(media=True), media=MEDIA)

    @pytest.fixture
    def failing_port(self) -> TypevetMediaSystemOnePort:
        failing = judgment_port(fail=ScoringError("offline"), media=True)
        return TypevetMediaSystemOnePort(failing, media=MEDIA)


class TestAsyncTypevetProviderConformance(BaseAsyncProviderConformance):
    """The kit rules for the async port, ``AsyncTypevetSystemOnePort``."""

    @pytest.fixture
    def provider_factory(self) -> AsyncFactory:
        @asynccontextmanager
        async def scope() -> AsyncIterator[AsyncTypevetSystemOnePort]:
            with open_fake_session() as session:
                yield AsyncTypevetSystemOnePort(TypevetSystemOnePort(session.port))

        return scope

    @pytest.fixture
    def failing_port(self) -> AsyncTypevetSystemOnePort:
        failing = judgment_port(fail=ScoringError("offline"))
        return AsyncTypevetSystemOnePort(TypevetSystemOnePort(failing))


@pytest.mark.parametrize("media", [None, MEDIA], ids=["text", "media"])
def test_each_scope_closes_its_typevet_session(
    media: MediaCapabilities | None,
) -> None:
    sessions: list[FakeSession] = []
    factory = provider_factory(
        lambda: open_fake_session(sessions, media=media is not None), media=media
    )
    for _ in range(2):
        with provider_scope(factory=factory):
            assert [session.closed for session in sessions][-1] is False
    assert [session.closed for session in sessions] == [True, True]


def test_media_factory_yields_a_media_port() -> None:
    factory = provider_factory(lambda: open_fake_session(media=True), media=MEDIA)
    with provider_scope(factory=factory) as port:
        assert isinstance(port, TypevetMediaSystemOnePort)
        assert port.capabilities("any-model") == MEDIA
