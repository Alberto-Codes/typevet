"""Unit tests for thread-safe lazy HTTP client creation (#340).

A scoring adapter built with ``client=None`` creates its ``httpx.Client`` on
first use. Several threads that race through ``_ensure_client`` must share one
client; no extra client may leak unclosed.
"""

from __future__ import annotations

import threading
import time
from typing import Any, ClassVar

import httpx
import pytest

from typevet.adapters.outbound.llama_cpp.scoring import (
    LlamaCppCandidateScoringAdapter,
)
from typevet.adapters.outbound.vllm.scoring import VllmCandidateScoringAdapter

_THREADS = 8
_PAUSE_SECONDS = 0.02


class _CountingClient:
    """Stand-in for ``httpx.Client`` that counts constructions.

    Attributes:
        constructions (int): Number of instances built so far.
    """

    constructions = 0
    _count_lock = threading.Lock()

    def __init__(self, **_kwargs: Any) -> None:
        """Pause to widen the race window, then count the construction.

        Args:
            **_kwargs: Ignored ``httpx.Client`` keyword arguments.
        """
        time.sleep(_PAUSE_SECONDS)
        with _CountingClient._count_lock:
            _CountingClient.constructions += 1


def _race(adapter: Any) -> list[object]:
    """Release ``_THREADS`` threads together into ``_ensure_client``.

    Args:
        adapter: A scoring adapter built with ``client=None``.

    Returns:
        The client each thread received.
    """
    barrier = threading.Barrier(_THREADS)
    results: list[object] = []
    results_lock = threading.Lock()

    def worker() -> None:
        barrier.wait()
        client = adapter._ensure_client()
        with results_lock:
            results.append(client)

    threads = [threading.Thread(target=worker) for _ in range(_THREADS)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return results


@pytest.fixture
def counting_client(monkeypatch: pytest.MonkeyPatch) -> type[_CountingClient]:
    """Replace ``httpx.Client`` with the counting stand-in.

    Both scoring modules call ``httpx.Client`` through ``import httpx``.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        The counting stand-in class, with its count reset.
    """
    monkeypatch.setattr(_CountingClient, "constructions", 0)
    monkeypatch.setattr(httpx, "Client", _CountingClient)
    return _CountingClient


@pytest.mark.unit
def test_vllm_scoring_lazy_client_is_created_once_across_threads(
    counting_client: type[_CountingClient],
) -> None:
    adapter = VllmCandidateScoringAdapter("http://127.0.0.1:1")
    results = _race(adapter)
    assert len(results) == _THREADS
    assert counting_client.constructions == 1
    assert len({id(client) for client in results}) == 1
    assert results[0] is adapter._client


@pytest.mark.unit
def test_llama_cpp_scoring_lazy_client_is_created_once_across_threads(
    counting_client: type[_CountingClient],
) -> None:
    adapter = LlamaCppCandidateScoringAdapter("http://127.0.0.1:1")
    results = _race(adapter)
    assert len(results) == _THREADS
    assert counting_client.constructions == 1
    assert len({id(client) for client in results}) == 1
    assert results[0] is adapter._client


_ADAPTERS = pytest.mark.parametrize(
    "build",
    [VllmCandidateScoringAdapter, LlamaCppCandidateScoringAdapter],
    ids=["vllm", "llama_cpp"],
)
_ROUNDS = 25
_STEP_SECONDS = 0.001


class _TrackedClient:
    """Stand-in for ``httpx.Client`` that records each build and close (#345).

    ``close`` marks the client closed first and then pauses, so a racing
    thread can see a closed client that the adapter still holds.

    Attributes:
        built (list[_TrackedClient]): Every instance built so far.
        closed (bool): Whether ``close`` was called.
    """

    built: ClassVar[list[_TrackedClient]] = []
    _built_lock = threading.Lock()

    def __init__(self, **_kwargs: Any) -> None:
        """Pause to widen the race window, then record the build.

        Args:
            **_kwargs: Ignored ``httpx.Client`` keyword arguments.
        """
        time.sleep(_STEP_SECONDS)
        self.closed = False
        with _TrackedClient._built_lock:
            _TrackedClient.built.append(self)

    def close(self) -> None:
        """Mark the client closed, then pause to widen the race window."""
        self.closed = True
        time.sleep(_STEP_SECONDS)


@pytest.fixture
def tracked_client(monkeypatch: pytest.MonkeyPatch) -> type[_TrackedClient]:
    """Replace ``httpx.Client`` with the tracking stand-in.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        The tracking stand-in class, with an empty build list.
    """
    monkeypatch.setattr(_TrackedClient, "built", [])
    monkeypatch.setattr(httpx, "Client", _TrackedClient)
    return _TrackedClient


def _race_close(adapter: Any) -> list[BaseException]:
    """Race ``_THREADS`` threads: half call ``close``, half ``_ensure_client``.

    Args:
        adapter: A scoring adapter.

    Returns:
        The ``AttributeError`` or ``AssertionError`` each thread raised, for
        example ``close`` on a client that another thread set to ``None``.
    """
    barrier = threading.Barrier(_THREADS)
    errors: list[BaseException] = []
    errors_lock = threading.Lock()

    def worker(index: int) -> None:
        barrier.wait()
        try:
            for _ in range(_ROUNDS):
                if index % 2:
                    adapter.close()
                else:
                    adapter._ensure_client()
        except (AttributeError, AssertionError) as exc:
            with errors_lock:
                errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(_THREADS)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return errors


@pytest.mark.unit
@_ADAPTERS
def test_close_racing_ensure_client_leaks_no_owned_client(
    build: Any, tracked_client: type[_TrackedClient]
) -> None:
    adapter = build("http://127.0.0.1:1")
    errors = _race_close(adapter)
    assert errors == []
    held = adapter._client
    orphans = [c for c in tracked_client.built if not c.closed and c is not held]
    assert orphans == []
    adapter.close()
    assert adapter._client is None
    assert [c for c in tracked_client.built if not c.closed] == []
    before = len(tracked_client.built)
    results = _race(adapter)
    assert len(tracked_client.built) - before == 1
    assert len({id(client) for client in results}) == 1
    assert isinstance(results[0], _TrackedClient)
    assert not results[0].closed


@pytest.mark.unit
@_ADAPTERS
def test_ensure_client_during_close_never_returns_the_closing_client(
    build: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    closing = threading.Event()
    release = threading.Event()

    class _SlowCloseClient(_TrackedClient):
        """Tracked client whose ``close`` waits until the test releases it."""

        def close(self) -> None:
            """Mark closed, signal the test, then wait for the release."""
            self.closed = True
            closing.set()
            release.wait(timeout=5)

    monkeypatch.setattr(_TrackedClient, "built", [])
    monkeypatch.setattr(httpx, "Client", _SlowCloseClient)
    adapter = build("http://127.0.0.1:1")
    first = adapter._ensure_client()
    closer = threading.Thread(target=adapter.close)
    closer.start()
    assert closing.wait(timeout=5)
    got: list[object] = []
    user = threading.Thread(target=lambda: got.append(adapter._ensure_client()))
    user.start()
    user.join(timeout=0.2)
    release.set()
    closer.join()
    user.join()
    assert first.closed
    assert got[0] is not first
    assert isinstance(got[0], _TrackedClient)
    assert not got[0].closed


class _NeverBuiltClient:
    """``httpx.Client`` stand-in that fails when the adapter builds one."""

    def __init__(self, **_kwargs: Any) -> None:
        """Refuse construction.

        Args:
            **_kwargs: Ignored ``httpx.Client`` keyword arguments.

        Raises:
            AssertionError: Always; an injected client must be used instead.
        """
        msg = "adapter built an httpx.Client despite an injected client"
        raise AssertionError(msg)


@pytest.mark.unit
@_ADAPTERS
def test_injected_client_is_returned_to_every_thread_and_left_open(
    build: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(httpx, "Client", _NeverBuiltClient)
    injected = _TrackedClient.__new__(_TrackedClient)
    injected.closed = False
    adapter = build("http://127.0.0.1:1", client=injected)
    assert _race_close(adapter) == []
    results = _race(adapter)
    adapter.close()
    assert len(results) == _THREADS
    assert all(client is injected for client in results)
    assert adapter._client is injected
    assert not injected.closed
