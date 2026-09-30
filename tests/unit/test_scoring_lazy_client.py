"""Unit tests for thread-safe lazy HTTP client creation (#340).

A scoring adapter built with ``client=None`` creates its ``httpx.Client`` on
first use. Several threads that race through ``_ensure_client`` must share one
client; no extra client may leak unclosed.
"""

from __future__ import annotations

import threading
import time
from typing import Any

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
