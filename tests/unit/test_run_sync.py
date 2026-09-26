"""Unit tests for the run_sync helper."""

from __future__ import annotations

from typing import Any, cast
from unittest.mock import patch

import pytest

from typevet import GenerationRequest
from typevet.adapters.inbound import run_sync
from typevet.adapters.outbound import AsyncFakeGenerationAdapter

SCHEMA = {
    "type": "object",
    "properties": {"n": {"type": "integer"}},
    "required": ["n"],
    "additionalProperties": False,
}


@pytest.mark.unit
def test_run_sync_runs_async_generation() -> None:
    port = AsyncFakeGenerationAdapter(value={"n": 7})
    result = run_sync(
        port.generate(GenerationRequest(prompt="n", schema=SCHEMA, model="fake"))
    )
    assert result.value == {"n": 7}


@pytest.mark.unit
def test_run_sync_propagates_exceptions() -> None:
    async def failing() -> None:
        msg = "boom"
        raise ValueError(msg)

    with pytest.raises(ValueError, match="boom"):
        run_sync(failing())


@pytest.mark.unit
def test_run_sync_rejects_non_coroutine() -> None:
    with pytest.raises(TypeError, match="coroutine"):
        run_sync(cast(Any, 42))


@pytest.mark.unit
def test_run_sync_rejects_unbound_async_method() -> None:
    port = AsyncFakeGenerationAdapter(value={"n": 1})

    with pytest.raises(TypeError, match="coroutine"):
        run_sync(cast(Any, port.generate))


@pytest.mark.unit
def test_run_sync_running_loop_raises_clear_error() -> None:
    async def dummy() -> str:
        return "ok"

    coro = dummy()
    try:
        with (
            patch(
                "asyncio.run",
                side_effect=RuntimeError(
                    "asyncio.run() cannot be called from a running event loop"
                ),
            ),
            pytest.raises(RuntimeError, match="event loop is already active"),
        ):
            run_sync(coro)
    finally:
        coro.close()
