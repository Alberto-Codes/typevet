"""Sync bridge for async generation callers (scripts and CLIs).

Examples:
    ```python
    from typevet import GenerationRequest
    from typevet.adapters.inbound import run_sync
    from typevet.adapters.outbound import AsyncFakeGenerationAdapter

    schema = {
        "type": "object",
        "properties": {"ok": {"type": "boolean"}},
        "required": ["ok"],
        "additionalProperties": False,
    }
    port = AsyncFakeGenerationAdapter(value={"ok": True})
    result = run_sync(
        port.generate(GenerationRequest(prompt="Say ok.", schema=schema, model="fake"))
    )
    assert result.value["ok"] is True
    ```

See Also:
    - [typevet.ports.async_generation][]: AsyncGenerationPort
    - [typevet.adapters.inbound.api][]: Sync ``generate`` helper
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any


def run_sync[T](coro: Coroutine[Any, Any, T]) -> T:
    """Run an async coroutine synchronously and return its result.

    Use this from plain scripts instead of adding ``*_sync`` methods on ports.
    Pass a coroutine object (for example ``port.generate(request)``), not the
    unbound method.

    Args:
        coro: Coroutine to execute (typically from ``AsyncGenerationPort.generate``).

    Returns:
        The coroutine result (for example a ``GenerationResult``).

    Raises:
        TypeError: When ``coro`` is not a coroutine object.
        RuntimeError: When called from a thread that already has a running loop.
    """
    if not asyncio.iscoroutine(coro):
        msg = (
            f"run_sync() requires a coroutine object, got {type(coro).__name__}. "
            "Call the async method first: run_sync(port.generate(request))"
        )
        raise TypeError(msg)

    try:
        return asyncio.run(coro)
    except RuntimeError as exc:
        if "asyncio.run() cannot be called from a running event loop" in str(exc):
            msg = (
                "run_sync() cannot run while an event loop is already active. "
                "Await port.generate(...) in async code instead."
            )
            raise RuntimeError(msg) from exc
        raise
