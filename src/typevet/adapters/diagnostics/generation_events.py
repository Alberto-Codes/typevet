"""Emit one terminal generation diagnostic per logical call when configured.

Examples:
    ```python
    from typevet.adapters.diagnostics.generation_events import generation_call_event

    with generation_call_event(model="fake") as event:
        event.outcome = "success"
    ```

See Also:
    - [typevet.adapters.diagnostics.fields][]: ``run_id`` and model field helpers
    - [typevet.adapters.diagnostics.logs][]: Configures structlog before events emit
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import structlog

from typevet.adapters.diagnostics.fields import current_run_id, diagnostic_model


@dataclass
class GenerationCallEvent:
    """Terminal generation metadata without prompt or schema payloads.

    Attributes:
        outcome (str): ``success`` or ``error`` after the adapter finishes.

    Examples:
        ```python
        from typevet.adapters.diagnostics.generation_events import GenerationCallEvent

        event = GenerationCallEvent(outcome="success")
        assert event.outcome == "success"
        ```
    """

    outcome: str = "error"


@contextmanager
def generation_call_event(*, model: str) -> Iterator[GenerationCallEvent]:
    """Yield terminal metadata and emit ``generation.call`` when configured.

    Args:
        model: Router model alias for the call.

    Yields:
        Event state the adapter updates before the block exits.

    Raises:
        BaseException: Re-raises any exception from the wrapped block after
            recording ``error_type`` on the terminal event.
    """
    event = GenerationCallEvent()
    failure: BaseException | None = None
    try:
        yield event
    except BaseException as exc:
        failure = exc
        raise
    finally:
        if structlog.is_configured():
            structlog.get_logger().debug(
                "generation.call",
                model=diagnostic_model(model),
                outcome=event.outcome,
                error_type=None if failure is None else type(failure).__name__,
                run_id=current_run_id(),
            )
