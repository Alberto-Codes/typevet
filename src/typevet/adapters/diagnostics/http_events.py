"""Emit one terminal HTTP diagnostic per logical request when logging is configured.

Unconfigured library calls stay silent. This module never opens a network sink;
it only writes through the configured stderr renderer.

Examples:
    ```python
    from typevet.adapters.diagnostics.http_events import http_request_event

    with http_request_event(model="fake", path="v1/chat/completions") as event:
        event.status_code = 200
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
class HttpRequestEvent:
    """Terminal HTTP metadata without prompts, schemas or response bodies.

    Attributes:
        status_code (int | None): HTTP status when the adapter knows it.
        outcome (str): ``success`` or ``error`` after the request completes.

    Examples:
        ```python
        from typevet.adapters.diagnostics.http_events import HttpRequestEvent

        event = HttpRequestEvent(status_code=200, outcome="success")
        assert event.status_code == 200
        ```
    """

    status_code: int | None = None
    outcome: str = "error"


@contextmanager
def http_request_event(
    *,
    model: str,
    method: str = "POST",
    path: str = "v1/chat/completions",
) -> Iterator[HttpRequestEvent]:
    """Yield mutable terminal metadata and emit ``http.request`` when configured.

    Args:
        model: Router model alias for the call.
        method: HTTP method (for example ``POST``).
        path: Path relative to the adapter base URL.

    Yields:
        Event state the adapter updates before the block exits.

    Raises:
        BaseException: Re-raises any exception from the wrapped block after
            recording ``error_type`` on the terminal event.
    """
    event = HttpRequestEvent()
    failure: BaseException | None = None
    try:
        yield event
    except BaseException as exc:
        failure = exc
        raise
    finally:
        if structlog.is_configured():
            structlog.get_logger().debug(
                "http.request",
                method=method,
                path=path,
                model=diagnostic_model(model),
                status_code=event.status_code,
                outcome=event.outcome,
                error_type=None if failure is None else type(failure).__name__,
                run_id=current_run_id(),
            )
