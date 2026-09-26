"""Configure stderr structlog diagnostics at the composition root.

Logs are diagnostics, not telemetry: no automatic network export. Library
imports stay silent until ``configure`` or ``configure_from_environ`` runs.

Examples:
    ```python
    from typevet.adapters.diagnostics.logs import (
        bind_run_id,
        configure_from_environ,
        new_run_id,
    )

    configure_from_environ()
    bind_run_id(new_run_id())
    ```

See Also:
    - [typevet.adapters.diagnostics.http_events][]: HTTP terminal events
    - [typevet.adapters.diagnostics.generation_events][]: Generation events
"""

from __future__ import annotations

import logging
import sys
import uuid
from typing import Any

import structlog
from structlog.tracebacks import ExceptionDictTransformer

from typevet.adapters.diagnostics.fields import filter_event_fields, set_run_id_context
from typevet.adapters.diagnostics.redaction import make_redact_processor
from typevet.adapters.diagnostics.settings import LogSettings, load_log_settings

LEVELS = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}
_DICT_TRACEBACKS = structlog.processors.ExceptionRenderer(
    ExceptionDictTransformer(show_locals=False)
)


def wants_json(settings: LogSettings, stream: Any) -> bool:
    """Return true when lines should render as JSON objects.

    Returns:
        ``True`` when JSON rendering is selected explicitly or inferred from
        a non-TTY stream under ``auto`` format.
    """
    if settings.format != "auto":
        return settings.format == "json"
    return not (hasattr(stream, "isatty") and stream.isatty())


def configure(settings: LogSettings, stream: Any = None) -> None:
    """Send structured diagnostics to stderr with redaction and field filters.

    Args:
        settings: Format, level and prompt logging policy.
        stream: Output stream; defaults to ``sys.stderr``.
    """
    target = sys.stderr if stream is None else stream
    redact = make_redact_processor(log_prompts=settings.log_prompts)
    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        filter_event_fields,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
    ]
    if wants_json(settings, target):
        processors += [_DICT_TRACEBACKS, redact, structlog.processors.JSONRenderer()]
    else:
        processors += [
            redact,
            structlog.dev.ConsoleRenderer(
                exception_formatter=structlog.dev.plain_traceback
            ),
        ]
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(LEVELS[settings.level]),
        logger_factory=structlog.PrintLoggerFactory(target),
        cache_logger_on_first_use=False,
    )


def configure_from_environ(stream: Any = None) -> LogSettings:
    """Load ``TYPEVET_LOG__*`` settings and configure stderr diagnostics once.

    Returns:
        The settings that were applied.
    """
    settings = load_log_settings()
    configure(settings, stream=stream)
    return settings


def new_run_id() -> str:
    """Return a short hex id that joins lines for one invocation.

    Returns:
        Twelve lowercase hex characters from a random UUID.
    """
    return uuid.uuid4().hex[:12]


def bind_run_id(run_id: str) -> None:
    """Bind ``run_id`` on every line until the context is cleared or replaced."""
    structlog.contextvars.bind_contextvars(run_id=run_id)
    set_run_id_context(run_id)
