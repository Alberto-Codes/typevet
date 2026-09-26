"""Structured stderr diagnostics for generation and HTTP adapters.

Diagnostics are operator signals, not telemetry: no automatic network export.
Importing this package does not configure structlog.

Examples:
    ```python
    from typevet.adapters.diagnostics import (
        bind_run_id,
        configure_from_environ,
        generation_call_event,
        http_request_event,
        new_run_id,
    )
    ```

See Also:
    - [typevet.adapters.diagnostics.logs][]: Composition-root configuration
    - [typevet.adapters.diagnostics.redaction][]: Secret and prompt redaction

Attributes:
    REDACTED (str): Placeholder written over redacted secret values.
    SECRET_KEYS (frozenset[str]): Field names always masked in diagnostics.
    LogSettings (type): Frozen log format, level and prompt policy.
    bind_run_id (function): Bind ``run_id`` on every diagnostic line.
    configure (function): Apply ``LogSettings`` to stderr structlog.
    configure_from_environ (function): Load env settings and configure once.
    diagnostic_model (function): Keep safe model aliases in event fields.
    generation_call_event (function): Terminal ``generation.call`` context manager.
    http_request_event (function): Terminal ``http.request`` context manager.
    load_log_settings (function): Read ``TYPEVET_LOG__*`` variables.
    new_run_id (function): Create a short hex invocation id.
"""

from typevet.adapters.diagnostics.fields import diagnostic_model
from typevet.adapters.diagnostics.generation_events import generation_call_event
from typevet.adapters.diagnostics.http_events import http_request_event
from typevet.adapters.diagnostics.logs import (
    bind_run_id,
    configure,
    configure_from_environ,
    new_run_id,
)
from typevet.adapters.diagnostics.redaction import REDACTED, SECRET_KEYS
from typevet.adapters.diagnostics.settings import LogSettings, load_log_settings

__all__ = [
    "REDACTED",
    "SECRET_KEYS",
    "LogSettings",
    "bind_run_id",
    "configure",
    "configure_from_environ",
    "diagnostic_model",
    "generation_call_event",
    "http_request_event",
    "load_log_settings",
    "new_run_id",
]
