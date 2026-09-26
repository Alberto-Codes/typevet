"""Filter built-in diagnostic events and bound model identifiers.

Built-in events form a closed set until the reference table lands (#40).
Importing this module does not configure logging.

Examples:
    ```python
    from typevet.adapters.diagnostics.fields import diagnostic_model

    assert diagnostic_model("gemma-4-test") == "gemma-4-test"
    ```

See Also:
    - [typevet.adapters.diagnostics.logs][]: Wires ``filter_event_fields`` into structlog
    - [typevet.adapters.diagnostics.generation_events][]: ``generation.call`` field set
    - [typevet.adapters.diagnostics.http_events][]: ``http.request`` field set
"""

from __future__ import annotations

import re
from contextvars import ContextVar, Token
from typing import Any

_RUN_ID: ContextVar[str | None] = ContextVar("typevet_run_id", default=None)
_MODEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,63}$")
_EVENT_FIELDS = {
    "generation.call": frozenset(
        {
            "event",
            "model",
            "outcome",
            "error_type",
            "run_id",
        }
    ),
    "http.request": frozenset(
        {
            "event",
            "method",
            "path",
            "model",
            "status_code",
            "outcome",
            "error_type",
            "run_id",
        }
    ),
}


def current_run_id() -> str | None:
    """Return the bound invocation ``run_id``, if any.

    Returns:
        The current ``run_id``, or ``None`` when nothing is bound.
    """
    return _RUN_ID.get()


def set_run_id_context(run_id: str | None) -> Token[str | None]:
    """Store ``run_id`` in a context variable; returns the reset token.

    Args:
        run_id: Identifier to attach to subsequent diagnostic lines.

    Returns:
        Token passed to ``reset_run_id_context`` to restore the prior value.
    """
    return _RUN_ID.set(run_id)


def reset_run_id_context(token: Token[str | None]) -> None:
    """Restore the previous ``run_id`` binding."""
    _RUN_ID.reset(token)


def diagnostic_model(model: str | None) -> str | None:
    """Keep bounded model aliases in diagnostics without changing API values.

    Args:
        model: Requested or resolved router model alias.

    Returns:
        The identifier when it matches the safe pattern, else ``None``.
    """
    if model is not None and _MODEL.fullmatch(model):
        return model
    return None


def filter_event_fields(
    _logger: Any, _method: str, event_dict: dict[str, Any]
) -> dict[str, Any]:
    """Drop unknown keys from built-in events; keep application events intact.

    Prompts, schemas, headers and response bodies never appear in the
    closed field sets for ``generation.call`` and ``http.request``.

    Returns:
        The event dict unchanged for unknown events, or filtered to the
        closed field set for built-in events.
    """
    name = event_dict.get("event")
    fields = _EVENT_FIELDS.get(name) if isinstance(name, str) else None
    if fields is None:
        return event_dict
    return {key: value for key, value in event_dict.items() if key in fields}
