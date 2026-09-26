"""Redact secrets and optional prompt content before stderr rendering.

Examples:
    ```python
    from typevet.adapters.diagnostics.redaction import REDACTED, redact

    assert redact(None, "info", {"api_key": "k"})["api_key"] == REDACTED
    ```

See Also:
    - [typevet.adapters.diagnostics.logs][]: Installs the redact processor on configure
    - [typevet.adapters.diagnostics.settings][]: ``log_prompts`` policy for prompt keys
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

REDACTED = "***"
SECRET_KEYS = frozenset(
    {
        "api_key",
        "authorization",
        "password",
        "private_key",
        "private_key_pem",
        "token",
    }
)
PROMPT_KEYS = frozenset(
    {
        "content",
        "messages",
        "prompt",
        "raw_text",
        "system",
        "user",
    }
)
_SCALARS = (bool, int, float)
_EXC_INFO = "exc_info"


def _is_pem(value: str) -> bool:
    return "-----BEGIN" in value


def _redact_pair(key: Any, value: Any, *, log_prompts: bool) -> Any:
    if isinstance(key, str):
        lowered = key.lower()
        if lowered in {k.lower() for k in SECRET_KEYS}:
            return REDACTED
        if not log_prompts and lowered in {k.lower() for k in PROMPT_KEYS}:
            return REDACTED
    return _mask(value, log_prompts=log_prompts)


def _mask(value: Any, *, log_prompts: bool) -> Any:
    if isinstance(value, str):
        return REDACTED if _is_pem(value) else value
    if value is None or isinstance(value, _SCALARS):
        return value
    if isinstance(value, dict):
        return {
            k: _redact_pair(k, v, log_prompts=log_prompts) for k, v in value.items()
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_mask(item, log_prompts=log_prompts) for item in value]
    return type(value).__name__


def make_redact_processor(
    *, log_prompts: bool
) -> Callable[[Any, str, dict[str, Any]], dict[str, Any]]:
    """Build a structlog processor that masks secrets and optional prompts.

    Args:
        log_prompts: When false, prompt-like field names and long strings
            are replaced with ``REDACTED``.

    Returns:
        A structlog-compatible processor callable.
    """

    def _redact_event(
        _logger: Any, _method: str, event_dict: dict[str, Any]
    ) -> dict[str, Any]:
        return {
            k: v if k == _EXC_INFO else _redact_pair(k, v, log_prompts=log_prompts)
            for k, v in event_dict.items()
        }

    return _redact_event
