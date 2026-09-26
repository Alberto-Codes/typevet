"""Environment-backed log settings for the composition root.

Diagnostics are operator signals on stderr, not telemetry and not remote export.
Importing this module does not configure structlog.

Examples:
    ```python
    from typevet.adapters.diagnostics.settings import load_log_settings

    settings = load_log_settings({"TYPEVET_LOG__LEVEL": "debug"})
    assert settings.level == "debug"
    ```

See Also:
    - [typevet.adapters.diagnostics.logs][]: Applies these settings to structlog
    - [typevet.adapters.diagnostics.redaction][]: Honors ``log_prompts`` when masking
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

_VALID_FORMATS = frozenset({"auto", "json", "console"})
_VALID_LEVELS = frozenset({"debug", "info", "warning", "error", "critical"})
_TRUTHY = frozenset({"1", "true", "yes", "on"})


@dataclass(frozen=True, slots=True)
class LogSettings:
    """How diagnostic lines are rendered and which severities are kept.

    Attributes:
        format (str): ``auto``, ``json`` or ``console``. Default ``auto``.
        level (str): ``debug`` through ``critical``. Default ``info``.
        log_prompts (bool): When false, prompt-like fields are stripped from output.

    Examples:
        ```python
        from typevet.adapters.diagnostics.settings import LogSettings

        LogSettings(format="json", level="debug", log_prompts=True)
        ```
    """

    format: str = "auto"
    level: str = "info"
    log_prompts: bool = False


def load_log_settings(
    environ: Mapping[str, str] | None = None,
) -> LogSettings:
    """Read ``TYPEVET_LOG__*`` variables for the composition root.

    Args:
        environ: Mapping to read. Defaults to ``os.environ``.

    Returns:
        Frozen settings with defaults for missing keys.

    Raises:
        ValueError: When format or level is not an allowed value.
    """
    source = os.environ if environ is None else environ
    fmt = source.get("TYPEVET_LOG__FORMAT", "auto").strip().lower()
    level = source.get("TYPEVET_LOG__LEVEL", "info").strip().lower()
    log_prompts_raw = source.get("TYPEVET_LOG__LOG_PROMPTS", "").strip().lower()
    if fmt not in _VALID_FORMATS:
        msg = f"TYPEVET_LOG__FORMAT must be one of {sorted(_VALID_FORMATS)}"
        raise ValueError(msg)
    if level not in _VALID_LEVELS:
        msg = f"TYPEVET_LOG__LEVEL must be one of {sorted(_VALID_LEVELS)}"
        raise ValueError(msg)
    log_prompts = log_prompts_raw in _TRUTHY
    return LogSettings(format=fmt, level=level, log_prompts=log_prompts)
