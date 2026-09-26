"""Environment-backed llama.cpp settings for the composition root.

Outbound adapters take explicit constructor arguments. Only a composition root
(CLI, MCP server, or live test harness) reads ``TYPEVET_LLAMA__*`` variables
and passes the values into [typevet.adapters.outbound.llama_cpp][].

Diagnostic log settings live in [typevet.adapters.diagnostics.settings][].

Examples:
    ```python
    from typevet.adapters.inbound.settings import (
        llama_cpp_adapter,
        load_llama_settings,
    )

    settings = load_llama_settings()
    with llama_cpp_adapter(settings) as port:
        pass  # port.generate(...)
    ```

See Also:
    - [typevet.adapters.outbound.llama_cpp][]: Sync llama.cpp adapter
    - [typevet.adapters.diagnostics.settings][]: ``TYPEVET_LOG__*`` settings
    - docs/reference/configuration.md: Environment variable reference
    - docs/how-to/run-a-multimodal-live-smoke.md: ``multimodal_model`` consumer
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

from typevet.adapters.outbound.llama_cpp import LlamaCppGenerationAdapter

_DEFAULT_BASE_URL = "http://127.0.0.1:8090"
_DEFAULT_TIMEOUT = 300.0
_DEFAULT_MULTIMODAL_MODEL = "gemma-3-4b-it-q4km-mm"


@dataclass(frozen=True, slots=True)
class LlamaSettings:
    """Connection options for a local llama.cpp OpenAI-compat router.

    Attributes:
        base_url (str): Router root without a trailing slash.
        timeout (float): HTTP request timeout in seconds.
        default_model (str | None): Default model id when a call omits one.
        multimodal_model (str): Model id the opt-in image smoke asks for.

    Examples:
        ```python
        from typevet.adapters.inbound.settings import LlamaSettings

        LlamaSettings(base_url="http://127.0.0.1:8090", timeout=300.0)
        ```
    """

    base_url: str = _DEFAULT_BASE_URL
    timeout: float = _DEFAULT_TIMEOUT
    default_model: str | None = None
    multimodal_model: str = _DEFAULT_MULTIMODAL_MODEL


def _read_base_url(source: Mapping[str, str]) -> str:
    nested = source.get("TYPEVET_LLAMA__BASE_URL", "").strip()
    if nested:
        return nested.rstrip("/")
    legacy = source.get("TYPEVET_LLAMA_URL", "").strip()
    if legacy:
        return legacy.rstrip("/")
    return _DEFAULT_BASE_URL


def _read_default_model(source: Mapping[str, str]) -> str | None:
    nested = source.get("TYPEVET_LLAMA__DEFAULT_MODEL", "").strip()
    if nested:
        return nested
    legacy = source.get("TYPEVET_GEMMA_MODEL", "").strip()
    return legacy or None


def _read_multimodal_model(source: Mapping[str, str]) -> str:
    nested = source.get("TYPEVET_LLAMA__MULTIMODAL_MODEL", "").strip()
    return nested or _DEFAULT_MULTIMODAL_MODEL


def _read_timeout(source: Mapping[str, str]) -> float:
    raw = source.get("TYPEVET_LLAMA__TIMEOUT", "").strip()
    if not raw:
        return _DEFAULT_TIMEOUT
    try:
        timeout = float(raw)
    except ValueError as exc:
        msg = "TYPEVET_LLAMA__TIMEOUT must be a positive number of seconds"
        raise ValueError(msg) from exc
    if timeout <= 0:
        msg = "TYPEVET_LLAMA__TIMEOUT must be a positive number of seconds"
        raise ValueError(msg)
    return timeout


def load_llama_settings(
    environ: Mapping[str, str] | None = None,
) -> LlamaSettings:
    """Read ``TYPEVET_LLAMA__*`` variables for the composition root.

    Legacy single-segment names ``TYPEVET_LLAMA_URL`` and ``TYPEVET_GEMMA_MODEL``
    remain supported when the nested names are unset.
    ``TYPEVET_LLAMA__MULTIMODAL_MODEL`` names the model id for the opt-in image
    smoke and has no legacy alias.

    Args:
        environ: Mapping to read. Defaults to ``os.environ``.

    Returns:
        Frozen settings with defaults for missing keys.

    Raises:
        ValueError: When ``TYPEVET_LLAMA__TIMEOUT`` is not a positive number.
    """
    source = os.environ if environ is None else environ
    return LlamaSettings(
        base_url=_read_base_url(source),
        timeout=_read_timeout(source),
        default_model=_read_default_model(source),
        multimodal_model=_read_multimodal_model(source),
    )


def llama_cpp_adapter(settings: LlamaSettings) -> LlamaCppGenerationAdapter:
    """Build a sync llama.cpp adapter from composition-root settings.

    Args:
        settings: Values read from the environment or constructed in tests.

    Returns:
        An adapter that does not read ``os.environ`` itself.

    Examples:
        ```python
        from typevet.adapters.inbound.settings import (
            llama_cpp_adapter,
            load_llama_settings,
        )

        with llama_cpp_adapter(load_llama_settings()) as port:
            pass
        ```
    """
    return LlamaCppGenerationAdapter(
        base_url=settings.base_url,
        timeout=settings.timeout,
    )
