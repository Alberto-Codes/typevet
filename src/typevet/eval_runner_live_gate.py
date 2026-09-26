"""Skip reasons for opt-in live eval when the router is unavailable (#98).

Examples:
    ```python
    from typevet.adapters.inbound.settings import load_llama_settings
    from typevet.eval_runner_live_gate import live_skip_reason

    reason = live_skip_reason(load_llama_settings())
    assert reason is None or isinstance(reason, str)
    ```

See Also:
    - [typevet.adapters.inbound.settings][]: ``LlamaSettings`` composition root
"""

from __future__ import annotations

import httpx

from typevet.adapters.inbound.settings import LlamaSettings

_HTTP_OK: int = 200


def _router_up(base_url: str) -> bool:
    try:
        response = httpx.get(f"{base_url.rstrip('/')}/v1/models", timeout=5.0)
    except httpx.HTTPError:
        return False
    return response.status_code == _HTTP_OK


def _model_listed(base_url: str, model: str) -> bool:
    try:
        response = httpx.get(f"{base_url.rstrip('/')}/v1/models", timeout=5.0)
        data = response.json()
    except (httpx.HTTPError, ValueError):
        return False
    ids = {item.get("id") for item in data.get("data", [])}
    return model in ids


def live_skip_reason(settings: LlamaSettings) -> str | None:
    """Return a skip message when live eval cannot run, else ``None``.

    Args:
        settings: Composition-root llama settings (env-backed).

    Returns:
        Human-readable skip reason, or ``None`` when the router looks ready.
    """
    model = settings.default_model
    if not model:
        return "TYPEVET_LLAMA__DEFAULT_MODEL (or TYPEVET_GEMMA_MODEL) not set"
    base = settings.base_url
    if not _router_up(base):
        return "llama.cpp router not reachable"
    if not _model_listed(base, model):
        return f"{model} not in router catalog"
    return None
