"""Skip reasons for opt-in live eval when the router is unavailable (#98/#127).

Examples:
    ```python
    from typevet.adapters.inbound.settings import load_llama_settings
    from typevet.evaluation.runner.live_gate import live_skip_reason

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
_CATALOG_INVALID = "llama.cpp router catalog invalid"
_CATALOG_EMPTY = "llama.cpp router catalog empty"


def _catalog_model_ids(payload: object) -> tuple[frozenset[str] | None, str | None]:
    """Parse an OpenAI-style ``/v1/models`` JSON body.

    Returns:
        ``(ids, problem)`` where ``problem`` is ``None``, ``"empty"``, or
        ``"invalid"``. When ``problem`` is ``"invalid"``, ``ids`` is ``None``.
    """
    if not isinstance(payload, dict):
        return None, "invalid"
    data = payload.get("data", [])
    if data is None or not isinstance(data, list):
        return None, "invalid"
    ids: set[str] = set()
    for item in data:
        if not isinstance(item, dict):
            return None, "invalid"
        model_id = item.get("id")
        if not isinstance(model_id, str):
            return None, "invalid"
        ids.add(model_id)
    if not ids:
        return frozenset(), "empty"
    return frozenset(ids), None


def _fetch_models_payload(base_url: str) -> tuple[object | None, str | None, bool]:
    """GET ``/v1/models`` once.

    Returns:
        ``(json_body, skip_reason, json_decode_failed)``. When
        ``json_decode_failed`` is true, callers should treat the catalog as
        unreadable but keep the historical "model not in catalog" skip text.
    """
    try:
        response = httpx.get(f"{base_url.rstrip('/')}/v1/models", timeout=5.0)
    except httpx.HTTPError:
        return None, "llama.cpp router not reachable", False
    if response.status_code != _HTTP_OK:
        return None, "llama.cpp router not reachable", False
    try:
        return response.json(), None, False
    except ValueError:
        return None, None, True


def _router_catalog_skip_reason(model: str, base_url: str) -> str | None:
    payload, fetch_reason, json_failed = _fetch_models_payload(base_url)
    if fetch_reason is not None:
        return fetch_reason
    if json_failed:
        return f"{model} not in router catalog"
    ids, catalog_problem = _catalog_model_ids(payload)
    if catalog_problem == "invalid" or ids is None:
        return _CATALOG_INVALID
    if catalog_problem == "empty":
        return _CATALOG_EMPTY
    return None if model in ids else f"{model} not in router catalog"


def live_skip_reason(settings: LlamaSettings) -> str | None:
    """Return a skip message when live eval cannot run, else ``None``.

    Args:
        settings: Composition-root llama settings (env-backed).

    Returns:
        Human-readable skip reason (unreachable router, invalid/empty catalog,
        or missing model id), or ``None`` when the router looks ready.
    """
    model = settings.default_model
    if not model:
        return "TYPEVET_LLAMA__DEFAULT_MODEL (or TYPEVET_GEMMA_MODEL) not set"
    return _router_catalog_skip_reason(model, settings.base_url)
