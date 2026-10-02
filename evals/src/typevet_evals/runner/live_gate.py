"""Skip reasons for opt-in live eval when the router is unavailable (#98/#127).

``live_backend`` reads ``TYPEVET_BACKEND`` for a live test and rejects ``fake``
before any HTTP call (#407).

Examples:
    ```python
    from typevet.adapters.inbound.settings import load_llama_settings
    from typevet_evals.runner.live_gate import live_gate_action, live_skip_reason

    reason = live_skip_reason(load_llama_settings())
    assert live_gate_action(reason).name in {"RUN", "SKIP", "FAIL"}
    ```

See Also:
    - [typevet.adapters.inbound.settings][]: ``LlamaSettings`` composition root
"""

from __future__ import annotations

import os
from enum import Enum
from typing import TYPE_CHECKING, Literal

import httpx

from typevet.adapters.inbound.backend_settings import load_backend
from typevet.adapters.inbound.settings import LlamaSettings

if TYPE_CHECKING:
    from collections.abc import Mapping

TYPEVET_REQUIRE_LIVE_ENV = "TYPEVET_REQUIRE_LIVE"
_TRUTHY = frozenset({"1", "true", "yes", "on"})
_HTTP_OK: int = 200


class LiveGateAction(Enum):
    """Outcome when the router or model is not ready for live work.

    Examples:
        ```python
        from typevet_evals.runner.live_gate import LiveGateAction, live_gate_action

        assert live_gate_action("router down") is LiveGateAction.SKIP
        ```
    """

    RUN = "run"
    SKIP = "skip"
    FAIL = "fail"


def require_live_enabled() -> bool:
    """Return whether live collection must fail instead of skip (#191).

    Returns:
        ``True`` when ``TYPEVET_REQUIRE_LIVE`` is truthy (``1``, ``true``, ``yes``,
        ``on``).
    """
    raw = os.environ.get(TYPEVET_REQUIRE_LIVE_ENV, "")
    return raw.strip().lower() in _TRUTHY


def live_gate_action(skip_reason: str | None) -> LiveGateAction:
    """Map ``live_skip_reason`` output to skip, fail, or run.

    When ``TYPEVET_REQUIRE_LIVE`` is truthy and ``skip_reason`` is set, return
    ``FAIL`` so pytest callers can ``pytest.fail``. Default remains ``SKIP``.

    Returns:
        ``RUN`` when ``skip_reason`` is ``None``; otherwise ``SKIP`` or ``FAIL``.
    """
    if skip_reason is None:
        return LiveGateAction.RUN
    if require_live_enabled():
        return LiveGateAction.FAIL
    return LiveGateAction.SKIP


def live_backend(environ: Mapping[str, str]) -> Literal["llama_cpp", "vllm"]:
    """Read ``TYPEVET_BACKEND`` for a live test and reject ``fake`` (#407).

    A live test needs a served model. The ``fake`` backend has none, so this
    raises before the caller opens any session or makes any HTTP call.

    Args:
        environ: Mapping to read ``TYPEVET_BACKEND`` from.

    Returns:
        ``"llama_cpp"`` when unset or empty, else ``"llama_cpp"`` or ``"vllm"``.

    Raises:
        ValueError: When the value is ``fake``, or when ``load_backend``
            rejects it.

    Examples:
        ```python
        from typevet_evals.runner.live_gate import live_backend

        assert live_backend({"TYPEVET_BACKEND": "vllm"}) == "vllm"
        ```
    """
    backend = load_backend(environ)
    if backend == "fake":
        msg = "TYPEVET_BACKEND must be llama_cpp or vllm for a live test, not fake"
        raise ValueError(msg)
    return backend


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
