"""llama.cpp model vocabulary size read from ``GET /v1/models`` (#321).

Measured on router build ``b11277-eae11d221`` (2026-09-30): ``/props?model=<id>``
does not report the vocabulary size. ``/v1/models`` reports it as
``meta.n_vocab`` on the entry of each **loaded** model (``262144`` for
``gemma-4-31b-24gib-kv11-decoder``). An unloaded router model has no ``meta``.
A single-model server lists one entry, whose ``id`` is its alias or model path.

Examples:
    ```python
    import httpx

    from typevet.adapters.outbound.llama_cpp.vocabulary import fetch_model_n_vocab

    with httpx.Client() as client:
        fetch_model_n_vocab(client, "http://127.0.0.1:8090/", "local")
    ```

See Also:
    - [typevet.adapters.outbound.llama_cpp.scoring][]: Caller of this module
    - [typevet.adapters.outbound.llama_cpp.http_mapping][]: One-retry send
      and HTTP error mapping that the read uses
"""

from __future__ import annotations

from typing import Any
from urllib.parse import urljoin

import httpx

from typevet.adapters.outbound.llama_cpp.http_mapping import (
    ensure_success_status,
    parse_json_response,
    send_idempotent,
)


def _model_entry(payload: Any, model: str) -> dict[str, Any] | None:
    """Return the ``/v1/models`` entry that serves ``model``, else ``None``.

    An entry matches when its ``id`` or one of its ``aliases`` equals
    ``model``. When no entry matches and the list holds exactly one entry,
    that entry is the model: a single-model server answers every model name
    with its one model.

    Args:
        payload: Parsed JSON body from ``GET /v1/models``.
        model: Model id the scoring request sent.

    Returns:
        The matching entry object, or ``None`` when no entry is found.
    """
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return None
    entries = [entry for entry in data if isinstance(entry, dict)]
    for entry in entries:
        aliases = entry.get("aliases")
        if entry.get("id") == model or (isinstance(aliases, list) and model in aliases):
            return entry
    if len(data) == 1 and len(entries) == 1:
        return entries[0]
    return None


def n_vocab_from_models(payload: Any, model: str) -> int | None:
    """Read ``meta.n_vocab`` of the entry for ``model`` from a models body.

    Args:
        payload: Parsed JSON body from ``GET /v1/models``.
        model: Model id the scoring request sent.

    Returns:
        The positive vocabulary size, or ``None`` when the body does not
        report one for that model.
    """
    entry = _model_entry(payload, model)
    meta = entry.get("meta") if entry is not None else None
    size = meta.get("n_vocab") if isinstance(meta, dict) else None
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        return None
    return size


def fetch_model_n_vocab(client: httpx.Client, base_url: str, model: str) -> int | None:
    """Read the vocabulary size of ``model`` from ``GET /v1/models``.

    An early close on a new or reused connection gets one retry (#305).

    Args:
        client: Open HTTP client for the server.
        base_url: Server root with a trailing slash.
        model: Model id the scoring request sent.

    Returns:
        The positive vocabulary size, or ``None`` when the server does not
        report one for that model.

    Raises:
        TransportError: When the HTTP client fails before a response.
        BackendHttpError: When the server returns HTTP status 400 or above.
        GenerationError: When the body is not valid JSON.
    """
    url = urljoin(base_url, "v1/models")
    response = send_idempotent(lambda: client.get(url))
    ensure_success_status(response)
    return n_vocab_from_models(parse_json_response(response), model)
