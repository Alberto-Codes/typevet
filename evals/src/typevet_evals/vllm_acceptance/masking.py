"""Receipt masking helpers for the #170 vLLM acceptance run.

``typevet_evals.vllm_acceptance.core`` uses these helpers so that the receipt
and its error message never hold the configured key or an extra header value
(#348). A header value is masked only as a whole token, between characters
that are not ASCII letters or digits. Only the receipt parts that a gateway
can echo are masked, and masked keys stay distinct (#351). The key itself is
replaced in its raw and JSON-escaped forms.

Examples:
    ```python
    from typevet_evals.vllm_acceptance.masking import _masked, _masked_echoes

    masked = _masked_echoes(receipt, settings)
    text = _masked(json.dumps(masked), settings)
    ```

See Also:
    - [typevet_evals.vllm_acceptance.core][]: run and receipt writer
    - [typevet.adapters.diagnostics.redaction][]: ``REDACTED`` (``***``) mask
    - [typevet.adapters.inbound.backend_settings][]: ``VllmSettings``
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Final

from typevet.adapters.diagnostics.redaction import REDACTED

if TYPE_CHECKING:
    from typevet.adapters.inbound.backend_settings import VllmSettings

_BOUND: Final = "(?<![A-Za-z0-9]){}(?![A-Za-z0-9])"
_ECHOES: Final = (("pins", "version"), ("pins", "served_models"), ("error", "message"))


def _key_needle(settings: VllmSettings, *, escaped: bool) -> str:
    """Return the configured key, raw or JSON-escaped, without binding it.

    Returns:
        The key text to replace with ``REDACTED``.
    """
    return json.dumps(settings.api_key)[1:-1] if escaped else str(settings.api_key)


def _masked_headers(value: Any, settings: VllmSettings | None) -> Any:
    """Mask each extra header value as a whole token in the strings of ``value``.

    A gateway can echo a request header into a body or an error that the
    receipt records (#348). A value matches only between characters that are
    not ASCII letters or digits, as in the adapter errors, so a value such as
    ``1`` leaves ``HTTP 401`` readable. Strings and the keys of nested dicts
    are masked. The keys of the outer dict, and numbers, keep their form (#351).

    Args:
        value: String, or a dict or list of JSON values.
        settings: vLLM settings with the extra headers, or ``None``.

    Returns:
        ``value`` with each header value replaced by ``REDACTED``.
    """
    headers = {} if settings is None else settings.headers
    values = sorted(filter(None, headers.values()), key=len, reverse=True)
    patterns = [re.compile(_BOUND.format(re.escape(v))) for v in values]
    return _masked_tokens(value, patterns) if patterns else value


def _masked_tokens(
    value: Any, patterns: list[re.Pattern[str]], *, keys: bool = False
) -> Any:
    if isinstance(value, str):
        for pattern in patterns:
            value = pattern.sub(REDACTED, value)
        return value
    if isinstance(value, Mapping):
        names = _masked_keys(list(value), patterns) if keys else list(value)
        items = zip(names, value.values(), strict=True)
        return {n: _masked_tokens(v, patterns, keys=True) for n, v in items}
    if isinstance(value, list | tuple):
        return [_masked_tokens(item, patterns, keys=keys) for item in value]
    return value


def _masked_keys(names: list[Any], patterns: list[re.Pattern[str]]) -> list[Any]:
    """Mask string keys as whole tokens, and keep them distinct (#351).

    A key that the masking changes gets the first free name of
    ``***``, ``***_2``, ``***_3`` and so on, in key order. A key that the
    masking does not change keeps its name. Thus no key collapses.

    Returns:
        The masked keys, in the same order as ``names``.
    """
    masked = [_masked_tokens(n, patterns) if isinstance(n, str) else n for n in names]
    taken = {new for old, new in zip(names, masked, strict=True) if old == new}
    out = []
    for old, new in zip(names, masked, strict=True):
        name, count = new, 1
        while old != new and name in taken:
            count += 1
            name = f"{new}_{count}"
        taken.add(name)
        out.append(name)
    return out


def _masked_echoes(
    receipt: Mapping[str, Any], settings: VllmSettings | None
) -> dict[str, Any]:
    """Mask header values only in the receipt parts that a gateway can echo.

    Those parts are ``pins.version`` and ``pins.served_models``, read from the
    server, and ``error.message``. The keys and the values that typevet sets,
    such as ``stopped``, ``passed`` and ``pins.configured_model``, stay as
    they are, so a header value equal to one of them cannot change the
    receipt schema (#348). The ``status`` and ``body`` keys of
    ``pins.version`` and the ``id`` and ``root`` keys of each served model
    are typevet keys too. Keys below them come from the server and are
    masked (#351).

    Args:
        receipt: Mapping from ``run_acceptance``.
        settings: vLLM settings with the extra headers, or ``None``.

    Returns:
        A copy of ``receipt`` with those parts masked.
    """
    masked = dict(receipt)
    for parent, child in _ECHOES:
        section = masked.get(parent)
        if isinstance(section, Mapping) and child in section:
            echo = _masked_headers(section[child], settings)
            masked[parent] = {**section, child: echo}
    return masked


def _masked(text: str, settings: VllmSettings | None) -> str:
    if settings is None or not settings.api_key:
        return text
    for escaped in (False, True):
        text = text.replace(_key_needle(settings, escaped=escaped), REDACTED)
    return text
