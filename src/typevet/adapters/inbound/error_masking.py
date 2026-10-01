"""Mask the configured vLLM key and header values in a raised error.

``backend_settings`` builds the patterns from the vLLM settings. This module
applies them to each argument and attribute of a ``GenerationError`` copy.
Each match becomes ``REDACTED``.

Attributes:
    Needles (type): Compiled patterns that match the key and header values.

Examples:
    ```python
    import re

    from typevet.adapters.inbound.error_masking import masked_error
    from typevet.domain.errors import GenerationError

    error = masked_error(GenerationError("key sk-1"), (re.compile("sk-1"),))
    assert "sk-1" not in str(error)
    ```

See Also:
    - [typevet.adapters.inbound.backend_settings][]: Builds the patterns.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

from typevet.adapters.diagnostics.redaction import REDACTED
from typevet.domain.errors import GenerationError

_PLAIN_CONTAINERS: tuple[type, ...] = (list, tuple, set, frozenset)
Needles = tuple[re.Pattern[str], ...]


def _masked_text(value: str | bytes, needles: Needles) -> str | bytes:
    """Replace each needle match in ``value`` with ``REDACTED``, keeping its type.

    Bytes stay bytes: they are decoded and encoded again as Latin-1, which
    keeps every byte. The key and header values are ASCII, so each needle
    matches the same bytes.

    Args:
        value: String or bytes to mask.
        needles: Patterns from the caller.

    Returns:
        The masked string or bytes.
    """
    text = value.decode("latin-1") if isinstance(value, bytes) else value
    for needle in needles:
        text = needle.sub(REDACTED, text)
    return text.encode("latin-1") if isinstance(value, bytes) else text


def _masked(value: Any, needles: Needles) -> Any:
    """Replace each key form in ``value`` with ``REDACTED``.

    Strings and bytes are masked by ``_masked_text`` and keep their type.
    Dicts, lists, tuples, sets and frozensets are copied as plain containers
    of the same kind with each key and item masked, so a parsed payload that
    holds the key loses it. Another ``Mapping``, such as the read-only
    ``rate_limit`` of a ``BackendHttpError``, becomes a read-only copy with
    each value masked and each key unchanged (#355). Other values are
    returned unchanged.

    Args:
        value: String, bytes, container or other attribute value.
        needles: Patterns from the caller.

    Returns:
        The masked value.
    """
    if isinstance(value, (str, bytes)):
        return _masked_text(value, needles)
    if isinstance(value, dict):
        return {_masked(k, needles): _masked(v, needles) for k, v in value.items()}
    if isinstance(value, Mapping):
        return MappingProxyType({k: _masked(v, needles) for k, v in value.items()})
    for kind in _PLAIN_CONTAINERS:
        if isinstance(value, kind):
            return kind(_masked(item, needles) for item in value)
    return value


def masked_error(exc: GenerationError, needles: Needles) -> GenerationError:
    """Rebuild ``exc`` as the same type with the key masked and no chain.

    ``BaseException.__new__`` makes the copy without calling ``__init__``, so
    every ``GenerationError`` subclass keeps its type and attributes.

    Args:
        exc: Error raised by the vLLM adapter.
        needles: Patterns from the caller; each match becomes ``REDACTED``.

    Returns:
        A new error whose arguments and attributes are masked, including
        strings and bytes inside dict, list, tuple, set and frozenset
        values such as a payload.
    """
    masked = type(exc).__new__(type(exc))
    masked.args = _masked(exc.args, needles)
    for name, value in vars(exc).items():
        setattr(masked, name, _masked(value, needles))
    return masked
