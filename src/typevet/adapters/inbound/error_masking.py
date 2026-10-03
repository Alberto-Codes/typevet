"""Mask the configured key and header values in a raised error.

``key_needles`` builds the patterns from the key and extra headers of the
vLLM or llama.cpp settings (#410). ``masked_error`` applies them to each
argument and attribute of a ``GenerationError`` copy. Each match becomes
``REDACTED``. ``KeyMaskingJudgmentPort`` and ``enter_masked`` raise such a
copy, with no cause or context, from a judgment port and from opening a
judgment session.

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
    - [typevet.adapters.inbound.backend_settings][]: vLLM clients and adapters
    - [typevet.adapters.inbound.settings][]: llama.cpp clients and adapters
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from contextlib import AbstractContextManager, ExitStack
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from typevet.adapters.diagnostics.redaction import REDACTED
from typevet.domain.errors import GenerationError

if TYPE_CHECKING:
    from typevet.domain.judgment_questions import Question
    from typevet.domain.judgment_response import JudgmentResponse
    from typevet.domain.media import ImageInput
    from typevet.ports.judgment import JudgmentPort

_PLAIN_CONTAINERS: tuple[type, ...] = (list, tuple, set, frozenset)
_BOUND = "(?<![A-Za-z0-9]){}(?![A-Za-z0-9])"
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
        exc: Error raised by a vLLM or llama.cpp adapter.
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


def key_needles(api_key: str | None, headers: Mapping[str, str]) -> Needles:
    """Return the masking patterns for the key and each extra header value.

    The raw and JSON-escaped key match anywhere. A raw or JSON-escaped header
    value matches only as a whole token, between characters that are not
    ASCII letters or digits, so a value such as ``1`` leaves ``HTTP 401``
    readable (#348). Longer forms come first.

    Args:
        api_key: Configured key, or ``None``.
        headers: Extra header names and literal values.

    Returns:
        Compiled patterns, or an empty tuple without a key or header value.
    """
    key = api_key
    forms = {(k, False) for k in (key, json.dumps(key)[1:-1])} if key else set()
    for value in filter(None, headers.values()):
        forms |= {(value, True), (json.dumps(value)[1:-1], True)}
    ordered = sorted(forms, key=lambda form: len(form[0]), reverse=True)
    return tuple(
        re.compile(_BOUND.format(re.escape(text)) if whole else re.escape(text))
        for text, whole in ordered
    )


def masked_if_keyed(exc: GenerationError, needles: Needles) -> GenerationError | None:
    """Return a masked copy of ``exc`` when a key or header value is configured.

    The copy is made whether or not the key text appears in ``exc``. The
    cause chain can hold the key where no text check sees it, for example in
    the headers of an httpx request, so the copy drops the chain in all cases.

    Args:
        exc: Error raised by a vLLM or llama.cpp adapter.
        needles: Patterns from ``key_needles``, or empty without a key.

    Returns:
        The masked copy, or ``None`` when ``needles`` is empty.
    """
    if not needles:
        return None
    return masked_error(exc, needles)


def enter_masked[T](
    stack: ExitStack, manager: AbstractContextManager[T], needles: Needles
) -> T:
    """Enter ``manager`` on ``stack`` and mask a ``GenerationError`` it raises.

    The masked copy is raised outside the ``except`` block, so it has no
    cause or context. Errors raised later, inside the ``with`` body, are
    not changed here.

    Args:
        stack: Exit stack that closes ``manager``.
        manager: Context manager to enter, for example a judgment session.
        needles: Patterns from ``key_needles``, or empty without a key.

    Returns:
        The value that ``manager`` gives on entry.

    Raises:
        GenerationError: The entry error, as a masked copy when ``needles``
            is not empty.
    """
    try:
        return stack.enter_context(manager)
    except GenerationError as exc:
        masked = masked_if_keyed(exc, needles)
        if masked is None:
            raise
    raise masked


class KeyMaskingJudgmentPort:
    """``JudgmentPort`` wrapper that masks the configured key in errors.

    Attributes:
        _port (JudgmentPort): Wrapped judgment port.
        _needles (Needles): Patterns for the key and header values, or empty.

    Examples:
        ```python
        KeyMaskingJudgmentPort(session.port, key_needles(key, headers))
        ```
    """

    def __init__(self, port: JudgmentPort, needles: Needles) -> None:
        """Wrap ``port``.

        Args:
            port: Judgment port whose errors are masked.
            needles: Patterns from ``key_needles``, or empty without a key.
        """
        self._port = port
        self._needles = needles

    def judge(
        self,
        state: str | dict[str, Any] | list[Any],
        questions: Mapping[str, Question | Mapping[str, Any]],
        model: str,
        *,
        media: tuple[ImageInput, ...] | None = None,
        off_option_threshold: float | None = None,
    ) -> JudgmentResponse:
        """Judge, and mask the configured key in any raised error.

        Args:
            state: Content under evaluation.
            questions: Question names to typed or raw questions.
            model: Served model name.
            media: Images to condition every scored field on.
            off_option_threshold: Forwarded to the wrapped port.

        Returns:
            The response from the wrapped port, unchanged.

        Raises:
            GenerationError: The port error, as a masked copy when a key is set.
        """
        try:
            return self._port.judge(
                state,
                questions,
                model,
                media=media,
                off_option_threshold=off_option_threshold,
            )
        except GenerationError as exc:
            masked = masked_if_keyed(exc, self._needles)
            if masked is None:
                raise
        raise masked
