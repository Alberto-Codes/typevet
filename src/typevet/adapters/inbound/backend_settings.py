"""Backend selection and vLLM settings for the composition root.

``TYPEVET_BACKEND`` selects ``llama_cpp`` (the default) or ``vllm``. The vLLM
branch reads ``TYPEVET_VLLM__*`` variables and builds one ``httpx.Client``
that carries the base URL, timeout and optional bearer key.
``TYPEVET_VLLM__MAX_CONCURRENCY`` sets ``VllmSettings.max_concurrency``, the
POST limit for an ``AsyncVllmGenerationAdapter``. Outbound adapters never read
the environment.

The configured key never reaches the caller in clear text. ``VllmSettings``
leaves it out of ``repr``. The adapter that ``generation_adapter`` returns
catches each ``GenerationError`` from ``generate``, and the port that
``open_judgment`` yields catches each one from ``judge``. When the error text
or its cause chain holds the raw or JSON-escaped key, the error is raised
again as the same type with the key replaced by ``***`` and with no cause or
context. A server that echoes the ``Authorization`` header therefore cannot
put the key into a ``BackendHttpError``. Successful results are not changed.
The HTTP client is built the same way with or without a key, so environment
proxy settings apply in both cases. Error messages name variables, never
values.

Examples:
    ```python
    from typevet.adapters.inbound.backend_settings import (
        generation_adapter,
        open_judgment,
    )

    with generation_adapter() as port:
        pass  # port.generate(...)
    with open_judgment() as session:
        pass  # session.port.judge(...)
    ```

See Also:
    - [typevet.adapters.inbound.settings][]: ``TYPEVET_LLAMA__*`` settings
    - [typevet.adapters.outbound.vllm_generation][]: vLLM generation adapter
    - [typevet.adapters.outbound.vllm_judgment_factory][]: vLLM judgment factory
    - [typevet.adapters.diagnostics.redaction][]: ``REDACTED`` (``***``) mask
    - docs/reference/configuration.md: Environment variable reference
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal

import httpx

from typevet.adapters.diagnostics.redaction import REDACTED
from typevet.adapters.inbound.settings import llama_cpp_adapter, load_llama_settings
from typevet.adapters.outbound.gemma_native_vision_factory import (
    GemmaNativeVisionSession,
    open_gemma_native_vision_judgment,
)
from typevet.adapters.outbound.llama_cpp import LlamaCppGenerationAdapter
from typevet.adapters.outbound.vllm_generation import VllmGenerationAdapter
from typevet.adapters.outbound.vllm_judgment_factory import (
    VllmJudgmentSession,
    open_vllm_judgment,
)
from typevet.domain.errors import GenerationError
from typevet.domain.models import GenerationRequest, GenerationResult

if TYPE_CHECKING:
    from typevet.domain.judgment_questions import Question
    from typevet.domain.judgment_response import JudgmentResponse
    from typevet.domain.media import ImageInput
    from typevet.ports.judgment import JudgmentPort

Backend = Literal["llama_cpp", "vllm"]

_DEFAULT_TIMEOUT = 300.0
_BACKENDS: tuple[Backend, ...] = ("llama_cpp", "vllm")
MASK = REDACTED


@dataclass(frozen=True, slots=True)
class VllmSettings:
    """Connection options for a vLLM OpenAI-compatible server.

    Attributes:
        base_url (str): Server root without a trailing slash.
        model (str): Served model name.
        timeout (float): HTTP request timeout in seconds.
        api_key (str | None): Bearer key, or ``None``. Excluded from ``repr``.
        max_concurrency (int): Maximum POSTs in flight for one async adapter.

    Examples:
        ```python
        from typevet.adapters.inbound.backend_settings import VllmSettings

        VllmSettings(base_url="http://127.0.0.1:8000", model="served-model")
        ```
    """

    base_url: str
    model: str
    timeout: float = _DEFAULT_TIMEOUT
    api_key: str | None = field(default=None, repr=False)
    max_concurrency: int = 1


def _masked(text: str, needles: tuple[str, ...]) -> str:
    for needle in needles:
        text = text.replace(needle, MASK)
    return text


def _chain_text(exc: BaseException) -> str:
    parts: list[str] = []
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        parts += [str(current), repr(current)]
        parts += [v for v in vars(current).values() if isinstance(v, str)]
        current = current.__cause__ or current.__context__
    return "\n".join(parts)


def _masked_error(exc: GenerationError, needles: tuple[str, ...]) -> GenerationError:
    """Rebuild ``exc`` as the same type with the key masked and no chain.

    ``BaseException.__new__`` makes the copy without calling ``__init__``, so
    every ``GenerationError`` subclass keeps its type and attributes.

    Args:
        exc: Error raised by the vLLM adapter.
        needles: Key forms to replace with ``MASK``.

    Returns:
        A new error whose arguments and string attributes are masked.
    """
    masked = type(exc).__new__(type(exc))
    masked.args = tuple(
        _masked(arg, needles) if isinstance(arg, str) else arg for arg in exc.args
    )
    for name, value in vars(exc).items():
        setattr(
            masked, name, _masked(value, needles) if isinstance(value, str) else value
        )
    return masked


def _key_needles(key: str | None) -> tuple[str, ...]:
    if key is None:
        return ()
    return tuple(sorted({key, json.dumps(key)[1:-1]}, key=len, reverse=True))


def _masked_if_keyed(
    exc: GenerationError, needles: tuple[str, ...]
) -> GenerationError | None:
    """Return a masked copy of ``exc`` when its chain holds a key form.

    Args:
        exc: Error raised by a vLLM adapter.
        needles: Key forms from ``_key_needles``.

    Returns:
        The masked copy, or ``None`` when no key form is present.
    """
    chain = _chain_text(exc)
    if not any(needle in chain for needle in needles):
        return None
    return _masked_error(exc, needles)


class _KeyMaskingJudgmentPort:
    """``JudgmentPort`` wrapper that masks the configured key in errors.

    Attributes:
        _port (JudgmentPort): Wrapped vLLM judgment port.
        _needles (tuple[str, ...]): Raw and JSON-escaped key, or empty.

    Examples:
        ```python
        _KeyMaskingJudgmentPort(session.port, _key_needles("sk-..."))
        ```
    """

    def __init__(self, port: JudgmentPort, needles: tuple[str, ...]) -> None:
        self._port = port
        self._needles = needles

    def judge(
        self,
        state: str | dict[str, Any] | list[Any],
        questions: Mapping[str, Question | Mapping[str, Any]],
        model: str,
        *,
        media: tuple[ImageInput, ...] | None = None,
    ) -> JudgmentResponse:
        """Judge, and mask the configured key in any raised error.

        Args:
            state: Content under evaluation.
            questions: Question names to typed or raw questions.
            model: Served model name.
            media: Images to condition every scored field on.

        Returns:
            The response from the wrapped port, unchanged.

        Raises:
            GenerationError: The port error, masked when it holds the key.
        """
        try:
            return self._port.judge(state, questions, model, media=media)
        except GenerationError as exc:
            masked = _masked_if_keyed(exc, self._needles)
            if masked is None:
                raise
        raise masked


class _ClientOwningVllmAdapter(VllmGenerationAdapter):
    """vLLM adapter that owns its client and masks the key in errors.

    Attributes:
        _settings_client (httpx.Client): Client from ``vllm_http_client``.
        _needles (tuple[str, ...]): Raw and JSON-escaped key, or empty.

    Examples:
        ```python
        settings = VllmSettings(base_url="http://127.0.0.1:8000", model="m")
        _ClientOwningVllmAdapter(settings, vllm_http_client(settings))
        ```
    """

    def __init__(self, settings: VllmSettings, client: httpx.Client) -> None:
        super().__init__(settings.base_url, timeout=settings.timeout, client=client)
        self._settings_client = client
        self._needles = _key_needles(settings.api_key)

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate, and mask the configured key in any raised error.

        A successful result is returned unchanged. ``_masked_if_keyed``
        checks each error: one whose text or cause chain holds the key is
        raised again as the same type with the key replaced by ``MASK`` and
        with no cause or context.

        Args:
            request: Prompt, schema and served model name.

        Returns:
            The result from ``VllmGenerationAdapter.generate``.

        Raises:
            GenerationError: The adapter error, masked when it holds the key.
        """
        try:
            return super().generate(request)
        except GenerationError as exc:
            masked = _masked_if_keyed(exc, self._needles)
            if masked is None:
                raise
        raise masked

    def close(self) -> None:
        """Close the composition-root client."""
        super().close()
        self._settings_client.close()


def _read_timeout(source: Mapping[str, str], name: str) -> float:
    raw = source.get(name, "").strip()
    if not raw:
        return _DEFAULT_TIMEOUT
    msg = f"{name} must be a positive number of seconds"
    try:
        timeout = float(raw)
    except ValueError as exc:
        raise ValueError(msg) from exc
    if timeout <= 0:
        raise ValueError(msg)
    return timeout


def _read_max_concurrency(source: Mapping[str, str], name: str) -> int:
    raw = source.get(name, "").strip()
    if not raw:
        return 1
    msg = f"{name} must be a positive integer"
    try:
        limit = int(raw)
    except ValueError:
        raise ValueError(msg) from None
    if limit < 1:
        raise ValueError(msg)
    return limit


def _read_required(source: Mapping[str, str], name: str) -> str:
    value = source.get(name, "").strip()
    if not value:
        msg = f"{name} is required when TYPEVET_BACKEND is vllm"
        raise ValueError(msg)
    return value


def load_backend(environ: Mapping[str, str] | None = None) -> Backend:
    """Read ``TYPEVET_BACKEND``.

    Args:
        environ: Mapping to read. Defaults to ``os.environ``.

    Returns:
        ``"llama_cpp"`` when unset or empty, else the named backend.

    Raises:
        ValueError: When the value is not ``llama_cpp`` or ``vllm``.
    """
    source = os.environ if environ is None else environ
    raw = source.get("TYPEVET_BACKEND", "").strip()
    if not raw:
        return "llama_cpp"
    for backend in _BACKENDS:
        if raw == backend:
            return backend
    msg = "TYPEVET_BACKEND must be llama_cpp or vllm"
    raise ValueError(msg)


def load_vllm_settings(environ: Mapping[str, str] | None = None) -> VllmSettings:
    """Read ``TYPEVET_VLLM__*`` variables.

    Args:
        environ: Mapping to read. Defaults to ``os.environ``.

    Returns:
        Frozen settings. An empty ``TYPEVET_VLLM__API_KEY`` gives ``None``.

    Raises:
        ValueError: When ``TYPEVET_VLLM__BASE_URL`` or ``TYPEVET_VLLM__MODEL``
            is missing, ``TYPEVET_VLLM__TIMEOUT`` is not a positive number,
            ``TYPEVET_VLLM__MAX_CONCURRENCY`` is not a positive integer, or
            ``TYPEVET_VLLM__API_KEY`` holds a non-ASCII character.
    """
    source = os.environ if environ is None else environ
    api_key = source.get("TYPEVET_VLLM__API_KEY", "").strip()
    if not api_key.isascii():
        msg = "TYPEVET_VLLM__API_KEY must contain only ASCII characters"
        raise ValueError(msg)
    return VllmSettings(
        base_url=_read_required(source, "TYPEVET_VLLM__BASE_URL").rstrip("/"),
        model=_read_required(source, "TYPEVET_VLLM__MODEL"),
        timeout=_read_timeout(source, "TYPEVET_VLLM__TIMEOUT"),
        api_key=api_key or None,
        max_concurrency=_read_max_concurrency(source, "TYPEVET_VLLM__MAX_CONCURRENCY"),
    )


def vllm_http_client(
    settings: VllmSettings,
    *,
    transport: httpx.BaseTransport | None = None,
) -> httpx.Client:
    """Build the shared vLLM HTTP client.

    Args:
        settings: vLLM connection settings.
        transport: Optional transport, for example ``httpx.MockTransport``.

    Returns:
        A client with ``base_url`` and ``timeout`` set. When a key is set, the
        client also sends ``Authorization: Bearer <key>``. Nothing else
        changes, so environment proxy settings apply with or without a key.
    """
    headers: dict[str, str] = {}
    if settings.api_key is not None:
        headers["Authorization"] = f"Bearer {settings.api_key}"
    return httpx.Client(
        base_url=settings.base_url,
        timeout=settings.timeout,
        headers=headers,
        transport=transport,
    )


def generation_adapter(
    environ: Mapping[str, str] | None = None,
    *,
    transport: httpx.BaseTransport | None = None,
) -> LlamaCppGenerationAdapter | VllmGenerationAdapter:
    """Build the generation adapter that ``TYPEVET_BACKEND`` selects.

    Args:
        environ: Mapping to read. Defaults to ``os.environ``.
        transport: Optional transport for the vLLM client. The llama.cpp
            branch ignores it.

    Returns:
        A llama.cpp adapter from ``load_llama_settings``, or a vLLM adapter on
        the ``vllm_http_client``. Closing the vLLM adapter closes its client,
        and its errors never show the configured key.

    Raises:
        ValueError: When a backend or vLLM variable is invalid.
    """
    source = os.environ if environ is None else environ
    if load_backend(source) == "llama_cpp":
        return llama_cpp_adapter(load_llama_settings(source))
    settings = load_vllm_settings(source)
    client = vllm_http_client(settings, transport=transport)
    return _ClientOwningVllmAdapter(settings, client)


@contextmanager
def open_judgment(
    environ: Mapping[str, str] | None = None,
    *,
    transport: httpx.BaseTransport | None = None,
) -> Iterator[GemmaNativeVisionSession | VllmJudgmentSession]:
    """Open the judgment session that ``TYPEVET_BACKEND`` selects.

    Args:
        environ: Mapping to read. Defaults to ``os.environ``.
        transport: Optional transport for the vLLM client. The llama.cpp
            branch ignores it.

    Yields:
        A llama.cpp session from ``open_gemma_native_vision_judgment`` with
        ``load_llama_settings``, or a vLLM session from ``open_vllm_judgment``
        on the ``vllm_http_client``. The vLLM client closes on exit, and
        judgment errors never show the configured key.

    Raises:
        ValueError: When a backend or vLLM variable is invalid.
    """
    source = os.environ if environ is None else environ
    if load_backend(source) == "llama_cpp":
        with open_gemma_native_vision_judgment(
            settings=load_llama_settings(source)
        ) as llama_session:
            yield llama_session
        return
    settings = load_vllm_settings(source)
    with (
        vllm_http_client(settings, transport=transport) as client,
        open_vllm_judgment(
            client=client, model=settings.model, base_url=settings.base_url
        ) as session,
    ):
        masked = _KeyMaskingJudgmentPort(session.port, _key_needles(settings.api_key))
        yield replace(session, port=masked)
