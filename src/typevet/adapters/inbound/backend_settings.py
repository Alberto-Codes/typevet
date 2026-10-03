"""Backend selection and vLLM settings for the composition root.

``TYPEVET_BACKEND`` selects ``llama_cpp`` (the default), ``vllm`` or ``fake``.
The ``fake`` backend is offline and serves judgment only: ``open_judgment``
yields the ``FakeJudgmentSession`` from
``typevet.adapters.inbound.fake_backend``, and ``generation_adapter`` refuses
it. The vLLM branch reads ``TYPEVET_VLLM__*`` variables and builds one ``httpx.Client``
that carries the base URL, timeout, optional key and gateway headers.
``TYPEVET_VLLM__MAX_CONCURRENCY`` sets ``VllmSettings.max_concurrency``, the
POST limit for the ``AsyncVllmGenerationAdapter`` that
``async_vllm_generation_adapter`` builds on an ``httpx.AsyncClient`` with the
same settings. The optional
``TYPEVET_VLLM__USER_AGENT`` sets the ``User-Agent`` header; when it is unset
the client sends the httpx default. Outbound adapters never read the
environment.

A server behind an API gateway (#331) needs more. ``TYPEVET_VLLM__AUTH_HEADER``
and ``TYPEVET_VLLM__AUTH_SCHEME`` set the header and scheme for the key,
``TYPEVET_VLLM__HEADERS`` adds literal headers from a JSON object, and
``TYPEVET_VLLM__REQUEST_ID_HEADER`` sends a fresh UUID4 hex value on each
request. ``typevet.adapters.inbound.http_headers`` checks each rule when
``VllmSettings`` is built, and ``typevet.adapters.inbound.gateway_headers``
gives the event hooks. The clients follow no redirect: a redirect raises
``BackendHttpError`` without the ``Location`` header, and an HTML error body
is withheld from the error.

The configured key never reaches the caller in clear text. ``VllmSettings``
leaves it out of ``repr``. The adapters that ``generation_adapter`` and
``async_vllm_generation_adapter`` return catch each ``GenerationError`` from
``generate``, and the port that ``open_judgment`` yields catches each one from
``judge``. When a key is configured, each such error is raised again as a copy
of the same type with no cause or context. In the copy, the raw or JSON-escaped
key in the text or an attribute, such as a parsed payload, is replaced by
``***``. Strings and bytes inside dicts, lists, tuples, sets and frozensets are
masked too, and masked bytes stay bytes. The copy is made even when the key
text is absent, because an httpx error in the cause chain holds the request
headers. A server that echoes the ``Authorization`` header therefore cannot put
the key into a ``BackendHttpError``. Each value in ``TYPEVET_VLLM__HEADERS`` is
masked too, with or without a key, but only as a whole token between
characters that are not ASCII letters or digits, so a short value such as
``1`` leaves ``HTTP 401`` readable (#348). ``VllmSettings.headers`` is a
read-only copy. Without a key or extra headers,
errors pass through unchanged. Successful results are not changed. The HTTP
clients are built the same way with or without a key, so environment proxy
settings apply in both cases. ``TYPEVET_VLLM__TIMEOUT`` must be finite and positive, so ``nan`` and
``inf`` fail. Error messages name variables, never values, and a parse failure
raises with no cause or context.

Examples:
    ```python
    from typevet.adapters.inbound.backend_settings import (
        async_vllm_generation_adapter,
        generation_adapter,
        open_judgment,
    )

    with generation_adapter() as port:
        pass  # port.generate(...)
    with open_judgment() as session:
        pass  # session.port.judge(...)
    async_port = async_vllm_generation_adapter()  # await async_port.generate(...)
    ```

See Also:
    - [typevet.adapters.inbound.settings][]: ``TYPEVET_LLAMA__*`` settings
    - [typevet.adapters.outbound.llama_cpp][]: llama.cpp adapters and the Gemma
      native vision factory module
    - [typevet.adapters.outbound.vllm.generation][]: vLLM generation adapter
    - [typevet.adapters.outbound.vllm.generation_async][]: Async vLLM adapter
    - [typevet.adapters.outbound.vllm.judgment_factory][]: vLLM judgment factory
    - [typevet.adapters.diagnostics.redaction][]: ``REDACTED`` (``***``) mask
    - [typevet.adapters.inbound.error_masking][]: Masked error copies
    - [typevet.adapters.inbound.fake_backend][]: Offline ``fake`` judgment
    - docs/reference/configuration.md: Environment variable reference
"""

from __future__ import annotations

import math
import os
from collections.abc import Iterator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Literal

import httpx

from typevet.adapters.diagnostics.redaction import REDACTED
from typevet.adapters.inbound.error_masking import (
    KeyMaskingJudgmentPort as _KeyMaskingJudgmentPort,
)
from typevet.adapters.inbound.error_masking import enter_masked, key_needles
from typevet.adapters.inbound.error_masking import masked_if_keyed as _masked_if_keyed
from typevet.adapters.inbound.fake_backend import (
    FakeJudgmentSession,
    open_fake_judgment,
)
from typevet.adapters.inbound.gateway_headers import (
    async_event_hooks,
    sync_event_hooks,
)
from typevet.adapters.inbound.http_headers import (
    check_auth_scheme,
    check_gateway_headers,
    check_header_name,
    client_headers,
    parse_headers_json,
)
from typevet.adapters.inbound.settings import (
    llama_cpp_adapter,
    llama_http_client,
    load_llama_settings,
)
from typevet.adapters.outbound.llama_cpp.gemma_native_vision_factory import (
    GemmaNativeVisionSession,
    open_gemma_native_vision_judgment,
)
from typevet.adapters.outbound.llama_cpp.generation import LlamaCppGenerationAdapter
from typevet.adapters.outbound.vllm.generation import VllmGenerationAdapter
from typevet.adapters.outbound.vllm.generation_async import AsyncVllmGenerationAdapter
from typevet.adapters.outbound.vllm.judgment_factory import (
    VllmJudgmentSession,
    open_vllm_judgment,
)
from typevet.domain.errors import GenerationError
from typevet.domain.models import GenerationRequest, GenerationResult

Backend = Literal["llama_cpp", "vllm", "fake"]

_DEFAULT_TIMEOUT = 300.0
_ENV = "TYPEVET_VLLM__"
_BACKENDS: tuple[Backend, ...] = ("llama_cpp", "vllm", "fake")
MASK = REDACTED


@dataclass(frozen=True, slots=True)
class VllmSettings:
    """Connection options for a vLLM OpenAI-compatible server.

    Attributes:
        base_url (str): Server root without a trailing slash.
        model (str): Served model name.
        timeout (float): HTTP request timeout in seconds.
        api_key (str | None): Key sent in ``auth_header``, or ``None``.
            Excluded from ``repr``.
        max_concurrency (int): Maximum POSTs in flight for one async adapter.
        user_agent (str | None): ``User-Agent`` header value, or ``None`` for
            the httpx default.
        auth_header (str): Header that carries the key.
        auth_scheme (str): Scheme before the key; empty sends the key bare.
        headers (Mapping[str, str]): Extra headers with literal values, sent
            on every request. A read-only copy after the checks. Excluded
            from ``repr``.
        request_id_header (str | None): Header that gets a fresh UUID4 hex
            value on each request, or ``None``.

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
    user_agent: str | None = None
    auth_header: str = "Authorization"
    auth_scheme: str = "Bearer"
    headers: Mapping[str, str] = field(default_factory=dict, repr=False)
    request_id_header: str | None = None

    def __post_init__(self) -> None:
        """Copy ``headers`` read-only and check the gateway header rules.

        Raises:
            TypeError: When ``headers`` is not a mapping.
            ValueError: When a gateway field breaks a rule of
                ``typevet.adapters.inbound.gateway_headers``. The message
                names the field and never holds a value.
        """
        if not isinstance(self.headers, Mapping):
            msg = f"headers ({_ENV}HEADERS) must be a mapping of header values"
            raise TypeError(msg)
        object.__setattr__(self, "headers", MappingProxyType(dict(self.headers)))
        check_header_name(self.auth_header, f"auth_header ({_ENV}AUTH_HEADER)")
        check_auth_scheme(self.auth_scheme, f"auth_scheme ({_ENV}AUTH_SCHEME)")
        check_gateway_headers(
            self.headers, auth_header=self.auth_header, field=f"headers ({_ENV}HEADERS)"
        )
        if self.request_id_header is not None:
            label = f"request_id_header ({_ENV}REQUEST_ID_HEADER)"
            name = check_header_name(self.request_id_header, label).lower()
            taken = {self.auth_header.lower(), *(k.lower() for k in self.headers)}
            if name in taken:
                msg = f"{label} names a header that is already set"
                raise ValueError(msg)


class _ClientOwningVllmAdapter(VllmGenerationAdapter):
    """vLLM adapter that owns its client and masks the key in errors.

    Attributes:
        _settings_client (httpx.Client): Client from ``vllm_http_client``.
        _needles (Needles): Patterns for the key and header values, or empty.

    Examples:
        ```python
        settings = VllmSettings(base_url="http://127.0.0.1:8000", model="m")
        _ClientOwningVllmAdapter(settings, vllm_http_client(settings))
        ```
    """

    def __init__(self, settings: VllmSettings, client: httpx.Client) -> None:
        super().__init__(settings.base_url, timeout=settings.timeout, client=client)
        self._settings_client = client
        self._needles = key_needles(settings.api_key, settings.headers)

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate, and mask the configured key in any raised error.

        A successful result is returned unchanged. When a key is configured,
        ``_masked_if_keyed`` copies each error as the same type with the key
        replaced by ``MASK`` and with no cause or context. The copy is raised
        outside the ``except`` block, so the original error is not reachable
        from it.

        Args:
            request: Prompt, schema and served model name.

        Returns:
            The result from ``VllmGenerationAdapter.generate``.

        Raises:
            GenerationError: The adapter error, as a masked copy when a key is set.
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


class _ClientOwningAsyncVllmAdapter(AsyncVllmGenerationAdapter):
    """Async vLLM adapter that owns its client and masks the key in errors.

    Attributes:
        _settings_client (httpx.AsyncClient): Client with the same base URL,
            timeout and headers as ``vllm_http_client``.
        _needles (Needles): Patterns for the key and header values, or empty.

    Examples:
        ```python
        settings = VllmSettings(base_url="http://127.0.0.1:8000", model="m")
        _ClientOwningAsyncVllmAdapter(settings, httpx.AsyncClient())
        ```
    """

    def __init__(self, settings: VllmSettings, client: httpx.AsyncClient) -> None:
        super().__init__(
            settings.base_url,
            timeout=settings.timeout,
            client=client,
            max_concurrency=settings.max_concurrency,
        )
        self._settings_client = client
        self._needles = key_needles(settings.api_key, settings.headers)
        self._binds_loop = True

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate, and mask the configured key in any raised error.

        Masking follows ``_ClientOwningVllmAdapter.generate``: when a key is
        configured, each error is raised again as a copy of the same type
        with the key replaced by ``MASK`` and with no cause or context. A
        successful result is returned unchanged.

        Args:
            request: Prompt, schema and served model name.

        Returns:
            The result from ``AsyncVllmGenerationAdapter.generate``.

        Raises:
            GenerationError: The adapter error, as a masked copy when a key is set.
            RuntimeError: When the call runs on a different event loop from
                the first call. It is not masked, because it holds no key.
        """
        try:
            return await super().generate(request)
        except GenerationError as exc:
            masked = _masked_if_keyed(exc, self._needles)
            if masked is None:
                raise
        raise masked

    async def close(self) -> None:
        """Close the composition-root client."""
        await super().close()
        await self._settings_client.aclose()


def _read_timeout(source: Mapping[str, str], name: str) -> float:
    raw = source.get(name, "").strip()
    if not raw:
        return _DEFAULT_TIMEOUT
    try:
        timeout = float(raw)
    except ValueError:
        timeout = 0.0
    if not math.isfinite(timeout) or timeout <= 0:
        msg = f"{name} must be a finite positive number of seconds"
        raise ValueError(msg) from None
    return timeout


def _read_max_concurrency(source: Mapping[str, str], name: str) -> int:
    raw = source.get(name, "").strip()
    if not raw:
        return 1
    try:
        limit = int(raw)
    except ValueError:
        limit = 0
    if limit < 1:
        msg = f"{name} must be a positive integer"
        raise ValueError(msg) from None
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
        ValueError: When the value is not ``llama_cpp``, ``vllm`` or ``fake``.
    """
    source = os.environ if environ is None else environ
    raw = source.get("TYPEVET_BACKEND", "").strip()
    if not raw:
        return "llama_cpp"
    for backend in _BACKENDS:
        if raw == backend:
            return backend
    msg = "TYPEVET_BACKEND must be llama_cpp, vllm or fake"
    raise ValueError(msg)


def load_vllm_settings(environ: Mapping[str, str] | None = None) -> VllmSettings:
    """Read ``TYPEVET_VLLM__*`` variables.

    Args:
        environ: Mapping to read. Defaults to ``os.environ``.

    Returns:
        Frozen settings. An empty ``TYPEVET_VLLM__API_KEY``,
        ``TYPEVET_VLLM__USER_AGENT`` or ``TYPEVET_VLLM__REQUEST_ID_HEADER``
        gives ``None``. An empty ``TYPEVET_VLLM__AUTH_HEADER`` gives
        ``Authorization``. An unset ``TYPEVET_VLLM__AUTH_SCHEME`` gives
        ``Bearer``; a set but empty one sends the key bare.
        ``TYPEVET_VLLM__HEADERS`` is a JSON object of literal string values.

    Raises:
        ValueError: When ``TYPEVET_VLLM__BASE_URL`` or ``TYPEVET_VLLM__MODEL``
            is missing, ``TYPEVET_VLLM__TIMEOUT`` is not a finite positive
            number,
            ``TYPEVET_VLLM__MAX_CONCURRENCY`` is not a positive integer,
            ``TYPEVET_VLLM__API_KEY`` holds a non-ASCII character, or a
            gateway variable breaks a header rule. No message holds a value.
    """
    source = os.environ if environ is None else environ
    api_key = source.get("TYPEVET_VLLM__API_KEY", "").strip()
    if not api_key.isascii():
        msg = "TYPEVET_VLLM__API_KEY must contain only ASCII characters"
        raise ValueError(msg)
    scheme = source.get(f"{_ENV}AUTH_SCHEME")
    return VllmSettings(
        base_url=_read_required(source, "TYPEVET_VLLM__BASE_URL").rstrip("/"),
        model=_read_required(source, "TYPEVET_VLLM__MODEL"),
        timeout=_read_timeout(source, "TYPEVET_VLLM__TIMEOUT"),
        api_key=api_key or None,
        max_concurrency=_read_max_concurrency(source, "TYPEVET_VLLM__MAX_CONCURRENCY"),
        user_agent=source.get("TYPEVET_VLLM__USER_AGENT", "").strip() or None,
        auth_header=source.get(f"{_ENV}AUTH_HEADER", "").strip() or "Authorization",
        auth_scheme="Bearer" if scheme is None else scheme.strip(),
        headers=parse_headers_json(source.get(f"{_ENV}HEADERS", ""), f"{_ENV}HEADERS"),
        request_id_header=source.get(f"{_ENV}REQUEST_ID_HEADER", "").strip() or None,
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
        A client with ``base_url`` and ``timeout`` set. ``client_headers``
        gives its headers: the extra ``settings.headers``, the key in
        ``settings.auth_header`` after ``settings.auth_scheme`` when a key is
        set, which by default gives ``Authorization: Bearer <key>``, and the
        ``User-Agent`` when a user agent is set. The event hooks from
        ``sync_event_hooks`` add the request id and refuse redirects. Nothing
        else changes, so environment proxy settings apply with or without a
        key. The async client from ``async_vllm_generation_adapter`` gets the
        same headers and hooks.
    """
    return httpx.Client(
        base_url=settings.base_url,
        timeout=settings.timeout,
        headers=client_headers(settings),
        transport=transport,
        event_hooks=sync_event_hooks(settings.request_id_header),
    )


def generation_adapter(
    environ: Mapping[str, str] | None = None,
    *,
    transport: httpx.BaseTransport | None = None,
) -> LlamaCppGenerationAdapter | VllmGenerationAdapter:
    """Build the generation adapter that ``TYPEVET_BACKEND`` selects.

    Args:
        environ: Mapping to read. Defaults to ``os.environ``.
        transport: Optional transport for the vLLM or llama.cpp client.

    Returns:
        A llama.cpp adapter on the ``llama_http_client`` from
        ``load_llama_settings``, or a vLLM adapter on the
        ``vllm_http_client``. Closing either adapter closes its client, and
        its errors never show the configured key.

    Raises:
        ValueError: When a backend or vLLM variable is invalid, or when
            ``TYPEVET_BACKEND`` is ``fake``, which has no generation adapter.
    """
    source = os.environ if environ is None else environ
    backend = load_backend(source)
    if backend == "fake":
        msg = "TYPEVET_BACKEND=fake: the fake backend has no generation adapter"
        raise ValueError(msg)
    if backend == "llama_cpp":
        return llama_cpp_adapter(load_llama_settings(source), transport=transport)
    settings = load_vllm_settings(source)
    client = vllm_http_client(settings, transport=transport)
    return _ClientOwningVllmAdapter(settings, client)


def async_vllm_generation_adapter(
    environ: Mapping[str, str] | None = None,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> AsyncVllmGenerationAdapter:
    """Build the async vLLM generation adapter from ``TYPEVET_VLLM__*``.

    The adapter reads the vLLM settings whatever ``TYPEVET_BACKEND`` says. Its
    ``httpx.AsyncClient`` has the same base URL, timeout, ``client_headers``,
    event hooks and proxy handling as ``vllm_http_client``, and ``TYPEVET_VLLM__MAX_CONCURRENCY``
    sets its POST limit. That client binds to the first event loop that uses
    it, so build one adapter per event loop, for example per ``asyncio.run``.
    The adapter records the first running loop that calls ``generate``, and a
    call on a different loop raises ``RuntimeError`` before any request.

    Args:
        environ: Mapping to read. Defaults to ``os.environ``.
        transport: Optional async transport, for example
            ``httpx.MockTransport``.

    Returns:
        An adapter that closes its client on ``close`` and whose errors never
        show the configured key.

    Raises:
        ValueError: When a vLLM variable is invalid.
    """
    settings = load_vllm_settings(environ)
    client = httpx.AsyncClient(
        base_url=settings.base_url,
        timeout=settings.timeout,
        headers=client_headers(settings),
        transport=transport,
        event_hooks=async_event_hooks(settings.request_id_header),
    )
    return _ClientOwningAsyncVllmAdapter(settings, client)


@contextmanager
def open_judgment(
    environ: Mapping[str, str] | None = None,
    *,
    transport: httpx.BaseTransport | None = None,
) -> Iterator[GemmaNativeVisionSession | VllmJudgmentSession | FakeJudgmentSession]:
    """Open the judgment session that ``TYPEVET_BACKEND`` selects.

    Args:
        environ: Mapping to read. Defaults to ``os.environ``.
        transport: Optional transport for the vLLM or llama.cpp client. The
            fake branch ignores it.

    Yields:
        A llama.cpp session from ``open_gemma_native_vision_judgment`` with
        ``load_llama_settings`` on the ``llama_http_client``, which closes on
        exit; the session-open and judgment errors never show the configured
        key or a header value. Or a vLLM session from ``open_vllm_judgment``
        on the ``vllm_http_client``, so ``/tokenize`` and scoring carry the
        gateway headers. The vLLM client closes on exit, and judgment errors
        never show the configured key or a header value. For ``fake``, an
        offline ``FakeJudgmentSession`` with ``model`` ``"fake"`` whose port
        answers from ``TYPEVET_FAKE__DISTRIBUTIONS`` or uniform distributions.

    Raises:
        ValueError: When a backend, vLLM or ``TYPEVET_FAKE__DISTRIBUTIONS``
            variable is invalid.
    """
    source = os.environ if environ is None else environ
    backend = load_backend(source)
    if backend == "fake":
        yield open_fake_judgment(source)
        return
    if backend == "llama_cpp":
        llama = load_llama_settings(source)
        needles = key_needles(llama.api_key, llama.headers)
        with (
            llama_http_client(llama, transport=transport) as http,
            ExitStack() as stack,
        ):
            opened = open_gemma_native_vision_judgment(settings=llama, http_client=http)
            llama_session = enter_masked(stack, opened, needles)
            port = _KeyMaskingJudgmentPort(llama_session.port, needles)
            yield replace(llama_session, port=port)
        return
    settings = load_vllm_settings(source)
    with (
        vllm_http_client(settings, transport=transport) as client,
        open_vllm_judgment(
            client=client, model=settings.model, base_url=settings.base_url
        ) as session,
    ):
        needles = key_needles(settings.api_key, settings.headers)
        masked = _KeyMaskingJudgmentPort(session.port, needles)
        yield replace(session, port=masked)
