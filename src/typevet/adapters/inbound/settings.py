"""Environment-backed llama.cpp settings for the composition root.

Outbound adapters take explicit constructor arguments. Only a composition root
(CLI, MCP server, or live test harness) reads ``TYPEVET_LLAMA__*`` variables
and passes the values into [typevet.adapters.outbound.llama_cpp][].

A router behind an API gateway needs a key and extra headers (#410).
``TYPEVET_LLAMA__API_KEY``, ``TYPEVET_LLAMA__AUTH_HEADER``,
``TYPEVET_LLAMA__AUTH_SCHEME``, ``TYPEVET_LLAMA__HEADERS`` and
``TYPEVET_LLAMA__USER_AGENT`` follow the ``TYPEVET_VLLM__*`` rules, and
[typevet.adapters.inbound.http_headers][] checks them when ``LlamaSettings``
is built. ``llama_http_client`` and ``async_llama_http_client`` send them.
``LlamaSettings`` leaves the key and the header values out of ``repr``. The
adapter from ``llama_cpp_adapter`` owns its client, and with a key or extra
headers each error it raises is a masked copy with no cause or context, as
in [typevet.adapters.inbound.error_masking][].

``TYPEVET_LLAMA__REQUEST_ID_HEADER`` names a header that gets a fresh UUID4
hex value on each request, as ``TYPEVET_VLLM__REQUEST_ID_HEADER`` does
(#411). The same name rules apply. The clients get one request hook that
calls ``stamp_request_id`` of [typevet.adapters.outbound.request_ids][]; they
get no response hook. A judgment response maps each question name to the id
of its ``/completion`` request in ``request_ids``. typevet never logs the id.

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
    - [typevet.adapters.outbound.llama_cpp][]: llama.cpp adapter package; the sync
      adapter lives in its ``generation`` module
    - [typevet.adapters.diagnostics.settings][]: ``TYPEVET_LOG__*`` settings
    - [typevet.adapters.inbound.http_headers][]: Shared header rules
    - [typevet.adapters.outbound.request_ids][]: Request id hook and receipt
    - docs/reference/configuration.md: Environment variable reference
    - docs/how-to/run-a-multimodal-live-smoke.md: ``multimodal_model`` consumer
"""

from __future__ import annotations

import os
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import httpx

from typevet.adapters.inbound.error_masking import key_needles, masked_if_keyed
from typevet.adapters.inbound.http_headers import (
    check_auth_scheme,
    check_gateway_headers,
    check_header_name,
    client_headers,
    parse_headers_json,
)
from typevet.adapters.outbound.llama_cpp.generation import LlamaCppGenerationAdapter
from typevet.adapters.outbound.request_ids import stamp_request_id
from typevet.domain.errors import GenerationError
from typevet.domain.models import GenerationRequest, GenerationResult

_ENV = "TYPEVET_LLAMA__"
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
        api_key (str | None): Key sent in ``auth_header``, or ``None``.
            Excluded from ``repr``.
        headers (Mapping[str, str]): Extra headers with literal values, sent
            on every request. A read-only copy after the checks. Excluded
            from ``repr``.
        auth_header (str): Header that carries the key.
        auth_scheme (str): Scheme before the key; empty sends the key bare.
        user_agent (str | None): ``User-Agent`` header value, or ``None`` for
            the httpx default.
        request_id_header (str | None): Header that gets a fresh UUID4 hex
            value on each request, or ``None`` to send none.

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
    api_key: str | None = field(default=None, repr=False)
    headers: Mapping[str, str] = field(default_factory=dict, repr=False)
    auth_header: str = "Authorization"
    auth_scheme: str = "Bearer"
    user_agent: str | None = None
    request_id_header: str | None = None

    def __post_init__(self) -> None:
        """Copy ``headers`` read-only and check the header rules.

        ``request_id_header`` follows the name rules and must not name the
        auth header or an extra header (#411).

        Raises:
            TypeError: When ``headers`` is not a mapping.
            ValueError: When a header field breaks a rule of
                ``typevet.adapters.inbound.http_headers``, or
                ``request_id_header`` names a header that is already set. The
                message names the field and never holds a value.
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


def _read_api_key(source: Mapping[str, str]) -> str | None:
    api_key = source.get(f"{_ENV}API_KEY", "").strip()
    if not api_key.isascii():
        msg = f"{_ENV}API_KEY must contain only ASCII characters"
        raise ValueError(msg)
    return api_key or None


def load_llama_settings(
    environ: Mapping[str, str] | None = None,
) -> LlamaSettings:
    """Read ``TYPEVET_LLAMA__*`` variables for the composition root.

    Legacy single-segment names ``TYPEVET_LLAMA_URL`` and ``TYPEVET_GEMMA_MODEL``
    remain supported when the nested names are unset.
    ``TYPEVET_LLAMA__MULTIMODAL_MODEL`` names the model id for the opt-in image
    smoke and has no legacy alias. The key and header variables follow the
    ``TYPEVET_VLLM__*`` rules (#410).

    Args:
        environ: Mapping to read. Defaults to ``os.environ``.

    Returns:
        Frozen settings with defaults for missing keys. An empty
        ``TYPEVET_LLAMA__API_KEY`` or ``TYPEVET_LLAMA__USER_AGENT`` gives
        ``None``. An empty ``TYPEVET_LLAMA__AUTH_HEADER`` gives
        ``Authorization``. An unset ``TYPEVET_LLAMA__AUTH_SCHEME`` gives
        ``Bearer``; a set but empty one sends the key bare.
        ``TYPEVET_LLAMA__HEADERS`` is a JSON object of literal string values.
        An empty ``TYPEVET_LLAMA__REQUEST_ID_HEADER`` gives ``None``.

    Raises:
        ValueError: When ``TYPEVET_LLAMA__TIMEOUT`` is not a positive number,
            ``TYPEVET_LLAMA__API_KEY`` holds a non-ASCII character, or a
            header variable breaks a header rule. No message holds a value.
            ``TYPEVET_LLAMA__REQUEST_ID_HEADER`` also must not name the auth
            header or an extra header.
    """
    source = os.environ if environ is None else environ
    scheme = source.get(f"{_ENV}AUTH_SCHEME")
    return LlamaSettings(
        base_url=_read_base_url(source),
        timeout=_read_timeout(source),
        default_model=_read_default_model(source),
        multimodal_model=_read_multimodal_model(source),
        api_key=_read_api_key(source),
        headers=parse_headers_json(source.get(f"{_ENV}HEADERS", ""), f"{_ENV}HEADERS"),
        auth_header=source.get(f"{_ENV}AUTH_HEADER", "").strip() or "Authorization",
        auth_scheme="Bearer" if scheme is None else scheme.strip(),
        user_agent=source.get(f"{_ENV}USER_AGENT", "").strip() or None,
        request_id_header=source.get(f"{_ENV}REQUEST_ID_HEADER", "").strip() or None,
    )


def _sync_request_hooks(
    header: str | None,
) -> dict[str, list[Callable[[httpx.Request], None]]]:
    def stamp(request: httpx.Request) -> None:
        """Set the request-id header when it is configured and absent.

        ``stamp_request_id`` records the sent value in
        ``request.extensions``.

        Args:
            request: Outgoing request.
        """
        stamp_request_id(request, header)

    return {"request": [stamp], "response": []}


def _async_request_hooks(
    header: str | None,
) -> dict[str, list[Callable[[httpx.Request], Awaitable[None]]]]:
    async def stamp(request: httpx.Request) -> None:
        """Set the request-id header when it is configured and absent.

        ``stamp_request_id`` records the sent value in
        ``request.extensions``.

        Args:
            request: Outgoing request.
        """
        stamp_request_id(request, header)

    return {"request": [stamp], "response": []}


def llama_http_client(
    settings: LlamaSettings,
    *,
    transport: httpx.BaseTransport | None = None,
) -> httpx.Client:
    """Build a llama.cpp HTTP client that sends the key and headers.

    Args:
        settings: llama.cpp connection settings.
        transport: Optional transport, for example ``httpx.MockTransport``.

    Returns:
        A client with ``base_url`` and ``timeout`` set and the headers from
        ``client_headers``: the extra ``settings.headers``, the key in
        ``settings.auth_header`` after ``settings.auth_scheme`` when a key is
        set, and the ``User-Agent`` when a user agent is set. Without a key or
        headers it sends the httpx defaults. Environment proxy settings apply.
        Its one request hook sets a fresh UUID4 hex value in
        ``settings.request_id_header`` when that is set and the request has
        none; it has no response hook.
    """
    return httpx.Client(
        base_url=settings.base_url,
        timeout=settings.timeout,
        headers=client_headers(settings),
        transport=transport,
        event_hooks=_sync_request_hooks(settings.request_id_header),
    )


def async_llama_http_client(
    settings: LlamaSettings,
    *,
    transport: httpx.AsyncBaseTransport | None = None,
) -> httpx.AsyncClient:
    """Build a llama.cpp async HTTP client with the same headers.

    The caller owns the client and closes it. Errors that an adapter raises
    on it are not masked here.

    Args:
        settings: llama.cpp connection settings.
        transport: Optional async transport, for example
            ``httpx.MockTransport``.

    Returns:
        An ``httpx.AsyncClient`` with the base URL, timeout, headers and
        request-id hook of ``llama_http_client``.
    """
    return httpx.AsyncClient(
        base_url=settings.base_url,
        timeout=settings.timeout,
        headers=client_headers(settings),
        transport=transport,
        event_hooks=_async_request_hooks(settings.request_id_header),
    )


class _ClientOwningLlamaAdapter(LlamaCppGenerationAdapter):
    """llama.cpp adapter that owns its client and masks the key in errors.

    Attributes:
        _settings_client (httpx.Client): Client from ``llama_http_client``.
        _needles (Needles): Patterns for the key and header values, or empty.

    Examples:
        ```python
        settings = LlamaSettings(base_url="http://127.0.0.1:8090")
        _ClientOwningLlamaAdapter(settings, llama_http_client(settings))
        ```
    """

    def __init__(self, settings: LlamaSettings, client: httpx.Client) -> None:
        """Wrap ``client`` with the base URL and timeout of ``settings``.

        Args:
            settings: llama.cpp settings that hold the key and headers.
            client: Client from ``llama_http_client``; closed by ``close``.
        """
        super().__init__(settings.base_url, timeout=settings.timeout, client=client)
        self._settings_client = client
        self._needles = key_needles(settings.api_key, settings.headers)

    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate, and mask the configured key in any raised error.

        A successful result is returned unchanged. With a key or extra
        headers, ``masked_if_keyed`` copies each error as the same type with
        each match replaced by ``***`` and with no cause or context. The copy
        is raised outside the ``except`` block, so the original error is not
        reachable from it.

        Args:
            request: Prompt, schema and model alias on the router.

        Returns:
            The result from ``LlamaCppGenerationAdapter.generate``.

        Raises:
            GenerationError: The adapter error, as a masked copy when a key or
                header value is set.
        """
        try:
            return super().generate(request)
        except GenerationError as exc:
            masked = masked_if_keyed(exc, self._needles)
            if masked is None:
                raise
        raise masked

    def close(self) -> None:
        """Close the composition-root client."""
        super().close()
        self._settings_client.close()


def llama_cpp_adapter(
    settings: LlamaSettings,
    *,
    transport: httpx.BaseTransport | None = None,
) -> LlamaCppGenerationAdapter:
    """Build a sync llama.cpp adapter from composition-root settings.

    Args:
        settings: Values read from the environment or constructed in tests.
        transport: Optional transport for the client, for example
            ``httpx.MockTransport``.

    Returns:
        An adapter on the ``llama_http_client`` that does not read
        ``os.environ`` itself. Closing it closes the client. With a key or
        extra headers, its errors are masked copies with no cause or context.

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
    client = llama_http_client(settings, transport=transport)
    return _ClientOwningLlamaAdapter(settings, client)
