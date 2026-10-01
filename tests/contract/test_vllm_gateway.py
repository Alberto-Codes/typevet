"""Contract tests: typevet reaches a vLLM server behind an API gateway (#331).

The composition-root clients from ``generation_adapter``,
``async_vllm_generation_adapter`` and ``open_judgment`` run the real vLLM
adapters over ``httpx.MockTransport``. The mock plays the gateway: it records
each request and answers under the ``/vllm`` path prefix. The tests prove the
path prefix, the configured auth header and scheme, the extra headers, the
request-id header, the 429 and redirect answers of a gateway, and that no key
or header value leaves through ``repr`` or an error chain.

[i331]: https://github.com/Alberto-Codes/typevet/issues/331
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable, Iterator
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from pathlib import Path
from typing import Any

import httpx
import pytest

from typevet.adapters.inbound.backend_settings import (
    async_vllm_generation_adapter,
    generation_adapter,
    load_vllm_settings,
    open_judgment,
)
from typevet.domain.errors import BackendHttpError, GenerationError, TransportError
from typevet.domain.judgment_questions import Noul
from typevet.domain.models import GenerationRequest

pytestmark = pytest.mark.contract

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "vllm"
_BASE = "https://gw.example.com/vllm"
_MODEL = "served-model"
_KEY = "sk-GATEWAY-SENTINEL-331"
_TENANT = "acme-tenant-SENTINEL"
_ROUTE = "blue-route-SENTINEL"
_EXTRA = {"X-Tenant": _TENANT, "X-Route": _ROUTE}
_SECRETS = (_KEY, _TENANT, _ROUTE)
_SCHEMA = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"],
    "additionalProperties": False,
}
_REPLY = {"choices": [{"message": {"content": '{"ok": true}'}}]}
_HTML = "<html><body><h1>429 Too Many Requests</h1>PAGE-SENTINEL</body></html>"
_LOCATION = "https://login.example.com/sso?next=LOCATION-SENTINEL"
Handler = Callable[[httpx.Request], httpx.Response]


def _fixture(name: str) -> dict[str, Any]:
    return json.loads((_FIXTURES / f"{name}.json").read_text(encoding="utf-8"))


def _gateway_env(**extra: str) -> dict[str, str]:
    env = {
        "TYPEVET_BACKEND": "vllm",
        "TYPEVET_VLLM__BASE_URL": _BASE,
        "TYPEVET_VLLM__MODEL": _MODEL,
        "TYPEVET_VLLM__API_KEY": _KEY,
        "TYPEVET_VLLM__AUTH_HEADER": "X-API-Key",
        "TYPEVET_VLLM__AUTH_SCHEME": "",
        "TYPEVET_VLLM__HEADERS": json.dumps(_EXTRA),
    }
    env.update(extra)
    return env


class _Gateway:
    """MockTransport handler that serves vLLM under the ``/vllm`` prefix."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self._scoring = _fixture("text_three_way")["response"]
        self._tokens = _fixture("tokenize_ordinals")["responses"]

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        body = json.loads(request.content.decode())
        if request.url.path == "/vllm/tokenize":
            return httpx.Response(200, json=self._tokens[body["prompt"]])
        if request.url.path == "/vllm/v1/chat/completions":
            reply = self._scoring if "logprobs" in body else _REPLY
            return httpx.Response(200, json=reply)
        return httpx.Response(404, json={"error": "unexpected path"})


def _request() -> GenerationRequest:
    return GenerationRequest(prompt="hi", schema=_SCHEMA, model=_MODEL)


def _generate(env: dict[str, str], handler: Handler, calls: int = 1) -> None:
    adapter = generation_adapter(env, transport=httpx.MockTransport(handler))
    with adapter:
        for _ in range(calls):
            adapter.generate(_request())


def _generate_async(env: dict[str, str], handler: Handler) -> None:
    adapter = async_vllm_generation_adapter(env, transport=httpx.MockTransport(handler))

    async def run() -> None:
        try:
            await adapter.generate(_request())
        finally:
            await adapter.close()

    asyncio.run(run())


def _judge(env: dict[str, str], handler: Handler) -> None:
    with open_judgment(env, transport=httpx.MockTransport(handler)) as session:
        session.port.judge("state", {"f": Noul(instructions="Is it ok?")}, _MODEL)


def _sync_error(env: dict[str, str], handler: Handler) -> GenerationError:
    with pytest.raises(GenerationError) as info:
        _generate(env, handler)
    return info.value


def _async_error(env: dict[str, str], handler: Handler) -> GenerationError:
    with pytest.raises(GenerationError) as info:
        _generate_async(env, handler)
    return info.value


def _judge_error(env: dict[str, str], handler: Handler) -> GenerationError:
    with pytest.raises(GenerationError) as info:
        _judge(env, handler)
    return info.value


_ENTRY_POINTS = pytest.mark.parametrize(
    "raised", [_sync_error, _async_error, _judge_error], ids=["sync", "async", "judge"]
)


def _chain(exc: BaseException) -> Iterator[BaseException]:
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = current.__cause__ or current.__context__


def _http_texts(value: object) -> Iterator[str]:
    if isinstance(value, httpx.Request):
        yield str(value.headers.raw)
    elif isinstance(value, httpx.Response):
        yield str(value.headers.raw)
        yield value.text


def _exposed(exc: BaseException) -> str:
    parts: list[str] = []
    for link in _chain(exc):
        parts += [str(link), repr(link), repr(vars(link))]
        for name in ("request", "response"):
            try:
                parts += _http_texts(getattr(link, name, None))
            except RuntimeError:
                continue
        tb = link.__traceback__
        while tb is not None:
            for value in tb.tb_frame.f_locals.values():
                parts += _http_texts(value)
            tb = tb.tb_next
    return "\n".join(parts)


@pytest.mark.parametrize("base", [_BASE, _BASE + "/"], ids=["bare", "slash"])
def test_prefix_with_and_without_trailing_slash(base: str) -> None:
    env = _gateway_env(TYPEVET_VLLM__BASE_URL=base)
    gateway = _Gateway()
    _generate(env, gateway)
    _judge(env, gateway)
    urls = [str(request.url) for request in gateway.requests]
    assert urls[0] == "https://gw.example.com/vllm/v1/chat/completions"
    assert set(urls[1:]) == {
        "https://gw.example.com/vllm/tokenize",
        "https://gw.example.com/vllm/v1/chat/completions",
    }
    assert all(r.headers.get("X-API-Key") == _KEY for r in gateway.requests)


def test_custom_auth_header_without_scheme() -> None:
    gateway = _Gateway()
    _generate(_gateway_env(), gateway)
    _generate_async(_gateway_env(), gateway)
    _judge(_gateway_env(), gateway)
    assert len(gateway.requests) >= 3
    for request in gateway.requests:
        assert request.headers["X-API-Key"] == _KEY
        assert "Authorization" not in request.headers


def test_default_auth_unchanged() -> None:
    env = {
        "TYPEVET_BACKEND": "vllm",
        "TYPEVET_VLLM__BASE_URL": _BASE,
        "TYPEVET_VLLM__MODEL": _MODEL,
        "TYPEVET_VLLM__API_KEY": _KEY,
    }
    gateway = _Gateway()
    _generate(env, gateway)
    _generate_async(env, gateway)
    _judge(env, gateway)
    for request in gateway.requests:
        assert request.headers["Authorization"] == f"Bearer {_KEY}"
        assert "X-API-Key" not in request.headers
        assert "X-Tenant" not in request.headers


def test_extra_headers_sent() -> None:
    gateway = _Gateway()
    _judge(_gateway_env(), gateway)
    _generate_async(_gateway_env(), gateway)
    paths = {request.url.path for request in gateway.requests}
    assert paths == {"/vllm/tokenize", "/vllm/v1/chat/completions"}
    for request in gateway.requests:
        assert request.headers["X-Tenant"] == _TENANT
        assert request.headers["X-Route"] == _ROUTE


def test_request_id_header_fresh_per_request() -> None:
    env = _gateway_env(TYPEVET_VLLM__REQUEST_ID_HEADER="X-Request-Id")
    gateway = _Gateway()
    _generate(env, gateway, calls=2)
    _generate_async(env, gateway)
    ids = [request.headers["X-Request-Id"] for request in gateway.requests]
    assert len(ids) == 3
    assert len(set(ids)) == 3
    assert all(re.fullmatch(r"[0-9a-f]{32}", value) for value in ids)
    unset = _Gateway()
    _generate(_gateway_env(), unset)
    assert "X-Request-Id" not in unset.requests[0].headers


def _too_many(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        429,
        text=_HTML,
        headers={"Content-Type": "text/html; charset=utf-8", "Retry-After": "7"},
        request=request,
    )


@_ENTRY_POINTS
def test_gateway_429_html_body_is_a_backend_http_error_and_not_echoed(
    raised: Callable[[dict[str, str], Handler], GenerationError],
) -> None:
    err = raised(_gateway_env(), _too_many)
    assert type(err) is BackendHttpError
    assert err.status_code == 429
    assert err.body_snippet == ""
    assert "PAGE-SENTINEL" not in _exposed(err)
    assert "<html>" not in str(err)


@_ENTRY_POINTS
def test_redirect_not_followed_and_location_not_disclosed(
    raised: Callable[[dict[str, str], Handler], GenerationError],
) -> None:
    seen: list[httpx.Request] = []

    def redirect(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            302,
            text=f'<a href="{_LOCATION}">moved</a>',
            headers={"Location": _LOCATION, "Content-Type": "text/html"},
            request=request,
        )

    err = raised(_gateway_env(), redirect)
    assert len(seen) == 1
    assert type(err) is BackendHttpError
    assert err.status_code == 302
    assert "LOCATION-SENTINEL" not in _exposed(err)


@_ENTRY_POINTS
def test_no_header_value_in_repr_or_errors(
    raised: Callable[[dict[str, str], Handler], GenerationError],
) -> None:
    text = repr(load_vllm_settings(_gateway_env()))
    assert not [secret for secret in _SECRETS if secret in text]
    sent: list[httpx.Request] = []

    def refuse(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        raise httpx.ConnectError("connection refused", request=request)

    err = raised(_gateway_env(), refuse)
    assert sent[0].headers["X-Tenant"] == _TENANT
    assert type(err) is TransportError
    assert "connection refused" in str(err)
    exposed = _exposed(err)
    assert not [secret for secret in _SECRETS if secret in exposed]
    assert err.__cause__ is None
    assert err.__context__ is None


@_ENTRY_POINTS
def test_direct_errors_pass_through_unmasked(
    raised: Callable[[dict[str, str], Handler], GenerationError],
) -> None:
    env = {
        "TYPEVET_BACKEND": "vllm",
        "TYPEVET_VLLM__BASE_URL": _BASE,
        "TYPEVET_VLLM__MODEL": _MODEL,
    }

    def refuse(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    err = raised(env, refuse)
    assert type(err) is TransportError
    assert isinstance(err.__cause__, httpx.ConnectError)


@_ENTRY_POINTS
def test_header_values_masked_without_a_key(
    raised: Callable[[dict[str, str], Handler], GenerationError],
) -> None:
    env = _gateway_env()
    del env["TYPEVET_VLLM__API_KEY"]

    def echo(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401, json={"error": f"bad tenant {request.headers['X-Tenant']}"}
        )

    err = raised(env, echo)
    assert type(err) is BackendHttpError
    assert _TENANT not in _exposed(err)
    assert err.__cause__ is None


@_ENTRY_POINTS
def test_short_header_value_masks_whole_tokens_only(
    raised: Callable[[dict[str, str], Handler], GenerationError],
) -> None:
    """A header value ``1`` masks the token ``1``, not the ``1`` in ``401`` (#348)."""
    headers = json.dumps({"X-Tenant": "1", "X-Route": _ROUTE})

    def deny(request: httpx.Request) -> httpx.Response:
        detail = f"tenant 1 denied; key x{_KEY}y; route {_ROUTE}"
        return httpx.Response(401, json={"detail": detail}, request=request)

    err = raised(_gateway_env(TYPEVET_VLLM__HEADERS=headers), deny)
    assert type(err) is BackendHttpError
    assert err.status_code == 401
    assert "HTTP 401" in str(err)
    assert "tenant *** denied" in str(err)
    exposed = _exposed(err)
    assert "tenant 1 denied" not in exposed
    assert not [secret for secret in _SECRETS if secret in exposed]


_LIMITS = {
    "X-RateLimit-Remaining": "0",
    "X-RateLimit-Limit": "100",
    "RateLimit-Reset": "7",
    "X-Other": "ignored",
}
_EXPECTED_LIMITS = {
    "x-ratelimit-remaining": "0",
    "x-ratelimit-limit": "100",
    "ratelimit-reset": "7",
}
_DATE = "Wed, 30 Sep 2026 12:00:00 GMT"


def _limited(headers: dict[str, str], *, html: bool = True) -> Handler:
    """Return a gateway that answers 429 with ``_LIMITS`` and ``headers``."""

    def handler(request: httpx.Request) -> httpx.Response:
        sent = {**_LIMITS, **headers}
        if html:
            sent["Content-Type"] = "text/html"
            return httpx.Response(429, text=_HTML, headers=sent, request=request)
        body = {"error": "slow down"}
        return httpx.Response(429, json=body, headers=sent, request=request)

    return handler


@_ENTRY_POINTS
@pytest.mark.parametrize("html", [True, False], ids=["html", "json"])
def test_gateway_429_exposes_retry_after_seconds_and_rate_limit(
    raised: Callable[[dict[str, str], Handler], GenerationError], html: bool
) -> None:
    """A delta-seconds ``Retry-After`` and rate-limit headers reach the caller (#355)."""
    sent: list[httpx.Request] = []
    gateway = _limited({"Retry-After": "7"}, html=html)

    def counted(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return gateway(request)

    err = raised(_gateway_env(), counted)
    assert len(sent) == 1
    assert type(err) is BackendHttpError
    assert err.status_code == 429
    assert err.retry_after_seconds == 7.0
    assert dict(err.rate_limit) == _EXPECTED_LIMITS
    writable: Any = err.rate_limit
    with pytest.raises(TypeError):
        writable["x-ratelimit-remaining"] = "1"


@_ENTRY_POINTS
@pytest.mark.parametrize("html", [True, False], ids=["html", "json"])
def test_gateway_429_http_date_is_relative_to_the_date_header(
    raised: Callable[[dict[str, str], Handler], GenerationError], html: bool
) -> None:
    """An HTTP-date ``Retry-After`` counts from the response ``Date`` (#355)."""
    later = "Wed, 30 Sep 2026 12:00:30 GMT"
    handler = _limited({"Retry-After": later, "Date": _DATE}, html=html)
    err = raised(_gateway_env(), handler)
    assert type(err) is BackendHttpError
    assert err.retry_after_seconds == 30.0


def test_gateway_429_http_date_without_date_header_counts_from_now() -> None:
    """Without ``Date``, the HTTP-date counts from now; a past date gives 0 (#355)."""
    ahead = datetime.now(UTC) + timedelta(seconds=120)
    later = format_datetime(ahead, usegmt=True)
    err = _sync_error(_gateway_env(), _limited({"Retry-After": later}))
    assert type(err) is BackendHttpError
    assert err.retry_after_seconds is not None
    assert 100.0 < err.retry_after_seconds <= 120.0
    past = _sync_error(_gateway_env(), _limited({"Retry-After": _DATE}))
    assert type(past) is BackendHttpError
    assert past.retry_after_seconds == 0.0


@pytest.mark.parametrize(
    "value", ["soon", "-1", "1.5", "", "7, 8", "Wed, 31 Feb 2026 99:00:00 GMT"]
)
def test_gateway_429_invalid_retry_after_is_none(value: str) -> None:
    """A ``Retry-After`` that is neither form gives ``None`` (#355)."""
    err = _sync_error(_gateway_env(), _limited({"Retry-After": value}, html=False))
    assert type(err) is BackendHttpError
    assert err.retry_after_seconds is None
    assert dict(err.rate_limit) == _EXPECTED_LIMITS


@_ENTRY_POINTS
def test_gateway_429_without_retry_after_is_none(
    raised: Callable[[dict[str, str], Handler], GenerationError],
) -> None:
    """No ``Retry-After`` gives ``None``; no rate-limit header gives empty (#355)."""

    def bare(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": "slow"}, request=request)

    err = raised(_gateway_env(), bare)
    assert type(err) is BackendHttpError
    assert err.retry_after_seconds is None
    assert dict(err.rate_limit) == {}


@_ENTRY_POINTS
def test_gateway_429_rate_limit_never_holds_auth_or_configured_headers(
    raised: Callable[[dict[str, str], Handler], GenerationError],
) -> None:
    """Echoed auth and configured headers stay out of ``rate_limit`` (#355)."""
    tier = "gold-tier-SENTINEL"
    configured = json.dumps({**_EXTRA, "X-RateLimit-Tier": tier})
    env = _gateway_env(
        TYPEVET_VLLM__HEADERS=configured, TYPEVET_VLLM__AUTH_HEADER="RateLimit-Key"
    )
    echoed = {
        "X-RateLimit-Tier": tier,
        "RateLimit-Key": _KEY,
        "Authorization": f"Bearer {_KEY}",
        "Retry-After": "7",
    }
    err = raised(env, _limited(echoed, html=False))
    assert type(err) is BackendHttpError
    assert dict(err.rate_limit) == _EXPECTED_LIMITS
    text = f"{err} {err!r} {vars(err)!r}"
    assert not [secret for secret in (*_SECRETS, tier) if secret in text]


@_ENTRY_POINTS
def test_gateway_429_rate_limit_values_mask_key_and_header_values(
    raised: Callable[[dict[str, str], Handler], GenerationError],
) -> None:
    """A key or header value inside a rate-limit value is masked (#355 F1)."""
    policy = f"tenant={_TENANT};key={_KEY}"
    handler = _limited({"X-RateLimit-Policy": policy}, html=False)
    err = raised(_gateway_env(), handler)
    assert type(err) is BackendHttpError
    assert err.rate_limit["x-ratelimit-remaining"] == "0"
    assert not [secret for secret in _SECRETS if secret in _exposed(err)]
    writable: Any = err.rate_limit
    with pytest.raises(TypeError):
        writable["x-ratelimit-policy"] = "1"


def test_gateway_429_retry_after_over_one_year_is_none() -> None:
    """A delta-seconds value above one year gives ``None`` (#355 F2)."""
    huge = _sync_error(_gateway_env(), _limited({"Retry-After": "9" * 400}))
    assert type(huge) is BackendHttpError
    assert huge.retry_after_seconds is None
    year = _sync_error(_gateway_env(), _limited({"Retry-After": "31536000"}))
    assert type(year) is BackendHttpError
    assert year.retry_after_seconds == 31_536_000.0
