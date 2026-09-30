"""Contract tests: a configured vLLM key never leaves through an error chain.

The composition-root wrappers from ``generation_adapter``,
``async_vllm_generation_adapter`` and ``open_judgment`` run the real vLLM
adapters over ``httpx.MockTransport``. Each test walks the raised error, its
``__cause__`` and ``__context__`` chain, the ``request`` and ``response`` of
each error, and the locals of each traceback frame. The key must be absent
from all of them ([#276][i276]).

[i276]: https://github.com/Alberto-Codes/typevet/issues/276
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator

import httpx
import pytest

from typevet.adapters.inbound.backend_settings import (
    async_vllm_generation_adapter,
    generation_adapter,
    open_judgment,
)
from typevet.domain.errors import BackendHttpError, GenerationError, TransportError
from typevet.domain.judgment_questions import Noul
from typevet.domain.models import GenerationRequest

pytestmark = pytest.mark.contract

_KEY = "sk-SENTINEL-276c"
_MODEL = "served-model"
_SCHEMA = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"],
    "additionalProperties": False,
}
Handler = Callable[[httpx.Request], httpx.Response]


def _env() -> dict[str, str]:
    return {
        "TYPEVET_BACKEND": "vllm",
        "TYPEVET_VLLM__BASE_URL": "http://vllm.test:9000",
        "TYPEVET_VLLM__MODEL": _MODEL,
        "TYPEVET_VLLM__API_KEY": _KEY,
    }


def _refuse(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("connection refused", request=request)


def _unavailable(request: httpx.Request) -> httpx.Response:
    return httpx.Response(503, text="service unavailable", request=request)


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
        try:
            yield from _http_texts(value.request)
        except RuntimeError:
            return


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
                if isinstance(value, BaseException) and value is not link:
                    parts.append(_exposed(value))
            tb = tb.tb_next
    return "\n".join(parts)


def _request() -> GenerationRequest:
    return GenerationRequest(prompt="hi", schema=_SCHEMA, model=_MODEL)


def _sync_error(handler: Handler) -> GenerationError:
    adapter = generation_adapter(_env(), transport=httpx.MockTransport(handler))
    with adapter, pytest.raises(GenerationError) as info:
        adapter.generate(_request())
    return info.value


def _async_error(handler: Handler) -> GenerationError:
    adapter = async_vllm_generation_adapter(
        _env(), transport=httpx.MockTransport(handler)
    )

    async def run() -> None:
        try:
            await adapter.generate(_request())
        finally:
            await adapter.close()

    with pytest.raises(GenerationError) as info:
        asyncio.run(run())
    return info.value


def _judge_error(handler: Handler) -> GenerationError:
    transport = httpx.MockTransport(handler)
    with (
        open_judgment(_env(), transport=transport) as session,
        pytest.raises(GenerationError) as info,
    ):
        session.port.judge("state", {"f": Noul()}, _MODEL)
    return info.value


_ENTRY_POINTS = pytest.mark.parametrize(
    "raised", [_sync_error, _async_error, _judge_error], ids=["sync", "async", "judge"]
)


@_ENTRY_POINTS
def test_connect_error_chain_does_not_expose_the_key(
    raised: Callable[[Handler], GenerationError],
) -> None:
    err = raised(_refuse)
    assert type(err) is TransportError
    assert "connection refused" in str(err)
    assert _KEY not in _exposed(err)
    assert err.__cause__ is None
    assert err.__context__ is None


@_ENTRY_POINTS
def test_status_error_without_echo_does_not_expose_the_key(
    raised: Callable[[Handler], GenerationError],
) -> None:
    err = raised(_unavailable)
    assert type(err) is BackendHttpError
    assert err.status_code == 503
    assert err.body_snippet == "service unavailable"
    assert _KEY not in _exposed(err)
    assert err.__cause__ is None
    assert err.__context__ is None
