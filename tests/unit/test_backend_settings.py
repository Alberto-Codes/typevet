"""Unit tests for backend selection and vLLM composition-root settings."""

from __future__ import annotations

import asyncio
import io
import json
import traceback
from collections.abc import Callable, Mapping

import httpx
import pytest
import structlog

from typevet.adapters.diagnostics.logs import configure
from typevet.adapters.diagnostics.settings import LogSettings
from typevet.adapters.inbound.backend_settings import (
    MASK,
    VllmSettings,
    async_vllm_generation_adapter,
    generation_adapter,
    load_backend,
    load_vllm_settings,
    vllm_http_client,
)
from typevet.adapters.inbound.settings import load_llama_settings
from typevet.adapters.outbound.llama_cpp import LlamaCppGenerationAdapter
from typevet.adapters.outbound.vllm_generation import VllmGenerationAdapter
from typevet.adapters.outbound.vllm_generation_async import AsyncVllmGenerationAdapter
from typevet.domain.errors import (
    BackendHttpError,
    GenerationError,
    SchemaValidationError,
    TransportError,
)
from typevet.domain.models import GenerationRequest, GenerationResult

SENTINEL = "sk-SENTINEL-4f1c9e"
_SCHEMA = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"],
    "additionalProperties": False,
}
_REPLY = {"choices": [{"message": {"content": '{"ok": true}'}}]}


def _vllm_env(**extra: str) -> dict[str, str]:
    env = {
        "TYPEVET_BACKEND": "vllm",
        "TYPEVET_VLLM__BASE_URL": "http://vllm.test:9000/",
        "TYPEVET_VLLM__MODEL": "served-model",
    }
    env.update(extra)
    return env


def _request() -> GenerationRequest:
    return GenerationRequest(prompt="hi", schema=_SCHEMA, model="served-model")


@pytest.mark.unit
def test_backend_defaults_to_llama_cpp_when_unset() -> None:
    assert load_backend({}) == "llama_cpp"
    assert load_backend({"TYPEVET_BACKEND": "  "}) == "llama_cpp"
    assert load_backend({"TYPEVET_BACKEND": "vllm"}) == "vllm"


@pytest.mark.unit
def test_unknown_backend_rejected_without_echoing_value() -> None:
    with pytest.raises(ValueError, match="TYPEVET_BACKEND") as info:
        load_backend({"TYPEVET_BACKEND": "secret-backend-name"})
    assert "secret-backend-name" not in str(info.value)


@pytest.mark.unit
def test_unset_backend_builds_llama_adapter_from_llama_settings() -> None:
    env = {
        "TYPEVET_LLAMA__BASE_URL": "http://llama.test:1/",
        "TYPEVET_LLAMA__TIMEOUT": "7",
    }
    expected = load_llama_settings(env)
    adapter = generation_adapter(env)
    assert isinstance(adapter, LlamaCppGenerationAdapter)
    assert adapter._base_url == expected.base_url + "/"
    assert adapter._timeout == expected.timeout == 7.0


@pytest.mark.unit
def test_load_vllm_settings_reads_all_fields() -> None:
    settings = load_vllm_settings(
        _vllm_env(TYPEVET_VLLM__API_KEY=SENTINEL, TYPEVET_VLLM__TIMEOUT="12.5")
    )
    assert settings == VllmSettings(
        base_url="http://vllm.test:9000",
        model="served-model",
        timeout=12.5,
        api_key=SENTINEL,
    )
    defaults = load_vllm_settings(_vllm_env(TYPEVET_VLLM__API_KEY=""))
    assert defaults.timeout == 300.0
    assert defaults.api_key is None


@pytest.mark.unit
@pytest.mark.parametrize(
    ("env", "name"),
    [
        ({"TYPEVET_VLLM__MODEL": "m"}, "TYPEVET_VLLM__BASE_URL"),
        ({"TYPEVET_VLLM__BASE_URL": "http://h"}, "TYPEVET_VLLM__MODEL"),
        (_vllm_env(TYPEVET_VLLM__TIMEOUT="soon-ish"), "TYPEVET_VLLM__TIMEOUT"),
        (_vllm_env(TYPEVET_VLLM__TIMEOUT="-3"), "TYPEVET_VLLM__TIMEOUT"),
        (_vllm_env(TYPEVET_VLLM__TIMEOUT="0"), "TYPEVET_VLLM__TIMEOUT"),
    ],
)
def test_invalid_vllm_values_rejected_by_name(env: dict[str, str], name: str) -> None:
    with pytest.raises(ValueError, match=name) as info:
        load_vllm_settings(env)
    assert "soon-ish" not in str(info.value)
    assert "-3" not in str(info.value)


@pytest.mark.unit
def test_api_key_absent_from_settings_repr() -> None:
    settings = load_vllm_settings(_vllm_env(TYPEVET_VLLM__API_KEY=SENTINEL))
    assert SENTINEL not in repr(settings)
    assert SENTINEL not in str(settings)


@pytest.mark.unit
def test_client_carries_bearer_header_base_url_and_timeout() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={})

    settings = load_vllm_settings(
        _vllm_env(TYPEVET_VLLM__API_KEY=SENTINEL, TYPEVET_VLLM__TIMEOUT="9")
    )
    with vllm_http_client(settings, transport=httpx.MockTransport(handler)) as client:
        assert client.timeout == httpx.Timeout(9.0)
        assert str(client.base_url) == "http://vllm.test:9000"
        client.get("/v1/models")
    assert seen[0].headers["Authorization"] == f"Bearer {SENTINEL}"
    assert str(seen[0].url) == "http://vllm.test:9000/v1/models"


@pytest.mark.unit
def test_client_omits_authorization_without_key() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={})

    settings = load_vllm_settings(_vllm_env(TYPEVET_VLLM__TIMEOUT="6"))
    with vllm_http_client(settings, transport=httpx.MockTransport(handler)) as client:
        assert client.timeout == httpx.Timeout(6.0)
        client.get("/v1/models")
    assert "Authorization" not in seen[0].headers


@pytest.mark.unit
def test_vllm_backend_adapter_posts_to_configured_host_with_header() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_REPLY)

    env = _vllm_env(TYPEVET_VLLM__API_KEY=SENTINEL, TYPEVET_VLLM__TIMEOUT="4")
    adapter = generation_adapter(env, transport=httpx.MockTransport(handler))
    assert isinstance(adapter, VllmGenerationAdapter)
    with adapter:
        result = adapter.generate(_request())
    assert result.value == {"ok": True}
    assert str(seen[0].url) == "http://vllm.test:9000/v1/chat/completions"
    assert seen[0].headers["Authorization"] == f"Bearer {SENTINEL}"
    assert seen[0].extensions["timeout"]["read"] == 4.0


def _echoing_401(request: httpx.Request) -> httpx.Response:
    return httpx.Response(401, text=f"bad key {request.headers['Authorization']}")


@pytest.mark.unit
def test_401_error_that_echoes_the_key_is_redacted() -> None:
    env = _vllm_env(TYPEVET_VLLM__API_KEY=SENTINEL)
    adapter = generation_adapter(env, transport=httpx.MockTransport(_echoing_401))
    with adapter, pytest.raises(BackendHttpError) as info:
        adapter.generate(_request())
    err = info.value
    assert err.status_code == 401
    assert "bad key Bearer ***" in str(err)
    assert SENTINEL not in str(err)
    assert SENTINEL not in err.body_snippet
    assert SENTINEL not in repr(err)


@pytest.mark.unit
def test_key_absent_from_structlog_output() -> None:
    stream = io.StringIO()
    configure(LogSettings(format="json", level="debug"), stream)
    try:
        settings = load_vllm_settings(_vllm_env(TYPEVET_VLLM__API_KEY=SENTINEL))
        adapter = generation_adapter(
            _vllm_env(TYPEVET_VLLM__API_KEY=SENTINEL),
            transport=httpx.MockTransport(_echoing_401),
        )
        log = structlog.get_logger()
        log.info("settings", settings=repr(settings), api_key=settings.api_key)
        try:
            adapter.generate(_request())
        except BackendHttpError as exc:
            log.exception("call failed", error=str(exc))
        adapter.close()
    finally:
        structlog.reset_defaults()
    lines = [json.loads(line) for line in stream.getvalue().splitlines()]
    assert len(lines) == 2
    assert "bad key Bearer ***" in lines[1]["error"]
    assert SENTINEL not in stream.getvalue()


@pytest.mark.unit
def test_closing_vllm_adapter_closes_its_client() -> None:
    adapter = generation_adapter(
        _vllm_env(), transport=httpx.MockTransport(_echoing_401)
    )
    assert isinstance(adapter, VllmGenerationAdapter)
    client = adapter._ensure_client()
    adapter.close()
    assert client.is_closed


_QUOTED_KEY = 'sk-"quo/te"-9d2'
_NOTE_SCHEMA = {
    "type": "object",
    "properties": {"note": {"type": "string"}},
    "required": ["note"],
    "additionalProperties": False,
}


def _keyed_adapter(
    key: str, handler: Callable[[httpx.Request], httpx.Response]
) -> VllmGenerationAdapter:
    adapter = generation_adapter(
        _vllm_env(TYPEVET_VLLM__API_KEY=key),
        transport=httpx.MockTransport(handler),
    )
    assert isinstance(adapter, VllmGenerationAdapter)
    return adapter


def _raised(
    adapter: VllmGenerationAdapter, schema: Mapping[str, object]
) -> BaseException:
    with adapter, pytest.raises(GenerationError) as info:
        adapter.generate(GenerationRequest(prompt="hi", schema=schema, model="m"))
    return info.value


def _all_text(exc: BaseException) -> str:
    return repr(exc) + "".join(traceback.format_exception(exc))


@pytest.mark.unit
def test_json_escaped_key_is_masked_and_chain_is_dropped() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            401, json={"detail": f"bad key {request.headers['Authorization']}"}
        )

    err = _raised(_keyed_adapter(_QUOTED_KEY, handler), _SCHEMA)
    assert isinstance(err, BackendHttpError)
    assert err.status_code == 401
    escaped = json.dumps(_QUOTED_KEY)[1:-1]
    assert escaped in json.dumps(f"x{_QUOTED_KEY}")
    for text in (_all_text(err), err.body_snippet):
        assert _QUOTED_KEY not in text
        assert escaped not in text
    assert "bad key Bearer ***" in str(err)
    assert err.__cause__ is None
    assert err.__context__ is None


@pytest.mark.unit
def test_transport_error_carrying_the_key_is_masked() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        msg = f"refused {request.headers['Authorization']}"
        raise httpx.ConnectError(msg, request=request)

    err = _raised(_keyed_adapter(SENTINEL, handler), _SCHEMA)
    assert type(err) is TransportError
    assert "refused Bearer ***" in str(err)
    assert SENTINEL not in _all_text(err)


@pytest.mark.unit
def test_error_without_the_key_is_raised_unchanged() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    err = _raised(_keyed_adapter(SENTINEL, handler), _SCHEMA)
    assert isinstance(err, BackendHttpError)
    assert err.body_snippet == "boom"
    assert err.__traceback__ is not None


@pytest.mark.unit
def test_successful_content_containing_the_key_is_not_altered() -> None:
    content = json.dumps({"note": f"echo {SENTINEL}"})

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"choices": [{"message": {"content": content}}]}
        )

    with _keyed_adapter(SENTINEL, handler) as adapter:
        result = adapter.generate(
            GenerationRequest(prompt="hi", schema=_NOTE_SCHEMA, model="m")
        )
    assert result.raw_text == content
    assert result.value == {"note": f"echo {SENTINEL}"}


@pytest.mark.unit
def test_proxy_mounts_match_with_and_without_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("HTTP_PROXY", "ALL_PROXY", "NO_PROXY"):
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.lower(), raising=False)
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.test:3128")
    keyed = vllm_http_client(
        load_vllm_settings(_vllm_env(TYPEVET_VLLM__API_KEY=SENTINEL))
    )
    plain = vllm_http_client(load_vllm_settings(_vllm_env()))
    with keyed, plain:
        keyed_patterns = sorted(p.pattern for p in keyed._mounts)
        plain_patterns = sorted(p.pattern for p in plain._mounts)
    assert plain_patterns == ["https://"]
    assert keyed_patterns == plain_patterns


@pytest.mark.unit
def test_non_ascii_api_key_rejected_by_name() -> None:
    key = "sk-clé-7a"
    with pytest.raises(ValueError, match="TYPEVET_VLLM__API_KEY") as info:
        load_vllm_settings(_vllm_env(TYPEVET_VLLM__API_KEY=key))
    assert key not in str(info.value)
    assert info.value.__cause__ is None
    assert info.value.__context__ is None


_OPEN_SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}}


def _content_reply(content: str) -> Callable[[httpx.Request], httpx.Response]:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"choices": [{"message": {"content": content}}]}
        )

    return handler


@pytest.mark.unit
@pytest.mark.parametrize(
    ("content", "schema"),
    [
        (json.dumps({"ok": SENTINEL}), _SCHEMA),
        (json.dumps({"note": f"x {SENTINEL}", "extra": [SENTINEL]}), _NOTE_SCHEMA),
        (json.dumps([SENTINEL]), _SCHEMA),
        (json.dumps({"ok": "x", SENTINEL: 1}), _OPEN_SCHEMA),
    ],
)
def test_schema_failure_echoing_the_key_masks_payload_and_drops_chain(
    content: str, schema: Mapping[str, object]
) -> None:
    err = _raised(_keyed_adapter(SENTINEL, _content_reply(content)), schema)
    assert isinstance(err, SchemaValidationError)
    assert SENTINEL not in repr(err.payload)
    assert SENTINEL not in str(err)
    assert SENTINEL not in _all_text(err)
    assert MASK in repr(err.payload)
    assert err.__cause__ is None
    assert err.__context__ is None


@pytest.mark.unit
@pytest.mark.parametrize(
    ("name", "raw"),
    [
        ("TYPEVET_VLLM__TIMEOUT", "soon-ish-5b7e"),
        ("TYPEVET_VLLM__MAX_CONCURRENCY", "many-5b7e"),
    ],
)
def test_invalid_number_drops_the_parse_error_chain(name: str, raw: str) -> None:
    with pytest.raises(ValueError, match=name) as info:
        load_vllm_settings(_vllm_env(**{name: raw}))
    err = info.value
    assert err.__cause__ is None
    assert err.__suppress_context__
    assert raw not in "".join(traceback.format_exception(err))


async def _gathered(
    adapter: AsyncVllmGenerationAdapter, count: int
) -> list[GenerationResult]:
    async with adapter:
        return await asyncio.gather(
            *(adapter.generate(_request()) for _ in range(count))
        )


@pytest.mark.unit
def test_async_adapter_applies_bearer_user_agent_timeout_and_limit() -> None:
    seen: list[httpx.Request] = []
    in_flight: list[int] = [0]
    peaks: list[int] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        in_flight[0] += 1
        peaks.append(in_flight[0])
        for _ in range(5):
            await asyncio.sleep(0)
        in_flight[0] -= 1
        return httpx.Response(200, json=_REPLY)

    env = _vllm_env(
        TYPEVET_VLLM__API_KEY=SENTINEL,
        TYPEVET_VLLM__TIMEOUT="4",
        TYPEVET_VLLM__USER_AGENT="typevet-test/1",
        TYPEVET_VLLM__MAX_CONCURRENCY="2",
    )
    adapter = async_vllm_generation_adapter(env, transport=httpx.MockTransport(handler))
    assert isinstance(adapter, AsyncVllmGenerationAdapter)
    results = asyncio.run(_gathered(adapter, 5))
    assert [result.value for result in results] == [{"ok": True}] * 5
    assert max(peaks) == 2
    for request in seen:
        assert str(request.url) == "http://vllm.test:9000/v1/chat/completions"
        assert request.headers["Authorization"] == f"Bearer {SENTINEL}"
        assert request.headers["User-Agent"] == "typevet-test/1"
        assert request.extensions["timeout"]["read"] == 4.0


@pytest.mark.unit
def test_async_adapter_without_key_or_user_agent_sends_defaults() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_REPLY)

    adapter = async_vllm_generation_adapter(
        _vllm_env(), transport=httpx.MockTransport(handler)
    )
    asyncio.run(_gathered(adapter, 1))
    assert "Authorization" not in seen[0].headers
    assert seen[0].headers["User-Agent"].startswith("python-httpx/")
    assert seen[0].extensions["timeout"]["read"] == 300.0


@pytest.mark.unit
def test_async_401_echoing_the_key_is_masked_without_chain() -> None:
    adapter = async_vllm_generation_adapter(
        _vllm_env(TYPEVET_VLLM__API_KEY=SENTINEL),
        transport=httpx.MockTransport(_echoing_401),
    )
    with pytest.raises(BackendHttpError) as info:
        asyncio.run(_gathered(adapter, 1))
    err = info.value
    assert err.status_code == 401
    assert "bad key Bearer ***" in str(err)
    assert SENTINEL not in err.body_snippet
    assert SENTINEL not in _all_text(err)
    assert err.__cause__ is None
    assert err.__context__ is None


@pytest.mark.unit
def test_async_error_without_the_key_is_raised_unchanged() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    adapter = async_vllm_generation_adapter(
        _vllm_env(TYPEVET_VLLM__API_KEY=SENTINEL),
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(BackendHttpError) as info:
        asyncio.run(_gathered(adapter, 1))
    assert info.value.body_snippet == "boom"


@pytest.mark.unit
def test_async_adapter_closes_its_client() -> None:
    adapter = async_vllm_generation_adapter(
        _vllm_env(), transport=httpx.MockTransport(_echoing_401)
    )
    client = adapter._ensure_client()
    asyncio.run(adapter.close())
    assert client.is_closed


@pytest.mark.unit
def test_async_proxy_mounts_match_sync_client_with_and_without_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("HTTP_PROXY", "ALL_PROXY", "NO_PROXY"):
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.lower(), raising=False)
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.test:3128")
    for env in (_vllm_env(TYPEVET_VLLM__API_KEY=SENTINEL), _vllm_env()):
        with vllm_http_client(load_vllm_settings(env)) as sync_client:
            expected = sorted(p.pattern for p in sync_client._mounts)
        adapter = async_vllm_generation_adapter(env)
        client = adapter._ensure_client()
        assert sorted(p.pattern for p in client._mounts) == expected == ["https://"]
        asyncio.run(adapter.close())
