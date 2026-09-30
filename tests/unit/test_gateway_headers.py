"""Unit tests for the vLLM gateway header settings and their limits (#331).

Each rule is checked when ``VllmSettings`` is built, directly or from
``TYPEVET_VLLM__*``. An error names the field and never holds a value.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from typevet.adapters.inbound.backend_settings import VllmSettings, load_vllm_settings

pytestmark = pytest.mark.unit

_V = "VALUE-SENTINEL"


def _settings(**fields: Any) -> VllmSettings:
    return VllmSettings(base_url="https://gw.example.com/vllm", model="m", **fields)


def _env(**extra: str) -> dict[str, str]:
    env = {
        "TYPEVET_VLLM__BASE_URL": "https://gw.example.com/vllm",
        "TYPEVET_VLLM__MODEL": "m",
    }
    env.update(extra)
    return env


def _total(size: int) -> dict[str, str]:
    """Four headers whose names and values add up to ``size`` bytes."""
    names = ("X-A", "X-B", "X-C", "X-D")
    rest = size - sum(len(n) for n in names) - 3 * 2048
    return {n: ("v" * (2048 if i < 3 else rest)) for i, n in enumerate(names)}


_INVALID_HEADERS: dict[str, dict[str, str]] = {
    "hop_by_hop": {"Connection": _V},
    "hop_by_hop_case": {"transfer-encoding": _V},
    "proxy_authorization": {"Proxy-Authorization": _V},
    "host": {"HOST": _V},
    "content_length": {"Content-Length": _V},
    "default_auth_header": {"authorization": _V},
    "non_token_name": {"Bad Name": _V},
    "non_ascii_value": {"X-Tenant": f"café-{_V}"},
    "control_value": {"X-Tenant": f"{_V}\r\nX-Injected: 1"},
    "duplicate_name": {"X-Tenant": _V, "x-tenant": _V},
    "too_many_fields": {f"X-H{i}": f"{_V}{i}" for i in range(33)},
    "long_name": {"X-" + "a" * 127: _V},
    "long_value": {"X-Tenant": _V + "v" * (2049 - len(_V))},
    "long_total": {k: _V + v[len(_V) :] for k, v in _total(8193).items()},
}


@pytest.mark.parametrize("headers", _INVALID_HEADERS.values(), ids=_INVALID_HEADERS)
def test_invalid_headers_rejected_by_field_without_value(
    headers: dict[str, str],
) -> None:
    with pytest.raises(ValueError, match="headers") as info:
        _settings(headers=headers)
    assert _V not in str(info.value)
    assert "café" not in str(info.value)
    with pytest.raises(ValueError, match="TYPEVET_VLLM__HEADERS") as env_info:
        load_vllm_settings(_env(TYPEVET_VLLM__HEADERS=json.dumps(headers)))
    assert _V not in str(env_info.value)
    assert env_info.value.__cause__ is None


def test_limits_are_inclusive() -> None:
    at_limit = {
        "fields": {f"X-H{i}": "v" for i in range(32)},
        "name": {"X-" + "a" * 126: "v"},
        "value": {"X-Tenant": "v" * 2048},
        "total": _total(8192),
    }
    for headers in at_limit.values():
        assert _settings(headers=headers).headers == headers


def test_custom_auth_header_may_be_sent_as_extra_authorization() -> None:
    settings = _settings(auth_header="X-API-Key", headers={"Authorization": "Basic x"})
    assert settings.headers == {"Authorization": "Basic x"}


def test_custom_auth_header_is_protected_from_headers() -> None:
    with pytest.raises(ValueError, match="headers") as info:
        _settings(auth_header="X-API-Key", headers={"x-api-key": _V})
    assert _V not in str(info.value)


@pytest.mark.parametrize(
    ("fields", "field"),
    [
        ({"auth_header": "Host"}, "auth_header"),
        ({"auth_header": "Bad Name"}, "auth_header"),
        ({"auth_header": ""}, "auth_header"),
        ({"auth_scheme": "Bearer x"}, "auth_scheme"),
        ({"request_id_header": "Connection"}, "request_id_header"),
        ({"request_id_header": "Bad Name"}, "request_id_header"),
        ({"request_id_header": "authorization"}, "request_id_header"),
        (
            {"request_id_header": "X-Request-Id", "headers": {"x-request-id": _V}},
            "request_id_header",
        ),
    ],
)
def test_invalid_names_rejected_by_field(fields: dict[str, Any], field: str) -> None:
    with pytest.raises(ValueError, match=field) as info:
        _settings(**fields)
    assert _V not in str(info.value)


@pytest.mark.parametrize(
    "raw",
    ["not json", "[1, 2]", '{"X-Tenant": 1}', '{"X-Tenant": null}', '"text"'],
)
def test_headers_env_must_be_a_json_object_of_strings(raw: str) -> None:
    with pytest.raises(ValueError, match="TYPEVET_VLLM__HEADERS") as info:
        load_vllm_settings(_env(TYPEVET_VLLM__HEADERS=raw))
    assert raw not in str(info.value)
    assert info.value.__cause__ is None
    assert info.value.__context__ is None or info.value.__suppress_context__


def test_env_reads_gateway_fields() -> None:
    settings = load_vllm_settings(
        _env(
            TYPEVET_VLLM__AUTH_HEADER=" X-API-Key ",
            TYPEVET_VLLM__AUTH_SCHEME="",
            TYPEVET_VLLM__HEADERS='{"X-Tenant": "acme"}',
            TYPEVET_VLLM__REQUEST_ID_HEADER="X-Request-Id",
        )
    )
    assert settings.auth_header == "X-API-Key"
    assert settings.auth_scheme == ""
    assert settings.headers == {"X-Tenant": "acme"}
    assert settings.request_id_header == "X-Request-Id"
    assert "acme" not in repr(settings)


def test_env_defaults_keep_bearer_authorization() -> None:
    settings = load_vllm_settings(
        _env(TYPEVET_VLLM__AUTH_HEADER=" ", TYPEVET_VLLM__HEADERS=" ")
    )
    assert settings.auth_header == "Authorization"
    assert settings.auth_scheme == "Bearer"
    assert settings.headers == {}
    assert settings.request_id_header is None


def test_header_values_are_literal() -> None:
    settings = load_vllm_settings(
        _env(TYPEVET_VLLM__HEADERS='{"X-A": "$HOME", "X-B": "!echo hi"}')
    )
    assert settings.headers == {"X-A": "$HOME", "X-B": "!echo hi"}


def test_headers_must_be_a_mapping() -> None:
    with pytest.raises(TypeError, match="headers"):
        _settings(headers=[("X-Tenant", _V)])


def test_headers_are_copied_at_construction() -> None:
    source = {"X-Tenant": "acme"}
    settings = _settings(headers=source)
    source["X-Tenant"] = "changed"
    assert settings.headers == {"X-Tenant": "acme"}
