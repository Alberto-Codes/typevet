"""Unit tests for llama.cpp composition-root settings."""

from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from typing import Any, cast

import pytest

from typevet.adapters.inbound.settings import (
    LlamaSettings,
    llama_cpp_adapter,
    load_llama_settings,
)
from typevet.adapters.outbound.llama_cpp.generation import LlamaCppGenerationAdapter


@pytest.mark.unit
def test_load_llama_settings_defaults() -> None:
    settings = load_llama_settings({})
    assert settings == LlamaSettings()


@pytest.mark.unit
def test_load_llama_settings_nested_env() -> None:
    settings = load_llama_settings(
        {
            "TYPEVET_LLAMA__BASE_URL": "http://127.0.0.1:9000",
            "TYPEVET_LLAMA__TIMEOUT": "120.5",
            "TYPEVET_LLAMA__DEFAULT_MODEL": "gemma-test",
        }
    )
    assert settings.base_url == "http://127.0.0.1:9000"
    assert settings.timeout == 120.5
    assert settings.default_model == "gemma-test"


@pytest.mark.unit
def test_load_llama_settings_defaults_the_multimodal_model() -> None:
    assert load_llama_settings({}).multimodal_model == "gemma-3-4b-it-q4km-mm"


@pytest.mark.unit
@pytest.mark.parametrize(
    "raw",
    ["gemma-vision", "  gemma-vision  "],
    ids=["plain", "padded"],
)
def test_load_llama_settings_reads_multimodal_model(raw: str) -> None:
    settings = load_llama_settings({"TYPEVET_LLAMA__MULTIMODAL_MODEL": raw})
    assert settings.multimodal_model == "gemma-vision"


@pytest.mark.unit
def test_load_llama_settings_ignores_blank_multimodal_model() -> None:
    settings = load_llama_settings({"TYPEVET_LLAMA__MULTIMODAL_MODEL": "   "})
    assert settings.multimodal_model == "gemma-3-4b-it-q4km-mm"


@pytest.mark.unit
def test_load_llama_settings_legacy_aliases() -> None:
    settings = load_llama_settings(
        {
            "TYPEVET_LLAMA_URL": "http://127.0.0.1:8091",
            "TYPEVET_GEMMA_MODEL": "legacy-model",
        }
    )
    assert settings.base_url == "http://127.0.0.1:8091"
    assert settings.default_model == "legacy-model"


@pytest.mark.unit
def test_load_llama_settings_nested_overrides_legacy() -> None:
    settings = load_llama_settings(
        {
            "TYPEVET_LLAMA_URL": "http://127.0.0.1:8091",
            "TYPEVET_LLAMA__BASE_URL": "http://127.0.0.1:9000",
            "TYPEVET_GEMMA_MODEL": "legacy-model",
            "TYPEVET_LLAMA__DEFAULT_MODEL": "nested-model",
        }
    )
    assert settings.base_url == "http://127.0.0.1:9000"
    assert settings.default_model == "nested-model"


@pytest.mark.unit
def test_load_llama_settings_rejects_non_positive_timeout() -> None:
    with pytest.raises(ValueError, match="TYPEVET_LLAMA__TIMEOUT"):
        load_llama_settings({"TYPEVET_LLAMA__TIMEOUT": "0"})


@pytest.mark.unit
def test_llama_settings_frozen() -> None:
    settings = LlamaSettings()
    with pytest.raises(FrozenInstanceError):
        cast(Any, settings).timeout = 1.0


@pytest.mark.unit
def test_llama_cpp_adapter_uses_explicit_constructor_args() -> None:
    settings = LlamaSettings(base_url="http://127.0.0.1:7000", timeout=42.0)
    adapter = llama_cpp_adapter(settings)
    assert isinstance(adapter, LlamaCppGenerationAdapter)
    assert adapter._base_url == "http://127.0.0.1:7000/"
    assert adapter._timeout == 42.0


_KEY = "sk-LLAMA-SENTINEL-410"
_TENANT = "acme-tenant-SENTINEL-410"


@pytest.mark.unit
@pytest.mark.parametrize(
    ("raw", "expected"),
    [(None, None), ("", None), ("   ", None), (f"  {_KEY}  ", _KEY)],
    ids=["unset", "empty", "blank", "padded"],
)
def test_load_llama_settings_reads_api_key(
    raw: str | None, expected: str | None
) -> None:
    env = {} if raw is None else {"TYPEVET_LLAMA__API_KEY": raw}
    assert load_llama_settings(env).api_key == expected


@pytest.mark.unit
def test_load_llama_settings_rejects_non_ascii_api_key_by_name() -> None:
    key = "sk-clé-410"
    with pytest.raises(ValueError, match="TYPEVET_LLAMA__API_KEY") as info:
        load_llama_settings({"TYPEVET_LLAMA__API_KEY": key})
    assert key not in str(info.value)
    assert info.value.__cause__ is None
    assert info.value.__context__ is None


@pytest.mark.unit
def test_load_llama_settings_reads_headers_read_only() -> None:
    raw = json.dumps({"X-Tenant": _TENANT})
    settings = load_llama_settings({"TYPEVET_LLAMA__HEADERS": raw})
    assert dict(settings.headers) == {"X-Tenant": _TENANT}
    with pytest.raises(TypeError):
        cast(Any, settings.headers)["X-Other"] = "1"
    assert dict(load_llama_settings({}).headers) == {}


@pytest.mark.unit
@pytest.mark.parametrize(
    ("raw", "field"),
    [
        ("not json", "TYPEVET_LLAMA__HEADERS"),
        ('{"X-Tenant": 1}', "TYPEVET_LLAMA__HEADERS"),
        ('{"Host": "evil"}', "TYPEVET_LLAMA__HEADERS"),
        ('{"Authorization": "Bearer x"}', "TYPEVET_LLAMA__HEADERS"),
    ],
    ids=["not-json", "not-string", "protected", "auth-header"],
)
def test_load_llama_settings_rejects_bad_headers_by_name(raw: str, field: str) -> None:
    with pytest.raises(ValueError, match=field) as info:
        load_llama_settings({"TYPEVET_LLAMA__HEADERS": raw})
    assert "evil" not in str(info.value)
    assert "Bearer x" not in str(info.value)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("raw", "expected"),
    [(None, "Authorization"), ("", "Authorization"), ("  X-Api-Key ", "X-Api-Key")],
    ids=["unset", "blank", "custom"],
)
def test_load_llama_settings_reads_auth_header(raw: str | None, expected: str) -> None:
    env = {} if raw is None else {"TYPEVET_LLAMA__AUTH_HEADER": raw}
    assert load_llama_settings(env).auth_header == expected


@pytest.mark.unit
def test_load_llama_settings_rejects_protected_auth_header_by_name() -> None:
    with pytest.raises(ValueError, match="TYPEVET_LLAMA__AUTH_HEADER"):
        load_llama_settings({"TYPEVET_LLAMA__AUTH_HEADER": "Host"})


@pytest.mark.unit
@pytest.mark.parametrize(
    ("raw", "expected"),
    [(None, "Bearer"), ("", ""), ("  ", ""), (" Token ", "Token")],
    ids=["unset", "empty", "blank", "custom"],
)
def test_load_llama_settings_reads_auth_scheme(raw: str | None, expected: str) -> None:
    env = {} if raw is None else {"TYPEVET_LLAMA__AUTH_SCHEME": raw}
    assert load_llama_settings(env).auth_scheme == expected


@pytest.mark.unit
def test_load_llama_settings_rejects_bad_auth_scheme_by_name() -> None:
    with pytest.raises(ValueError, match="TYPEVET_LLAMA__AUTH_SCHEME"):
        load_llama_settings({"TYPEVET_LLAMA__AUTH_SCHEME": "Bad Scheme"})


@pytest.mark.unit
@pytest.mark.parametrize(
    ("raw", "expected"),
    [(None, None), ("", None), ("   ", None), (" typevet/1.0 ", "typevet/1.0")],
    ids=["unset", "empty", "blank", "padded"],
)
def test_load_llama_settings_reads_user_agent(
    raw: str | None, expected: str | None
) -> None:
    env = {} if raw is None else {"TYPEVET_LLAMA__USER_AGENT": raw}
    assert load_llama_settings(env).user_agent == expected


@pytest.mark.unit
def test_llama_settings_repr_omits_key_and_header_values() -> None:
    settings = load_llama_settings(
        {
            "TYPEVET_LLAMA__API_KEY": _KEY,
            "TYPEVET_LLAMA__HEADERS": json.dumps({"X-Tenant": _TENANT}),
        }
    )
    assert settings.api_key == _KEY
    for text in (repr(settings), str(settings)):
        assert _KEY not in text
        assert _TENANT not in text


@pytest.mark.unit
def test_llama_settings_rejects_non_mapping_headers_by_name() -> None:
    with pytest.raises(TypeError, match="TYPEVET_LLAMA__HEADERS"):
        LlamaSettings(headers=cast(Any, [("X-Tenant", _TENANT)]))
