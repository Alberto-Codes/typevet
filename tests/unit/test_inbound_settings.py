"""Unit tests for llama.cpp composition-root settings."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from typing import Any, cast

import pytest

from typevet.adapters.inbound.settings import (
    LlamaSettings,
    llama_cpp_adapter,
    load_llama_settings,
)
from typevet.adapters.outbound.llama_cpp import LlamaCppGenerationAdapter


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
