"""Unit tests for stderr structlog diagnostics."""

from __future__ import annotations

import io
import json
from dataclasses import FrozenInstanceError
from typing import Any, cast

import pytest
import structlog

from typevet.adapters import diagnostics as diagnostics_pkg
from typevet.adapters.diagnostics.fields import diagnostic_model, filter_event_fields
from typevet.adapters.diagnostics.generation_events import generation_call_event
from typevet.adapters.diagnostics.http_events import http_request_event
from typevet.adapters.diagnostics.logs import (
    bind_run_id,
    configure,
    new_run_id,
    wants_json,
)
from typevet.adapters.diagnostics.redaction import (
    PROMPT_KEYS,
    REDACTED,
    SECRET_KEYS,
    make_redact_processor,
)
from typevet.adapters.diagnostics.settings import LogSettings, load_log_settings


@pytest.mark.unit
def test_diagnostics_public_exports() -> None:
    assert "configure" in diagnostics_pkg.__all__


@pytest.mark.unit
def test_load_log_settings_defaults() -> None:
    settings = load_log_settings({})
    assert settings == LogSettings()


@pytest.mark.unit
def test_load_log_settings_env() -> None:
    settings = load_log_settings(
        {
            "TYPEVET_LOG__FORMAT": "json",
            "TYPEVET_LOG__LEVEL": "debug",
            "TYPEVET_LOG__LOG_PROMPTS": "true",
        }
    )
    assert settings.format == "json"
    assert settings.level == "debug"
    assert settings.log_prompts is True


@pytest.mark.unit
def test_log_settings_frozen() -> None:
    settings = LogSettings()
    with pytest.raises(FrozenInstanceError):
        cast(Any, settings).level = "debug"


@pytest.mark.unit
@pytest.mark.parametrize("key", sorted(SECRET_KEYS))
def test_redact_secret_keys(key: str) -> None:
    redact = make_redact_processor(log_prompts=False)
    result = redact(None, "info", {key: "secret"})
    assert result[key] == REDACTED


@pytest.mark.unit
@pytest.mark.parametrize("key", sorted(PROMPT_KEYS))
def test_redact_prompt_keys_by_default(key: str) -> None:
    redact = make_redact_processor(log_prompts=False)
    result = redact(None, "info", {key: "user text"})
    assert result[key] == REDACTED


@pytest.mark.unit
def test_redact_prompt_keys_when_enabled() -> None:
    redact = make_redact_processor(log_prompts=True)
    result = redact(None, "info", {"prompt": "hello"})
    assert result["prompt"] == "hello"


@pytest.mark.unit
def test_redact_pem_string() -> None:
    redact = make_redact_processor(log_prompts=False)
    pem = "-----BEGIN PRIVATE KEY-----\nabc"
    result = redact(None, "info", {"note": pem})
    assert result["note"] == REDACTED


@pytest.mark.unit
def test_filter_event_fields_strips_prompt_from_builtin() -> None:
    payload = {
        "event": "http.request",
        "method": "POST",
        "path": "v1/chat/completions",
        "model": "fake",
        "status_code": 200,
        "outcome": "success",
        "error_type": None,
        "run_id": "abc123",
        "prompt": "must not appear",
    }
    filtered = filter_event_fields(None, "debug", payload)
    assert "prompt" not in filtered
    assert filtered["outcome"] == "success"


@pytest.mark.unit
def test_diagnostic_model_safe_and_unsafe() -> None:
    assert diagnostic_model("gemma-4-test") == "gemma-4-test"
    assert diagnostic_model("x" * 80) is None


@pytest.mark.unit
def test_new_run_id_shape() -> None:
    run_id = new_run_id()
    assert len(run_id) == 12
    int(run_id, 16)


@pytest.mark.unit
def test_unconfigured_http_event_is_silent(capsys: pytest.CaptureFixture[str]) -> None:
    previous = structlog.get_config()
    was_configured = structlog.is_configured()
    structlog.reset_defaults()
    try:
        with http_request_event(model="fake") as event:
            event.status_code = 200
            event.outcome = "success"
    finally:
        if was_configured:
            structlog.configure(**previous)
    assert capsys.readouterr().out == ""
    assert capsys.readouterr().err == ""


@pytest.mark.unit
def test_configured_http_event_json_to_stderr() -> None:
    captured = io.StringIO()
    configure(LogSettings(format="json", level="debug"), stream=captured)
    bind_run_id(new_run_id())
    with http_request_event(model="fake-model") as event:
        event.status_code = 200
        event.outcome = "success"
    line = captured.getvalue().strip()
    payload = json.loads(line)
    assert payload["event"] == "http.request"
    assert payload["model"] == "fake-model"
    assert payload["status_code"] == 200
    assert "prompt" not in payload


@pytest.mark.unit
def test_generation_event_records_error_type() -> None:
    captured = io.StringIO()
    configure(LogSettings(format="json", level="debug"), stream=captured)
    with pytest.raises(ValueError, match="boom"), generation_call_event(model="fake"):
        raise ValueError("boom")
    payload = json.loads(captured.getvalue().strip())
    assert payload["event"] == "generation.call"
    assert payload["error_type"] == "ValueError"
    assert payload["outcome"] == "error"


@pytest.mark.unit
def test_wants_json_respects_format() -> None:
    stream = io.StringIO()
    assert wants_json(LogSettings(format="json"), stream) is True
    assert wants_json(LogSettings(format="console"), stream) is False
