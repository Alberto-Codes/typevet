"""Unit tests for llama.cpp router catalog parsing in live_skip_reason (#127)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from typevet.adapters.inbound.settings import LlamaSettings
from typevet_evals.runner.live_gate import (
    LiveGateAction,
    live_gate_action,
    live_skip_reason,
    require_live_enabled,
)

_SETTINGS = LlamaSettings(base_url="http://127.0.0.1:8090", default_model="gemma")


def _mock_models_response(payload: object) -> MagicMock:
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = payload
    return response


@pytest.mark.unit
@pytest.mark.parametrize(
    ("payload", "expected_reason"),
    [
        ([], "llama.cpp router catalog invalid"),
        ({"data": None}, "llama.cpp router catalog invalid"),
        ({"data": [None]}, "llama.cpp router catalog invalid"),
        ({"data": ["not-a-mapping"]}, "llama.cpp router catalog invalid"),
        ({"data": [{}]}, "llama.cpp router catalog invalid"),
        ({"data": [{"id": 1}]}, "llama.cpp router catalog invalid"),
    ],
    ids=[
        "non-object-root",
        "data-null",
        "data-contains-null",
        "data-row-not-mapping",
        "data-row-missing-id",
        "data-row-non-string-id",
    ],
)
def test_live_skip_malformed_catalog_returns_invalid_reason(
    payload: object, expected_reason: str
) -> None:
    with patch(
        "typevet_evals.runner.live_gate.httpx.get",
        return_value=_mock_models_response(payload),
    ):
        reason = live_skip_reason(_SETTINGS)
    assert reason == expected_reason


@pytest.mark.unit
def test_live_skip_empty_catalog() -> None:
    with patch(
        "typevet_evals.runner.live_gate.httpx.get",
        return_value=_mock_models_response({"data": []}),
    ):
        reason = live_skip_reason(_SETTINGS)
    assert reason == "llama.cpp router catalog empty"


@pytest.mark.unit
def test_live_skip_missing_model_in_nonempty_catalog() -> None:
    with patch(
        "typevet_evals.runner.live_gate.httpx.get",
        return_value=_mock_models_response({"data": [{"id": "other"}]}),
    ):
        reason = live_skip_reason(
            LlamaSettings(base_url="http://127.0.0.1:8090", default_model="missing")
        )
    assert reason == "missing not in router catalog"


@pytest.mark.unit
def test_live_skip_none_when_model_listed() -> None:
    with patch(
        "typevet_evals.runner.live_gate.httpx.get",
        return_value=_mock_models_response({"data": [{"id": "gemma"}]}),
    ):
        assert live_skip_reason(_SETTINGS) is None


@pytest.mark.unit
def test_live_skip_fetches_models_once() -> None:
    with patch(
        "typevet_evals.runner.live_gate.httpx.get",
        return_value=_mock_models_response({"data": [{"id": "gemma"}]}),
    ) as mock_get:
        assert live_skip_reason(_SETTINGS) is None
    assert mock_get.call_count == 1


@pytest.mark.unit
def test_live_gate_action_skips_when_reason_and_flag_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TYPEVET_REQUIRE_LIVE", raising=False)
    assert not require_live_enabled()
    assert live_gate_action("llama.cpp router not reachable") is LiveGateAction.SKIP
    assert live_gate_action(None) is LiveGateAction.RUN


@pytest.mark.unit
def test_live_gate_action_fails_when_require_live_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TYPEVET_REQUIRE_LIVE", "1")
    assert require_live_enabled()
    reason = "TYPEVET_LLAMA__DEFAULT_MODEL (or TYPEVET_GEMMA_MODEL) not set"
    assert live_gate_action(reason) is LiveGateAction.FAIL


@pytest.mark.unit
def test_live_skip_router_unreachable() -> None:
    settings = LlamaSettings(base_url="http://127.0.0.1:1", default_model="gemma")
    with patch(
        "typevet_evals.runner.live_gate.httpx.get",
        side_effect=httpx.HTTPError("down"),
    ):
        assert live_skip_reason(settings) == "llama.cpp router not reachable"
