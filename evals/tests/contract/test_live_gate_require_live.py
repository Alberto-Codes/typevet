"""Contract tests for require-live gate behaviour (#191)."""

from __future__ import annotations

import pytest

from typevet_evals.runner.live_gate import (
    TYPEVET_REQUIRE_LIVE_ENV,
    LiveGateAction,
    live_gate_action,
    require_live_enabled,
)


@pytest.mark.contract
def test_live_gate_exports_require_live_surface() -> None:
    assert TYPEVET_REQUIRE_LIVE_ENV == "TYPEVET_REQUIRE_LIVE"
    assert callable(require_live_enabled)
    assert callable(live_gate_action)


@pytest.mark.contract
def test_skip_reason_becomes_fail_under_require_live(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TYPEVET_REQUIRE_LIVE", "1")
    reason = "llama.cpp router not reachable"
    assert live_gate_action(reason) is LiveGateAction.FAIL
    assert live_gate_action(None) is LiveGateAction.RUN


@pytest.mark.contract
def test_skip_reason_stays_skip_without_require_live(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TYPEVET_REQUIRE_LIVE", raising=False)
    assert live_gate_action("llama.cpp router not reachable") is LiveGateAction.SKIP
