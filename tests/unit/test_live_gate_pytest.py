"""Unit tests for pytest live gate helper (#191)."""

from __future__ import annotations

import pytest
from _pytest.outcomes import Failed, Skipped

from tests.live.gate import gate_live
from typevet.adapters.inbound.settings import LlamaSettings


@pytest.mark.unit
def test_gate_live_skips_without_require_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TYPEVET_REQUIRE_LIVE", raising=False)
    settings = LlamaSettings(base_url="http://127.0.0.1:8090", default_model="")
    with pytest.raises(Skipped):
        gate_live(settings)


@pytest.mark.unit
def test_gate_live_fails_under_require_live(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TYPEVET_REQUIRE_LIVE", "1")
    settings = LlamaSettings(base_url="http://127.0.0.1:8090", default_model="")
    with pytest.raises(Failed, match="DEFAULT_MODEL"):
        gate_live(settings)
