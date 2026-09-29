"""Pytest helpers for opt-in live tests (#98, #191)."""

from __future__ import annotations

import pytest

from typevet.adapters.inbound.settings import LlamaSettings
from typevet_evals.runner.live_gate import (
    LiveGateAction,
    live_gate_action,
    live_skip_reason,
)


def gate_live(settings: LlamaSettings) -> None:
    """Skip, fail, or continue based on ``live_skip_reason`` and env."""
    reason = live_skip_reason(settings)
    action = live_gate_action(reason)
    if action is LiveGateAction.RUN:
        return
    assert reason is not None
    if action is LiveGateAction.FAIL:
        pytest.fail(reason)
    pytest.skip(reason)
