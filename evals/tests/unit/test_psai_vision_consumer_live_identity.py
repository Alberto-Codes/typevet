"""Unit tests: consumer live identity snapshot ([#177][i177]).

Examples:
    ```bash
    uv run pytest -q evals/tests/unit/test_psai_vision_consumer_live_identity.py
    ```

See Also:
    - [typevet_evals.psai_vision_consumer.live_identity][]: snapshot helpers
"""

from __future__ import annotations

from pathlib import Path

import pytest

from typevet_evals.psai_vision_consumer.dispatch import ConsumerDispatchLedger
from typevet_evals.psai_vision_consumer.live_identity import (
    consumer_fixture_paths,
    finalize_consumer_live_identity,
    start_consumer_live_identity,
)
from typevet_evals.psai_vision_consumer.offline import load_frozen_consumer_fixture

FIXTURE_ROOT = (
    Path(__file__).resolve().parents[3] / "tests" / "fixtures" / "psai" / "vision_smoke"
)


@pytest.mark.unit
def test_consumer_fixture_paths_include_manifest_and_images() -> None:
    """Frozen case images and manifest are named for identity snapshot."""
    fixture = load_frozen_consumer_fixture(FIXTURE_ROOT)
    paths = consumer_fixture_paths(FIXTURE_ROOT, fixture)
    assert "manifest" in paths
    assert any(key.startswith("image:") for key in paths)


@pytest.mark.unit
def test_start_and_finalize_consumer_live_identity() -> None:
    """Pre-dispatch snapshot finalizes with dispatch call counts."""
    fixture = load_frozen_consumer_fixture(FIXTURE_ROOT)
    run_start, evaluated = start_consumer_live_identity(
        fixture_root=FIXTURE_ROOT,
        fixture=fixture,
        model="offline",
        served_template="NATIVE_GEMMA4_TURN",
        health={"status": "ok"},
    )
    ledger = ConsumerDispatchLedger()
    ledger.judgment_calls = 14
    ledger.scoring_requests = 16
    identity = finalize_consumer_live_identity(
        run_start=run_start,
        evaluated=evaluated,
        ledger=ledger,
    )
    runtime = identity.get("runtime")
    assert isinstance(runtime, dict)
    assert runtime.get("model") == "offline"
    arm_counts = identity.get("arm_call_counts")
    assert isinstance(arm_counts, dict)
    assert arm_counts.get("scoring") == 16
