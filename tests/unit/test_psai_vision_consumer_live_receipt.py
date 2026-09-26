"""Unit tests: live consumer receipt payload builder ([#177][i177]).

Examples:
    ```bash
    uv run pytest -q tests/unit/test_psai_vision_consumer_live_receipt.py
    ```

See Also:
    - [typevet.evaluation.psai_vision_consumer_live_receipt][]: payload builder
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from typevet.adapters.outbound.gemma import ServedTemplateClass
from typevet.evaluation.psai_vision_consumer_accounting import (
    plan_frozen_consumer_calls,
)
from typevet.evaluation.psai_vision_consumer_dispatch import ConsumerDispatchLedger
from typevet.evaluation.psai_vision_consumer_harness import run_offline_consumer_proof
from typevet.evaluation.psai_vision_consumer_live_receipt import (
    LiveReceiptContext,
    build_live_receipt_payload,
)
from typevet.evaluation.psai_vision_consumer_live_router import ConsumerLiveMatrixResult

FIXTURE_ROOT = (
    Path(__file__).resolve().parents[1] / "fixtures" / "psai" / "vision_smoke"
)


@pytest.mark.unit
def test_build_live_receipt_payload_includes_identity_and_ledger() -> None:
    """Live payload merges identity block and instrumented ledger fields."""
    offline = run_offline_consumer_proof(fixture_root=FIXTURE_ROOT)
    ledger = ConsumerDispatchLedger()
    ledger.scoring_requests = 16
    ledger.judgment_calls = 14
    ledger.before_metadata_http()
    matrix = ConsumerLiveMatrixResult(
        matrix_rows=offline.receipt["matrix_rows"],
        probabilities={},
        health={"status": "ok"},
        capability=MagicMock(vision=True, marker="m"),
        served=ServedTemplateClass.NATIVE_GEMMA4_TURN,
        ledger=ledger,
        identity={"run_id": "r1", "runtime": {"model": "m"}},
        elapsed_s=1.0,
    )
    plan = plan_frozen_consumer_calls(visual_control_rows=12)
    payload = build_live_receipt_payload(
        LiveReceiptContext(
            fixture_root=FIXTURE_ROOT,
            plan=plan,
            matrix=matrix,
            model="gemma-test",
            wheel_sha256="abc",
            typevet_install_path="unknown",
            evidence_kind="unit_test",
            negative=offline.receipt["unsupported_capability_negative"],
            pairs=(),
            git_head="0" * 40,
        )
    )
    assert payload["require_live"] is True
    assert payload["auxiliary_http_metadata_count"] == 1
    assert payload["run_id"] == "r1"
