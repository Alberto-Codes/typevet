"""Unit tests: live consumer proof orchestration ([#177][i177]).

Examples:
    ```bash
    uv run pytest -q tests/unit/test_psai_vision_consumer_live.py
    ```

See Also:
    - [typevet.evaluation.psai_vision_consumer_live][]: live orchestration
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from typevet.evaluation.datasets.psai_vision_controls import paired_image_ordering
from typevet.evaluation.psai_vision_consumer_harness import run_offline_consumer_proof
from typevet.evaluation.psai_vision_consumer_live import run_live_consumer_proof
from typevet.evaluation.psai_vision_consumer_offline import (
    frozen_consumer_controls,
    load_frozen_consumer_fixture,
)

FIXTURE_ROOT = (
    Path(__file__).resolve().parents[1] / "fixtures" / "psai" / "vision_smoke"
)


@pytest.mark.unit
def test_run_live_consumer_proof_requires_live_env() -> None:
    """Live proof raises when ``TYPEVET_REQUIRE_LIVE`` is unset."""
    with (
        patch(
            "typevet.evaluation.psai_vision_consumer_live.require_live_enabled",
            return_value=False,
        ),
        pytest.raises(ValueError, match="TYPEVET_REQUIRE_LIVE"),
    ):
        run_live_consumer_proof(fixture_root=FIXTURE_ROOT)


@pytest.mark.unit
def test_run_live_consumer_proof_success_path() -> None:
    """Successful live proof delegates matrix dispatch and accepts offline-shaped rows."""
    offline = run_offline_consumer_proof(fixture_root=FIXTURE_ROOT)
    fixture = load_frozen_consumer_fixture(FIXTURE_ROOT)
    controls = frozen_consumer_controls(fixture)
    probabilities = {
        (row["unique_data_id"], row["condition"]): float(
            row["answers"]["shows_fox_news_chrome"]["noul"]
        )
        for row in offline.receipt["matrix_rows"]
        if row.get("leg") == "visual"
    }
    pairs = paired_image_ordering(controls, probabilities)
    mock_matrix = type(
        "M",
        (),
        {
            "matrix_rows": offline.receipt["matrix_rows"],
            "probabilities": {},
            "health": {"status": "ok"},
            "capability": type("C", (), {"vision": True, "marker": "m"})(),
            "served": type("S", (), {"name": "NATIVE_GEMMA4_TURN"})(),
            "ledger": type(
                "L",
                (),
                {
                    "scoring_requests": 16,
                    "judgment_calls": 14,
                    "auxiliary_http_total": 4,
                    "auxiliary_metadata_http": 3,
                    "auxiliary_tokenizer_http": 1,
                    "failed_attempts": 0,
                },
            )(),
            "identity": {"run_id": "unit"},
            "elapsed_s": 0.1,
        },
    )()

    with (
        patch(
            "typevet.evaluation.psai_vision_consumer_live.require_live_enabled",
            return_value=True,
        ),
        patch(
            "typevet.evaluation.psai_vision_consumer_live.run_consumer_live_matrix",
            return_value=mock_matrix,
        ),
        patch(
            "typevet.evaluation.psai_vision_consumer_live.load_llama_settings",
        ) as settings,
    ):
        settings.return_value.base_url = "http://127.0.0.1:8090"
        settings.return_value.timeout = 30.0
        settings.return_value.multimodal_model = "gemma-test"
        settings.return_value.default_model = None
        with (
            patch(
                "typevet.evaluation.psai_vision_consumer_live.live_skip_reason",
                return_value=None,
            ),
            patch(
                "typevet.evaluation.psai_vision_consumer_live.paired_image_ordering",
                return_value=pairs,
            ),
        ):
            result = run_live_consumer_proof(fixture_root=FIXTURE_ROOT)

    assert result.exit_code == 0
    assert result.receipt["scoring_requests_observed"] == 16
