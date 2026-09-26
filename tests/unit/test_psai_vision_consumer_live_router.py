"""Unit tests: consumer live router with mocked HTTP ([#177][i177]).

Examples:
    ```bash
    uv run pytest -q tests/unit/test_psai_vision_consumer_live_router.py
    ```

See Also:
    - [typevet.evaluation.psai_vision_consumer_live_router][]: router leg
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from typevet.adapters.inbound.settings import LlamaSettings
from typevet.adapters.outbound.gemma import ServedTemplateClass
from typevet.adapters.outbound.gemma_native_vision_factory import (
    GemmaNativeVisionSession,
)
from typevet.adapters.outbound.llama_cpp_multimodal import MediaCapability
from typevet.evaluation.psai_vision_consumer_live_router import run_consumer_live_matrix

FIXTURE_ROOT = (
    Path(__file__).resolve().parents[1] / "fixtures" / "psai" / "vision_smoke"
)


@pytest.mark.unit
def test_run_consumer_live_matrix_offline_scripted_router() -> None:
    """Matrix dispatch uses ledger-wrapped ports under mocked router probes."""
    health = {"status": "ok", "version": "test-build"}
    mock_client = MagicMock()
    mock_client.get.return_value.raise_for_status.return_value.json.return_value = (
        health
    )
    mock_client.post.return_value.raise_for_status.return_value.json.side_effect = [
        {"prompt": "<start_of_turn>user\nhello<end_of_turn>\n<start_of_turn>model\n"},
        {"tokens": [1, 2, 3]},
    ]

    capability = MediaCapability(vision=True, marker="m")
    settings = LlamaSettings(base_url="http://127.0.0.1:8090", timeout=30.0)

    with (
        patch(
            "typevet.evaluation.psai_vision_consumer_live_router.httpx.Client",
        ) as client_cls,
        patch(
            "typevet.evaluation.psai_vision_consumer_live_router.open_gemma_native_vision_judgment",
        ) as factory_ctx,
        patch(
            "typevet.evaluation.psai_vision_consumer_live_router.run_offline_consumer_matrix",
            return_value=([], {}),
        ),
    ):
        client_cls.return_value.__enter__.return_value = mock_client
        mock_port = MagicMock()
        factory_ctx.return_value.__enter__.return_value = GemmaNativeVisionSession(
            port=mock_port,
            client=mock_client,
            model="gemma-test",
            served=ServedTemplateClass.NATIVE_GEMMA4_TURN,
            capability=capability,
        )
        result = run_consumer_live_matrix(
            settings=settings,
            model="gemma-test",
            fixture_root=FIXTURE_ROOT,
        )

    assert result.served is ServedTemplateClass.NATIVE_GEMMA4_TURN
    assert result.ledger.judgment_calls == 0
    assert "runtime" in result.identity
