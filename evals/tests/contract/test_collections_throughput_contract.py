"""Contract test: the throughput runner over the real vLLM judgment path ([#236][i236]).

``httpx.MockTransport`` stands behind the real ``open_judgment`` at level 64.
The fake holds each scoring call until 64 calls are in flight, so the peak
in-flight count shows the pool width. One record fails with a reply that
echoes the ``Authorization`` header, so an unmasked key would reach the
receipt.

Examples:
    ```bash
    uv run pytest -q evals/tests/contract/test_collections_throughput_contract.py
    ```

See Also:
    - [typevet_evals.throughput.collections_throughput][]: the runner

[i236]: https://github.com/Alberto-Codes/typevet/issues/236
"""

from __future__ import annotations

import json

import httpx
import pytest

from evals.tests.unit.test_collections_throughput import (
    AGENT,
    KEY,
    QUESTIONS,
    FakeVllm,
    env,
    records,
)
from evals.tests.unit.test_server_args import CACHE_CONFIG
from typevet_evals.throughput.collections_throughput import run_throughput


@pytest.mark.contract
def test_level_64_peaks_at_64_in_flight_masks_key_and_sends_user_agent() -> None:
    server = FakeVllm(hold=64, echo_key=True)

    receipt = run_throughput(
        env(),
        records(128, failing=1),
        QUESTIONS,
        transport=httpx.MockTransport(server),
        levels=(64,),
    )

    (level,) = receipt["levels"]
    assert server.peak == 64
    assert level["stopped"] is None
    assert level["errors"] == 1
    assert level["answered"] == 127
    text = json.dumps(receipt)
    assert KEY not in text
    assert "boom" in text
    assert server.requests
    assert all(r.headers.get("user-agent") == AGENT for r in server.requests)
    assert receipt["pins"]["user_agent"] == AGENT
    assert receipt["server_args"] == {
        "cache_config": CACHE_CONFIG,
        "cache_config_source": "metrics",
        "caller_stated": None,
    }
