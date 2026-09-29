r"""Opt-in paid vLLM live acceptance run (#170).

Skips unless ``TYPEVET_REQUIRE_LIVE=1``. With it set, **fails** before any
network call when ``TYPEVET_BACKEND=vllm``, the ``TYPEVET_VLLM__*`` settings
or ``TYPEVET_VLLM_RECEIPT`` are missing or invalid. The run executes the five
pre-registered sets once through ``run_acceptance``, which writes one receipt
that ``typevet_evals.cli.cord_semantic_acceptance`` can read, also when the
run raises. The test prints the receipt sha256 and fails when a stop rule
fired or a gated set failed. A failing result is a valid result. The receipt
path must not exist.

Examples:
    ```bash
    TYPEVET_REQUIRE_LIVE=1 TYPEVET_BACKEND=vllm \
      TYPEVET_VLLM__BASE_URL=https://<pod>-8000.proxy.runpod.net \
      TYPEVET_VLLM__MODEL=google/gemma-4-31B-it \
      TYPEVET_VLLM_RECEIPT=scratchpad/vllm/170-receipt.json \
      uv run pytest evals/tests/live/test_vllm_acceptance_live.py -m live -q -s
    ```

See Also:
    - [typevet_evals.vllm_acceptance.core][]: caps, gates and receipt
    - [typevet_evals.vllm_acceptance.sets][]: the five set runners
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import httpx
import pytest

from typevet_evals.runner.live_gate import LiveGateAction, live_gate_action
from typevet_evals.vllm_acceptance.core import (
    AcceptanceInputs,
    live_gate_reason,
    run_acceptance,
)
from typevet_evals.vllm_acceptance.sets import DEVIATIONS, SET_RUNNERS

_REPO_ROOT = Path(__file__).resolve().parents[3]
_FIXTURES = _REPO_ROOT / "tests" / "fixtures"


@pytest.mark.live
def test_vllm_acceptance_run() -> None:
    """Run the five sets once and write the receipt."""
    reason = live_gate_reason(os.environ)
    if reason is not None:
        if live_gate_action(reason) is LiveGateAction.FAIL:
            pytest.fail(reason)
        pytest.skip(reason)
    path = Path(os.environ["TYPEVET_VLLM_RECEIPT"])
    inputs = AcceptanceInputs(fixtures_root=_FIXTURES, repo_root=_REPO_ROOT)
    with httpx.HTTPTransport() as transport:
        receipt = run_acceptance(
            os.environ,
            inputs,
            transport=transport,
            runners=SET_RUNNERS,
            deviations=DEVIATIONS,
            receipt_path=path,
        )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    print(f"receipt {path} sha256 {digest} calls {receipt['calls']}")
    assert receipt["stopped"] is None, receipt["stopped"]
    failed = [
        name
        for name in ("generation", "psai", "cord")
        if receipt["sets"][name].get("passed") is not True
    ]
    assert receipt["passed"], f"gated sets failed: {failed}"
