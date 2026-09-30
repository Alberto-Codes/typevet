r"""Opt-in paid vLLM collections throughput run ([#236][i236]).

Skips unless ``TYPEVET_REQUIRE_LIVE=1``. With it set, **fails** before any
network call when ``TYPEVET_BACKEND=vllm``, the ``TYPEVET_VLLM__*`` settings,
``TYPEVET_VLLM_RECEIPT``, ``TYPEVET_FINVET_JEV_DIR`` or
``TYPEVET_FINVET_QUESTIONS`` are missing or invalid. The run sweeps levels 1,
8, 32 and 64 on ``val``, then runs train, val and test once at the best level
with the call caps and the part of the 2,700 s run budget that the sweep
left. One receipt with both runs is written
with the key masked. A failing result is a valid result.

Examples:
    ```bash
    TYPEVET_REQUIRE_LIVE=1 TYPEVET_BACKEND=vllm \
      TYPEVET_VLLM__BASE_URL=https://<pod>-8000.proxy.runpod.net \
      TYPEVET_VLLM__MODEL=google/gemma-4-31B-it \
      TYPEVET_VLLM_RECEIPT=scratchpad/vllm/236-receipt.json \
      TYPEVET_FINVET_JEV_DIR=<finvet JEV split directory> \
      TYPEVET_FINVET_QUESTIONS=<seed>.json \
      uv run pytest evals/tests/live/test_collections_throughput_live.py -m live -q -s
    ```

See Also:
    - [typevet_evals.throughput.collections_throughput][]: the runner

[i236]: https://github.com/Alberto-Codes/typevet/issues/236
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import httpx
import pytest

from typevet.adapters.inbound.backend_settings import load_vllm_settings
from typevet_evals.runner.live_gate import LiveGateAction, live_gate_action
from typevet_evals.throughput.collections_throughput import (
    CAPS,
    RunOptions,
    remaining_run_seconds,
    run_throughput,
)
from typevet_evals.throughput.collections_workload import (
    load_questions,
    load_records,
    workload_paths,
)
from typevet_evals.vllm_acceptance.core import (
    CallCaps,
    live_gate_reason,
    write_receipt,
)


def _gate_reason(environ: Mapping[str, str]) -> str | None:
    reason = live_gate_reason(environ)
    if reason is not None:
        return reason
    try:
        paths = workload_paths(environ)
    except ValueError as exc:
        return str(exc)
    missing = [p.name for p in paths if not p.is_file()]
    return f"finvet files not found: {missing}" if missing else None


def _left(used: Mapping[str, int]) -> CallCaps:
    return CallCaps(
        model=CAPS.model - used["model"],
        tokenizer=CAPS.tokenizer - used["tokenizer"],
        metadata=CAPS.metadata - used["metadata"],
    )


@pytest.mark.live
def test_collections_throughput_run() -> None:
    """Sweep the levels on val, run all splits at the best level, write one receipt."""
    reason = _gate_reason(os.environ)
    if reason is not None:
        if live_gate_action(reason) is LiveGateAction.FAIL:
            pytest.fail(reason)
        pytest.skip(reason)
    val_path, seed_path = workload_paths(os.environ)
    questions = load_questions(seed_path)
    receipt: dict[str, Any] = {}
    with httpx.HTTPTransport() as transport:
        sweep = run_throughput(
            os.environ, load_records(val_path), questions, transport=transport
        )
        receipt["sweep"] = sweep
        best = sweep["best_level"]
        if isinstance(best, int):
            every = [
                row
                for split in ("train", "val", "test")
                for row in load_records(workload_paths(os.environ, split)[0])
            ]
            receipt["audit"] = run_throughput(
                os.environ,
                every,
                questions,
                transport=transport,
                levels=(best,),
                caps=_left(sweep["calls"]),
                options=RunOptions(run_seconds=remaining_run_seconds(sweep)),
            )
    path = Path(os.environ["TYPEVET_VLLM_RECEIPT"])
    digest = write_receipt(path, receipt, settings=load_vllm_settings(os.environ))
    print(f"receipt {path} sha256 {digest} best level {best}")
    assert "audit" in receipt, f"no usable level: {sweep['stopped']}"
    assert receipt["audit"]["stopped"] is None, receipt["audit"]["stopped"]
