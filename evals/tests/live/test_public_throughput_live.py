r"""Opt-in paid vLLM throughput run on the public datasets ([#236][i236]).

Skips unless ``TYPEVET_REQUIRE_LIVE=1``. With it set, **fails** before any
network call when the vLLM settings, ``TYPEVET_VLLM_RECEIPT`` or
``TYPEVET_PUBLIC_DATASET`` are missing or invalid, or when that directory
lacks the two data files. Fetch the files before the pod exists:

```bash
uv run python -c "from pathlib import Path; \
from typevet_evals.throughput.public_workload import fetch_public_data; \
print(fetch_public_data(Path('scratchpad/public-data')))"
```

The run sweeps levels 1, 8, 32 and 64 on the balanced Banking77 set. At the
best level it then runs DIFrauD SMS (parity) and the full Banking77 split
(record only), with the call caps and the run budget that remain. One receipt
with the three runs is written with the key masked. A failing result is a
valid result.

Examples:
    ```bash
    TYPEVET_REQUIRE_LIVE=1 TYPEVET_BACKEND=vllm \
      TYPEVET_VLLM__BASE_URL=https://<pod>-8000.proxy.runpod.net \
      TYPEVET_VLLM__MODEL=google/gemma-4-31B-it \
      TYPEVET_VLLM_RECEIPT=scratchpad/vllm/236-public-receipt.json \
      TYPEVET_PUBLIC_DATASET=scratchpad/public-data \
      uv run pytest evals/tests/live/test_public_throughput_live.py -m live -q -s
    ```

See Also:
    - [typevet_evals.throughput.public_workload][]: the workloads
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
from typevet_evals.datasets import banking77, difraud
from typevet_evals.runner.live_gate import LiveGateAction, live_gate_action
from typevet_evals.throughput.collections_throughput import (
    CAPS,
    RunOptions,
    remaining_run_seconds,
    run_throughput,
)
from typevet_evals.throughput.public_workload import (
    PUBLIC_DATASET_ENV,
    PublicWorkload,
    load_public_workloads,
    missing_data,
)
from typevet_evals.throughput.server_args import stated_server_args
from typevet_evals.vllm_acceptance.core import (
    CallCaps,
    live_gate_reason,
    write_receipt,
)

DATASETS: dict[str, dict[str, str]] = {
    "banking77": {
        "source": banking77.TEST_CSV_URL,
        "licence": "CC BY 4.0",
        "attribution": (
            "Banking77 by PolyAI (Casanueva et al., 2020), CC BY 4.0; "
            "attribution required"
        ),
    },
    "difraud_sms": {
        "source": difraud.TEST_JSONL_URL.format(domain="sms"),
        "licence": "MIT",
        "attribution": "DIFrauD (difraud/difraud on Hugging Face), MIT",
    },
}


def _gate_reason(environ: Mapping[str, str]) -> str | None:
    reason = live_gate_reason(environ)
    if reason is not None:
        return reason
    raw = environ.get(PUBLIC_DATASET_ENV, "").strip()
    if not raw:
        return f"{PUBLIC_DATASET_ENV} is not set"
    return missing_data(Path(raw))


def _left(receipts: list[dict[str, Any]]) -> CallCaps:
    kinds = ("model", "tokenizer", "metadata")
    used = {k: sum(r["calls"][k] for r in receipts) for k in kinds}
    return CallCaps(
        model=CAPS.model - used["model"],
        tokenizer=CAPS.tokenizer - used["tokenizer"],
        metadata=CAPS.metadata - used["metadata"],
    )


def _run(
    work: PublicWorkload,
    transport: httpx.BaseTransport,
    levels: tuple[int, ...],
    earlier: list[dict[str, Any]],
    stated: str | None,
) -> dict[str, Any]:
    spent = {"elapsed_seconds": sum(r["elapsed_seconds"] for r in earlier)}
    return run_throughput(
        os.environ,
        work.records,
        work.questions,
        transport=transport,
        levels=levels,
        caps=_left(earlier) if earlier else CAPS,
        options=RunOptions(
            run_seconds=remaining_run_seconds(spent), caller_stated=stated
        ),
        noul=work.noul,
        baseline=work.baseline,
    )


@pytest.mark.live
def test_public_throughput_run() -> None:
    """Sweep Banking77, run DIFrauD and full Banking77 at the best level."""
    reason = _gate_reason(os.environ)
    if reason is not None:
        if live_gate_action(reason) is LiveGateAction.FAIL:
            pytest.fail(reason)
        pytest.skip(reason)
    sets = load_public_workloads(Path(os.environ[PUBLIC_DATASET_ENV]))
    receipt: dict[str, Any] = {
        "datasets": DATASETS,
        "record_counts": {name: len(w.records) for name, w in sets.items()},
    }
    stated = stated_server_args(os.environ)
    with httpx.HTTPTransport() as transport:
        sweep = _run(sets["banking77_balanced"], transport, (1, 8, 32, 64), [], stated)
        receipt["banking77_balanced"] = sweep
        best = sweep["best_level"]
        if isinstance(best, int):
            done = [sweep]
            for name in ("difraud_sms", "banking77_full"):
                out = _run(sets[name], transport, (best,), done, stated)
                receipt[name] = out
                done.append(out)
    path = Path(os.environ["TYPEVET_VLLM_RECEIPT"])
    digest = write_receipt(path, receipt, settings=load_vllm_settings(os.environ))
    print(f"receipt {path} sha256 {digest} best level {best}")
    assert "difraud_sms" in receipt, f"no usable level: {sweep['stopped']}"
    assert receipt["difraud_sms"]["stopped"] is None, receipt["difraud_sms"]["stopped"]
