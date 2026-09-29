"""Contract test: a public workload over the real vLLM judgment path ([#236][i236]).

``httpx.MockTransport`` stands behind the real ``open_judgment``. The runner
gets Banking77 text records and the ``reports_unauthorized`` Noul name. Each
scoring request must carry the text state and the Noul instructions, and each
answered record must read its probability from that Noul.

Examples:
    ```bash
    uv run pytest -q evals/tests/contract/test_public_workload_contract.py
    ```

See Also:
    - [typevet_evals.throughput.public_workload][]: the workload builders

[i236]: https://github.com/Alberto-Codes/typevet/issues/236
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from evals.tests.unit.test_collections_throughput import FakeVllm, env
from typevet.evaluation.datasets import banking77
from typevet_evals.throughput.collections_throughput import run_throughput
from typevet_evals.throughput.public_workload import (
    BANKING77_BASELINE,
    banking77_workload,
)

_CSV = (
    Path(__file__).resolve().parents[3]
    / "tests"
    / "fixtures"
    / "banking77"
    / "test_subset.csv"
)


@pytest.mark.contract
def test_runner_sends_text_state_and_the_requested_noul() -> None:
    work = banking77_workload(
        banking77.load_test_split(csv_text=_CSV.read_text(encoding="utf-8"))
    )
    server = FakeVllm()

    receipt = run_throughput(
        env(),
        work.records,
        work.questions,
        transport=httpx.MockTransport(server),
        levels=(1,),
        noul=work.noul,
        baseline=work.baseline,
    )

    (level,) = receipt["levels"]
    assert level["answered"] == len(work.records)
    assert level["parity"]["baseline_ece"] == BANKING77_BASELINE.ece
    chats = [
        r.content.decode()
        for r in server.requests
        if r.url.path.endswith("/chat/completions")
    ]
    assert chats
    sent = "\n".join(json.dumps(json.loads(c)) for c in chats)
    for record in work.records:
        assert json.dumps(record.state)[1:-1] in sent
    instructions = work.questions[work.noul].instructions
    assert isinstance(instructions, str)
    assert json.dumps(instructions)[1:-1] in sent
