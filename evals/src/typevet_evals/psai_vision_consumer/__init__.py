"""PSAI vision consumer proof harness ([#177][i177], moved in #256 E5).

The harness runs the frozen PSAI vision consumer matrix over the committed
``vision_smoke`` fixture, through the offline scripted port or a live
llama.cpp server. It counts dispatched calls, writes a versioned receipt and
applies fail-closed acceptance. This package re-exports the names that
callers outside the package use.

Attributes:
    __all__ (list[str]): Public harness names re-exported from the submodules.

Examples:
    ```python
    from pathlib import Path

    from typevet_evals.psai_vision_consumer import run_offline_consumer_proof

    result = run_offline_consumer_proof(
        fixture_root=Path("tests/fixtures/psai/vision_smoke"),
    )
    assert result.exit_code in {0, 1}
    ```

See Also:
    - [typevet_evals.psai_vision_consumer.harness][]: offline proof and CLI
    - [typevet_evals.psai_vision_consumer.live][]: live orchestration
    - [typevet_evals.psai_vision_consumer.accounting][]: call budgets
    - [typevet_evals.psai_vision_consumer.dispatch][]: dispatch ledger
    - [typevet_evals.psai_vision_consumer.receipt][]: receipt acceptance

[i177]: https://github.com/Alberto-Codes/typevet/issues/177
"""

from __future__ import annotations

from typevet_evals.psai_vision_consumer.accounting import (
    ConsumerCallBudgetError,
    ConsumerCallCounts,
    enforce_consumer_call_budget,
)
from typevet_evals.psai_vision_consumer.dispatch import (
    ConsumerDispatchLedger,
    wrap_scoring_port,
)
from typevet_evals.psai_vision_consumer.harness import (
    consumer_proof_main,
    run_offline_consumer_proof,
)
from typevet_evals.psai_vision_consumer.http_accounting import DispatchAccounting
from typevet_evals.psai_vision_consumer.live import (
    live_consumer_proof_main,
    run_live_consumer_proof,
)
from typevet_evals.psai_vision_consumer.live_router import run_consumer_live_matrix
from typevet_evals.psai_vision_consumer.offline import (
    SequentialConsumerScoringFake,
    consumer_fixture_identity_pins,
    load_frozen_consumer_fixture,
    run_negative_template_probe,
)
from typevet_evals.psai_vision_consumer.receipt import serialize_answer

__all__ = [
    "ConsumerCallBudgetError",
    "ConsumerCallCounts",
    "ConsumerDispatchLedger",
    "DispatchAccounting",
    "SequentialConsumerScoringFake",
    "consumer_fixture_identity_pins",
    "consumer_proof_main",
    "enforce_consumer_call_budget",
    "live_consumer_proof_main",
    "load_frozen_consumer_fixture",
    "run_consumer_live_matrix",
    "run_live_consumer_proof",
    "run_negative_template_probe",
    "run_offline_consumer_proof",
    "serialize_answer",
    "wrap_scoring_port",
]
