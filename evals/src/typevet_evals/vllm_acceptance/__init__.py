"""The #170 vLLM acceptance run: call caps, set runners and receipt (#229).

``core`` holds the run, the gates and the receipt writer. ``sets`` holds the
five pre-registered set runners and imports ``core``. ``transport`` holds the
call caps, the counting transport and the ``/metrics`` read. ``masking`` holds
the receipt masking helpers that ``core`` uses. The first three modules moved
here from the library (#256).
Module constants stay on their modules, except the two run tables below.

Attributes:
    AcceptanceInputs (type): Paths the harness reads.
    RunState (type): State the set runners share during one acceptance run.
    latency_summary (function): Sample count and nearest-rank p50 and p95.
    covered (function): Whether one call answered with finite probabilities.
    psai_gates (function): Apply the #180 rev2 gates to scored PSAI rows.
    cord_passed (function): Whether the CORD set passed.
    cord_acceptance (function): Apply the CORD #161 floors to combined rows.
    run_acceptance (function): Run the five sets once and return the receipt
        mapping.
    write_receipt (function): Write the receipt JSON once, with the key
        masked, and return its sha256.
    live_gate_reason (function): Why the live acceptance cannot run, or
        ``None``.
    record_call (function): Add one call's seconds and coverage to its set.
    failed_row (function): Build the row for a call that raised.
    SET_RUNNERS (tuple): The five set runners in run order.
    DEVIATIONS (tuple): Pre-registered deviations recorded in the receipt.
    CallCaps (type): Hard call limits per request kind.
    AcceptanceStoppedError (type): A stop rule fired; the run still returns a
        receipt.
    CallCapReached (type): A request would pass its call cap.
    CountingTransport (type): Transport wrapper that counts calls and enforces
        caps.
    kv_cache_usage (function): Read the KV-cache usage gauge from
        ``/metrics``.

Examples:
    ```python
    from typevet_evals.vllm_acceptance import (
        DEVIATIONS,
        SET_RUNNERS,
        run_acceptance,
        write_receipt,
    )
    ```

See Also:
    - [typevet_evals.vllm_acceptance.core][]: caps, gates and receipt
    - [typevet_evals.vllm_acceptance.sets][]: the five set runners
    - [typevet_evals.vllm_acceptance.transport][]: call caps and ``/metrics``
    - [typevet_evals.throughput][]: throughput runs that reuse the caps
"""

from typevet_evals.vllm_acceptance.core import (
    AcceptanceInputs,
    RunState,
    cord_acceptance,
    cord_passed,
    covered,
    latency_summary,
    live_gate_reason,
    psai_gates,
    run_acceptance,
    write_receipt,
)
from typevet_evals.vllm_acceptance.sets import (
    DEVIATIONS,
    SET_RUNNERS,
    failed_row,
    record_call,
)
from typevet_evals.vllm_acceptance.transport import (
    AcceptanceStoppedError,
    CallCapReached,
    CallCaps,
    CountingTransport,
    kv_cache_usage,
)

__all__ = [
    "DEVIATIONS",
    "SET_RUNNERS",
    "AcceptanceInputs",
    "AcceptanceStoppedError",
    "CallCapReached",
    "CallCaps",
    "CountingTransport",
    "RunState",
    "cord_acceptance",
    "cord_passed",
    "covered",
    "failed_row",
    "kv_cache_usage",
    "latency_summary",
    "live_gate_reason",
    "psai_gates",
    "record_call",
    "run_acceptance",
    "write_receipt",
]
