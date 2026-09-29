"""Router HTTP helpers for the PSAI consumer live matrix ([#177][i177]).

Examples:
    ```python
    from pathlib import Path

    from typevet.adapters.inbound.settings import load_llama_settings
    from typevet.evaluation.psai_vision_consumer_live_router import (
        run_consumer_live_matrix,
    )

    settings = load_llama_settings()
    result = run_consumer_live_matrix(
        settings=settings,
        model="gemma-4-31b-kv9-q4km-mm",
        fixture_root=Path("tests/fixtures/psai/vision_smoke"),
    )
    assert result.matrix_rows
    ```

See Also:
    - [typevet.evaluation.psai_vision_consumer_dispatch][]: budget ledger
    - [typevet.evaluation.psai_vision_consumer_live][]: receipt orchestration
    - [typevet.adapters.outbound.llama_cpp.gemma_native_vision_factory][]: native vision factory

Probes health, capability, and template identity before matrix dispatch;
each HTTP leg increments auxiliary or tokenizer counters on the ledger.

[i177]: https://github.com/Alberto-Codes/typevet/issues/177
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from typevet.adapters.inbound.settings import LlamaSettings
from typevet.adapters.outbound.gemma import (
    ServedTemplateClass,
)
from typevet.adapters.outbound.llama_cpp.gemma_native_vision_factory import (
    open_gemma_native_vision_judgment,
)
from typevet.adapters.outbound.llama_cpp.multimodal import (
    MediaCapability,
)
from typevet.evaluation.psai_vision_consumer_dispatch import (
    ConsumerDispatchLedger,
    wrap_judgment_port,
    wrap_scoring_port,
)
from typevet.evaluation.psai_vision_consumer_live_identity import (
    finalize_consumer_live_identity,
    start_consumer_live_identity,
)
from typevet.evaluation.psai_vision_consumer_offline import (
    load_frozen_consumer_fixture,
    run_offline_consumer_matrix,
)


@dataclass(frozen=True, slots=True)
class ConsumerLiveMatrixResult:
    """Outputs from one live matrix dispatch through the router.

    Attributes:
        matrix_rows (list[dict[str, Any]]): Serialized judgment rows.
        probabilities (dict[tuple[str, str], float]): Visual Noul map.
        health (dict[str, Any]): Router ``/health`` JSON body.
        capability (MediaCapability): Vision capability probe.
        served (ServedTemplateClass): Classified native template family.
        ledger (ConsumerDispatchLedger): Dispatch counters for the run.
        identity (dict[str, object]): Pre-dispatch finalized experiment identity.
        elapsed_s (float): Wall seconds for the matrix leg.

    Examples:
        ```python
        from typevet.evaluation.psai_vision_consumer_live_router import (
            ConsumerLiveMatrixResult,
        )

        assert ConsumerLiveMatrixResult.__dataclass_fields__
        ```
    """

    matrix_rows: list[dict[str, Any]]
    probabilities: dict[tuple[str, str], float]
    health: dict[str, Any]
    capability: MediaCapability
    served: ServedTemplateClass
    ledger: ConsumerDispatchLedger
    identity: dict[str, object]
    elapsed_s: float


def run_consumer_live_matrix(
    *,
    settings: LlamaSettings,
    model: str,
    fixture_root: Path,
    ledger: ConsumerDispatchLedger | None = None,
) -> ConsumerLiveMatrixResult:
    """Run the frozen matrix, counting every factory and matrix HTTP dispatch.

    Args:
        settings: Router connection options.
        model: Multimodal model id under test.
        fixture_root: Committed ``vision_smoke`` directory.
        ledger: Optional dispatch ledger; a fresh ledger is used when omitted.

    Returns:
        Matrix rows, probabilities, and probe metadata.

    Raises:
        ValueError: When vision is unavailable or the template is wrong.
        ConsumerCallBudgetError: When auxiliary HTTP exceeds frozen ceilings.
    """
    dispatch = ledger or ConsumerDispatchLedger()
    fixture = load_frozen_consumer_fixture(fixture_root)
    base = settings.base_url.rstrip("/")
    t0 = time.perf_counter()
    with httpx.Client(base_url=base, timeout=settings.timeout) as client:
        client.event_hooks["request"].append(dispatch.before_http)
        health = client.get("/health").raise_for_status().json()
        with open_gemma_native_vision_judgment(
            settings=settings,
            model=model,
            http_client=client,
            scoring_port_wrapper=lambda scoring: wrap_scoring_port(scoring, dispatch),
        ) as session:
            capability, served = session.capability, session.served
            run_start, evaluated = start_consumer_live_identity(
                fixture_root=fixture_root,
                fixture=fixture,
                model=model,
                served_template=served.name,
                health=health,
            )
            matrix_rows, probabilities = run_offline_consumer_matrix(
                wrap_judgment_port(session.port, dispatch),
                fixture_root=fixture_root,
                model_id=model,
            )[:2]
    identity = finalize_consumer_live_identity(
        run_start=run_start,
        evaluated=evaluated,
        ledger=dispatch,
    )
    return ConsumerLiveMatrixResult(
        matrix_rows=matrix_rows,
        probabilities=probabilities,
        health=health,
        capability=capability,
        served=served,
        ledger=dispatch,
        identity=identity,
        elapsed_s=round(time.perf_counter() - t0, 3),
    )
