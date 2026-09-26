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
    classify_served_template,
)
from typevet.adapters.outbound.judgment_scoring import ScoringJudgmentAdapter
from typevet.adapters.outbound.llama_cpp_multimodal import (
    MediaCapability,
    fetch_media_capability,
)
from typevet.adapters.outbound.llama_cpp_scoring import LlamaCppCandidateScoringAdapter
from typevet.evaluation.datasets.psai_vision import VisionSmokeFixture
from typevet.evaluation.experiment_identity import (
    EvaluatedInputsSnapshot,
    RunIdentityStart,
)
from typevet.evaluation.psai_vision_consumer_dispatch import (
    ConsumerDispatchLedger,
    counting_tokenizer,
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

_N_VOCAB = 262144


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


def _served_template(
    client: httpx.Client,
    model: str,
    ledger: ConsumerDispatchLedger,
) -> ServedTemplateClass:
    """Classify the router template for ``model`` and require Gemma 4 native.

    Args:
        client: Router HTTP client.
        model: Model id under test.
        ledger: Dispatch ledger receiving metadata HTTP counts.

    Returns:
        ``NATIVE_GEMMA4_TURN`` when the router matches the consumer protocol.

    Raises:
        ValueError: When the classified template is not Gemma 4 native.
    """
    ledger.before_metadata_http()
    rendered = (
        client.post(
            "/apply-template",
            json={
                "model": model,
                "messages": [{"role": "user", "content": "hello"}],
                "add_generation_prompt": True,
            },
        )
        .raise_for_status()
        .json()["prompt"]
    )
    family = classify_served_template(rendered)
    if family is not ServedTemplateClass.NATIVE_GEMMA4_TURN:
        msg = f"expected NATIVE_GEMMA4_TURN, got {family}"
        raise ValueError(msg)
    return family


def _probe_router(
    *,
    client: httpx.Client,
    base: str,
    model: str,
    fixture_root: Path,
    fixture: VisionSmokeFixture,
    dispatch: ConsumerDispatchLedger,
) -> tuple[
    dict[str, Any],
    MediaCapability,
    ServedTemplateClass,
    RunIdentityStart,
    EvaluatedInputsSnapshot,
]:
    """Run health, capability and template probes; snapshot identity before matrix.

    Returns:
        Health JSON, capability, served template, and pre-dispatch identity parts.

    Raises:
        ValueError: When the model reports text-only input modalities.
    """
    dispatch.before_metadata_http()
    health = client.get("/health").raise_for_status().json()
    dispatch.before_metadata_http()
    capability = fetch_media_capability(client, f"{base}/", model)
    if not capability.vision:
        msg = "model reports text-only input modalities"
        raise ValueError(msg)
    served = _served_template(client, model, dispatch)
    run_start, evaluated = start_consumer_live_identity(
        fixture_root=fixture_root,
        fixture=fixture,
        model=model,
        served_template=served.name,
        health=health,
    )
    return health, capability, served, run_start, evaluated


def _dispatch_consumer_matrix(
    *,
    client: httpx.Client,
    base: str,
    settings: LlamaSettings,
    model: str,
    fixture_root: Path,
    dispatch: ConsumerDispatchLedger,
    served: ServedTemplateClass,
) -> tuple[list[dict[str, Any]], dict[tuple[str, str], float]]:
    """Run judgment/scoring matrix rows through a configured router client.

    Returns:
        Matrix rows and visual Noul probability map (negative leg omitted).
    """
    with LlamaCppCandidateScoringAdapter(
        base_url=base,
        timeout=settings.timeout,
        n_vocab=_N_VOCAB,
    ) as scoring:
        scoring_wrapped = wrap_scoring_port(scoring, dispatch)
        port = ScoringJudgmentAdapter(
            scoring_wrapped,
            tokenize_content=counting_tokenizer(
                client.post, ledger=dispatch, model=model
            ),
            served_template=served,
        )
        port = wrap_judgment_port(port, dispatch)
        return run_offline_consumer_matrix(
            port,
            fixture_root=fixture_root,
            model_id=model,
        )[:2]


def run_consumer_live_matrix(
    *,
    settings: LlamaSettings,
    model: str,
    fixture_root: Path,
    ledger: ConsumerDispatchLedger | None = None,
) -> ConsumerLiveMatrixResult:
    """Run the frozen consumer matrix against a live llama.cpp router.

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
        health, capability, served, run_start, evaluated = _probe_router(
            client=client,
            base=base,
            model=model,
            fixture_root=fixture_root,
            fixture=fixture,
            dispatch=dispatch,
        )
        matrix_rows, probabilities = _dispatch_consumer_matrix(
            client=client,
            base=base,
            settings=settings,
            model=model,
            fixture_root=fixture_root,
            dispatch=dispatch,
            served=served,
        )
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
