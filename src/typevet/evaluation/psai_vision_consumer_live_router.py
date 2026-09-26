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
        auxiliary_budget=3,
    )
    assert result.matrix_rows
    ```

See Also:
    - [typevet.evaluation.psai_vision_consumer_live][]: receipt orchestration

[i177]: https://github.com/Alberto-Codes/typevet/issues/177
"""

from __future__ import annotations

import time
from collections.abc import Callable
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
from typevet.evaluation.psai_vision_consumer_accounting import ConsumerCallBudgetError
from typevet.evaluation.psai_vision_consumer_offline import run_offline_consumer_matrix

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
        auxiliary_http (int): Non-judgment HTTP calls consumed.
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
    auxiliary_http: int
    elapsed_s: float


def _tokenizer(client: httpx.Client, model: str) -> Callable[[str], tuple[int, ...]]:
    """Return a llama.cpp tokenize closure for ``model``.

    Returns:
        Callable that maps prompt text to token id tuples.
    """

    def tokenize_content(text: str) -> tuple[int, ...]:
        """Tokenize one prompt string through the router.

        Returns:
            Token id tuple from the router ``/tokenize`` endpoint.
        """
        payload = (
            client.post(
                "/tokenize",
                json={"model": model, "content": text, "add_special": False},
            )
            .raise_for_status()
            .json()
        )
        return tuple(payload["tokens"])

    return tokenize_content


def _served_template(client: httpx.Client, model: str) -> ServedTemplateClass:
    """Classify the router template for ``model`` and require Gemma 4 native.

    Returns:
        ``NATIVE_GEMMA4_TURN`` when the router matches the consumer protocol.

    Raises:
        ValueError: When the classified template is not Gemma 4 native.
    """
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


def run_consumer_live_matrix(
    *,
    settings: LlamaSettings,
    model: str,
    fixture_root: Path,
    auxiliary_budget: int,
) -> ConsumerLiveMatrixResult:
    """Run the frozen consumer matrix against a live llama.cpp router.

    Args:
        settings: Router connection options.
        model: Multimodal model id under test.
        fixture_root: Committed ``vision_smoke`` directory.
        auxiliary_budget: Maximum health, capability and template HTTP calls.

    Returns:
        Matrix rows, probabilities, and probe metadata.

    Raises:
        ValueError: When vision is unavailable or the template is wrong.
        ConsumerCallBudgetError: When auxiliary HTTP exceeds ``auxiliary_budget``.
    """
    base = settings.base_url.rstrip("/")
    auxiliary_http = 0
    t0 = time.perf_counter()
    with httpx.Client(base_url=base, timeout=settings.timeout) as client:
        health = client.get("/health").raise_for_status().json()
        auxiliary_http += 1
        capability = fetch_media_capability(client, f"{base}/", model)
        auxiliary_http += 1
        if not capability.vision:
            msg = "model reports text-only input modalities"
            raise ValueError(msg)
        served = _served_template(client, model)
        auxiliary_http += 1
        if auxiliary_http > auxiliary_budget:
            raise ConsumerCallBudgetError(
                f"auxiliary_http {auxiliary_http} exceed budget {auxiliary_budget}"
            )
        with LlamaCppCandidateScoringAdapter(
            base_url=base,
            timeout=settings.timeout,
            n_vocab=_N_VOCAB,
        ) as scoring:
            port = ScoringJudgmentAdapter(
                scoring,
                tokenize_content=_tokenizer(client, model),
                served_template=served,
            )
            matrix_rows, probabilities, _ = run_offline_consumer_matrix(
                port,
                fixture_root=fixture_root,
                model_id=model,
            )
    return ConsumerLiveMatrixResult(
        matrix_rows=matrix_rows,
        probabilities=probabilities,
        health=health,
        capability=capability,
        served=served,
        auxiliary_http=auxiliary_http,
        elapsed_s=round(time.perf_counter() - t0, 3),
    )
