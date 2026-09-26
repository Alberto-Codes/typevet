"""Router dispatch for instruction-variant live matrix ([#177][i177]).

Examples:
    ```python
    from pathlib import Path

    from typevet.evaluation.instruction_variant_consumer_live_router import (
        run_live_variant_matrix,
    )

    run = run_live_variant_matrix(
        fixture_root=Path("tests/fixtures/psai/vision_smoke"),
        seed_instruction="seed",
        candidate_instruction="candidate",
    )
    assert run is None or run.ledger.judgment_calls == 4
    ```

See Also:
    - [typevet.evaluation.instruction_variant_consumer_live][]: receipt orchestration
    - [typevet.adapters.outbound.judgment_scoring][]: scoring-backed judgment

[i177]: https://github.com/Alberto-Codes/typevet/issues/177
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import httpx

from typevet.adapters.inbound.settings import LlamaSettings, load_llama_settings
from typevet.adapters.outbound.gemma import (
    ServedTemplateClass,
    classify_served_template,
)
from typevet.adapters.outbound.judgment_scoring import ScoringJudgmentAdapter
from typevet.adapters.outbound.llama_cpp_multimodal import fetch_media_capability
from typevet.adapters.outbound.llama_cpp_scoring import (
    DEFAULT_N_VOCAB,
    LlamaCppCandidateScoringAdapter,
)
from typevet.evaluation.instruction_variant_consumer_matrix import (
    _LedgerJudgmentPort as LedgerJudgmentPort,
)
from typevet.evaluation.instruction_variant_consumer_matrix import (
    probe_invalid_model,
    run_negative_probe,
    run_variant_arm,
    slice_present_controls,
)
from typevet.evaluation.instruction_variant_consumer_protocol import (
    VariantDispatchLedger,
)
from typevet.evaluation.instruction_variant_consumer_run import VariantMatrixRun
from typevet.evaluation.psai_vision_consumer_offline import load_frozen_consumer_fixture
from typevet.evaluation.runner.live_gate import live_gate_action, live_skip_reason
from typevet.ports.judgment import JudgmentPort

_MODEL_ENV = ("TYPEVET_GEMMA_MODEL", "TYPEVET_LLAMA__DEFAULT_MODEL")
_VARIANT_SCORING_PER_MATRIX = 4
_SUPPORTED_NATIVE = frozenset(
    {
        ServedTemplateClass.NATIVE_GEMMA3_TURN,
        ServedTemplateClass.NATIVE_GEMMA4_TURN,
    }
)


def _resolve_model() -> str:
    for key in _MODEL_ENV:
        val = os.environ.get(key)
        if val:
            return val
    settings = load_llama_settings()
    return settings.multimodal_model


def _classify_native_template(
    client: httpx.Client,
    model: str,
    *,
    require_gemma4: bool,
) -> ServedTemplateClass:
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
    if require_gemma4 and family is not ServedTemplateClass.NATIVE_GEMMA4_TURN:
        msg = f"expected NATIVE_GEMMA4_TURN, got {family.value}"
        raise ValueError(msg)
    if family not in _SUPPORTED_NATIVE:
        msg = f"unsupported served template for native vision: {family.value}"
        raise ValueError(msg)
    return family


def _tokenize_factory(
    client: httpx.Client,
    model: str,
) -> Callable[[str], tuple[int, ...]]:
    def tokenize(text: str) -> tuple[int, ...]:
        """Tokenize ``text`` through the router ``/tokenize`` endpoint.

        Returns:
            Token id tuple from the router JSON body.
        """
        body = client.post(
            "/tokenize",
            json={"model": model, "content": text, "add_special": False},
        )
        return tuple(body.raise_for_status().json()["tokens"])

    return tokenize


@contextmanager
def _open_native_vision_port(
    *,
    settings: LlamaSettings,
    model: str,
    http_client: httpx.Client,
) -> Iterator[JudgmentPort]:
    """Yield a Gemma native-turn judgment port for one live matrix.

    Yields:
        Scoring-backed judgment port wired to ``http_client``.

    Raises:
        ValueError: When vision or template probes fail.
    """
    base = settings.base_url.rstrip("/")
    capability = fetch_media_capability(http_client, f"{base}/", model)
    if not capability.vision:
        msg = "model reports text-only input modalities"
        raise ValueError(msg)
    served = _classify_native_template(http_client, model, require_gemma4=True)
    tokenize = _tokenize_factory(http_client, model)
    scoring = LlamaCppCandidateScoringAdapter(
        base_url=base,
        timeout=settings.timeout,
        client=http_client,
        n_vocab=DEFAULT_N_VOCAB,
    )
    port: JudgmentPort = ScoringJudgmentAdapter(
        scoring,
        tokenize_content=tokenize,
        served_template=served,
    )
    try:
        yield port
    finally:
        scoring.close()


def _live_variant_arms(
    port: JudgmentPort,
    *,
    controls: Any,
    fixture: Any,
    loader: Callable[[str], bytes],
    model_id: str,
    seed_instruction: str,
    candidate_instruction: str,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    seed_rows, seed_outcomes = run_variant_arm(
        port,
        controls=controls,
        fixture=fixture,
        loader=loader,
        model_id=model_id,
        instruction=seed_instruction,
        arm="seed",
    )
    candidate_rows, candidate_outcomes = run_variant_arm(
        port,
        controls=controls,
        fixture=fixture,
        loader=loader,
        model_id=model_id,
        instruction=candidate_instruction,
        arm="candidate",
    )
    return seed_rows, seed_outcomes, candidate_rows, candidate_outcomes


def run_live_variant_matrix(
    *,
    fixture_root: Path,
    seed_instruction: str,
    candidate_instruction: str,
) -> VariantMatrixRun | None:
    """Execute the live variant matrix when the router gate allows.

    Returns:
        Matrix run payload, or ``None`` when the live gate skips dispatch.
    """
    settings = load_llama_settings()
    if live_gate_action(live_skip_reason(settings)).name != "RUN":
        return None
    model_id = _resolve_model()
    fixture = load_frozen_consumer_fixture(fixture_root)
    controls = slice_present_controls(fixture)
    loader = lambda name: (fixture_root / name).read_bytes()
    ledger = VariantDispatchLedger()
    with (
        httpx.Client(base_url=settings.base_url, timeout=settings.timeout) as client,
        _open_native_vision_port(
            settings=settings,
            model=model_id,
            http_client=client,
        ) as bare,
    ):
        invalid = probe_invalid_model(bare)
        if invalid["ok"]:
            ledger.record_failure()
        port = LedgerJudgmentPort(bare, ledger)
        negative = run_negative_probe(model_id, fixture_root, loader, fixture)
        seed_rows, seed_outcomes, candidate_rows, candidate_outcomes = (
            _live_variant_arms(
                port,
                controls=controls,
                fixture=fixture,
                loader=loader,
                model_id=model_id,
                seed_instruction=seed_instruction,
                candidate_instruction=candidate_instruction,
            )
        )
    ledger.scoring_requests = _VARIANT_SCORING_PER_MATRIX
    return VariantMatrixRun(
        fixture_root=fixture_root,
        controls=controls,
        ledger=ledger,
        invalid=invalid,
        negative=negative,
        seed_rows=seed_rows,
        candidate_rows=candidate_rows,
        seed_outcomes=seed_outcomes,
        candidate_outcomes=candidate_outcomes,
    )
