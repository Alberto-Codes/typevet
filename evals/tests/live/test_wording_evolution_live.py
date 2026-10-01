r"""Opt-in live DIFrauD ``is_scam`` wording evolution and held-out check (#309).

Three tests, each gated by its own variable. Each skips when its variable is
unset and fails when ``TYPEVET_REQUIRE_LIVE`` is truthy and it is unset.

**Evolution** (``TYPEVET_WORDING_ARTIFACT`` names a new JSON file). gepa-adk
evolves the finvet-matching ``is_scam`` instructions on the local llama.cpp
router. The judge is ``gemma-4-31b-kv9-q4km-mm`` through
``TypevetSystemOnePort`` over a text-only scoring session. That alias serves
the native Gemma 4 template; the decoder alias
``gemma-4-31b-24gib-kv11-decoder`` serves ChatML (#324, #327). The reflector is
``Qwen3.8-27B-UD-Q4_K_M`` through the router's OpenAI-compatible ``/v1``
(LiteLLM, dummy key). The router holds one of the two models at a time, so
it swaps between evaluation and reflection. Train is a stratified
1,000-row subset of the DIFrauD SMS train split (seed 0); selection is a
stratified 200-row validation subset (seed 0); ``max_iterations`` 10; the
length cap is 1.5 times the seed. The held-out rows are never loaded into
the runner. The checkpoint must lie outside the
repository. Settings:

- ``TYPEVET_WORDING_CHECKPOINT``: checkpoint JSON (default: the artifact
  path with ``.checkpoint.json``); ``TYPEVET_WORDING_RESUME=1`` resumes.
- ``TYPEVET_WORDING_MAX_ITERATIONS`` (10), ``TYPEVET_WORDING_VALIDATION_ROWS``
  (200), ``TYPEVET_WORDING_TRAIN_ROWS`` (1000, a stratified subset,
  seed 0), ``TYPEVET_WORDING_MINIBATCH`` (8 train rows per proposal gate).
- ``TYPEVET_WORDING_JUDGE``, ``TYPEVET_WORDING_REFLECTOR``,
  ``TYPEVET_WORDING_REFLECTOR_BASE`` (router ``/v1``) and
  ``TYPEVET_WORDING_REFLECTOR_TIMEOUT`` (1800 seconds).
- ``TYPEVET_WORDING_JUDGE_PROVIDER``: ``gemma`` (default, the session above)
  or ``jev`` (#328). ``TYPEVET_WORDING_CONCURRENCY`` sets gepa-adk's
  ``max_concurrent_evals`` (default 5 for Gemma, 1 for Jev).

**Wording parts (#365).** ``TYPEVET_WORDING_COMPONENTS`` names the parts that
evolve, comma-separated, from ``instructions``, ``criteria_true`` and
``criteria_false`` (default ``instructions``); the other parts are frozen.
``TYPEVET_WORDING_SEED_CRITERIA=1`` adds the pre-registered ``true`` and
``false`` criteria (``SEED_CRITERIA``) to the seed; unset or ``0`` keeps the
#252 seed without criteria. A criteria part needs the criteria seed. The
test refuses an unknown part name before any call. The artifact records
``components``, ``seed_parts``, ``evolved_parts`` and their digests. The
held-out and comparison tests read the same ``TYPEVET_WORDING_SEED_CRITERIA``
and build both arms from the artifact's ``evolved_parts``; the seed must equal
the artifact's ``seed_parts``. An artifact without ``evolved_parts`` (#252)
gives its evolved ``instructions`` with the seed criteria in both arms. The
identity prompts hash each arm's criteria.

**Jev judge** (``TYPEVET_WORDING_JUDGE_PROVIDER=jev``). The judge is
judgevet's ``HTTPSystemOneAdapter``, built from judgevet's ``Settings``
(``JEV_API__*``; key ``JEV_API__KEY`` or ``TYPESAFE_API_KEY``). The test
fails before any network call unless ``JEV_API__SPEND_MAX_ATTEMPTS`` sets a
spend cap. One ``SpendCap`` spans the run; a refused attempt sends nothing,
and a gepa-adk stopper ends the run when the cap is spent. The requested
model is ``TYPEVET_WORDING_JUDGE`` or ``JEV_API__DEFAULT_MODEL``
(``jev-latest``); each call records the model id Jev reports. The adapter
retries a 429 or a 5xx under ``JEV_API__MAX_ATTEMPTS`` (3), and every attempt
counts against the cap. The default concurrency is 1, as in finvet's Jev
evolutions, because Jev states no rate limit. The artifact adds the spend,
the stop reason and the judge identity, and passes the key-free check.
A call the cap refuses scores its row 0 and can reject a better candidate, so
the artifact counts ``budget_refusals`` and sets ``valid`` false when any
occur, and the test then fails after it writes the artifact. The key is
resolved at each use and is not bound to a local of the test.

**Held-out check** (``TYPEVET_WORDING_HELD_OUT_RECEIPT`` names a new receipt;
``TYPEVET_WORDING_EVOLVED_ARTIFACT`` names the evolution artifact). Scores
the seed and evolved wording once each on all 158 held-out rows (316 calls)
and writes a key-free receipt: metrics for both, paired bootstrap 95%
intervals (2,000 resamples, seed 0; context only), the pre-registered
verdict, the #133 comparison, pins and call counts. ``TYPEVET_BACKEND``
selects the backend as in ``open_judgment``: ``llama_cpp`` (default; the
same text-only session as the evolution) or ``vllm``
(``TYPEVET_VLLM__*`` and ``TYPEVET_VLLM_MODEL_REVISION``).
``TYPEVET_GIT_STATUS_PORCELAIN`` carries the porcelain status text.

**Jev vs Gemma comparison (#329, parent #252)**
(``TYPEVET_WORDING_COMPARISON_RECEIPT`` names a new receipt;
``TYPEVET_WORDING_EVOLVED_ARTIFACT`` names the judge's own evolution
artifact). One judge scores the seed wording and its own evolved wording once
each on the 158 held-out rows. ``TYPEVET_WORDING_JUDGE_PROVIDER=jev`` uses
the Jev judge above (spend cap required before any call); ``gemma`` uses the
``TYPEVET_BACKEND`` session (``llama_cpp`` or ``vllm``). The artifact's
``judge_provider`` must be the judge's (``jev`` for Jev, ``gemma`` for both
Gemma backends), valid and free of budget refusals. The receipt holds per arm
Cohen's kappa against the DIFrauD labels (p >= 0.5 reads scam), Brier, ECE
(10 bins) and accuracy, the paired bootstrap (context only), the model,
backend, quant or revision, the probed served template, per-call latency and
input tokens and the wall time. It has no pass rule and does not apply the
#309 verdict. ``TYPEVET_WORDING_COMPARISON_SMOKE_ROWS`` (1 to 6) scores that
many stratified validation rows instead; a smoke never loads a held-out row.

**Framing smoke** (``TYPEVET_WORDING_SMOKE_RECEIPT`` names a new receipt).
Scores the seed wording once on 10 stratified validation rows (seed 0) on
the ``TYPEVET_BACKEND`` backend and writes the served template, each call's
latency and input tokens and the wall time. No held-out row is loaded.

**Served template (#327).** Each session probes the served template:
llama.cpp ``/apply-template`` or vLLM ``/tokenize`` with chat messages. A
session refuses any template other than ``native_gemma4_turn`` unless
``TYPEVET_WORDING_ALLOW_DEGRADED_TEMPLATE=1``. Receipts and the evolution
artifact record per-call latency and input tokens.

Examples:
    ```bash
    TYPEVET_WORDING_ARTIFACT=$HOME/typevet-309/evolution.json \
      uv run pytest evals/tests/live/test_wording_evolution_live.py -m live -q -s \
      -k evolution

    TYPEVET_WORDING_JUDGE_PROVIDER=jev JEV_API__SPEND_MAX_ATTEMPTS=4000 \
      TYPEVET_WORDING_ARTIFACT=$HOME/typevet-328/evolution_jev.json \
      uv run pytest evals/tests/live/test_wording_evolution_live.py -m live -q -s \
      -k evolution

    TYPEVET_BACKEND=vllm TYPEVET_VLLM_MODEL_REVISION=<sha> \
      TYPEVET_VLLM__BASE_URL=https://<pod>-8000.proxy.runpod.net \
      TYPEVET_VLLM__MODEL=google/gemma-4-31B-it \
      TYPEVET_WORDING_EVOLVED_ARTIFACT=$HOME/typevet-309/evolution.json \
      TYPEVET_WORDING_HELD_OUT_RECEIPT=evals/fixtures/difraud/receipts/wording_held_out_vllm.json \
      uv run pytest evals/tests/live/test_wording_evolution_live.py -m live -q -s \
      -k held_out

    R=evals/fixtures/difraud/receipts
    TYPEVET_WORDING_JUDGE_PROVIDER=jev JEV_API__SPEND_MAX_ATTEMPTS=400 \
      TYPEVET_WORDING_EVOLVED_ARTIFACT=$R/wording252_evolution_jev.json \
      TYPEVET_WORDING_COMPARISON_RECEIPT=$R/wording252_held_out_jev.json \
      uv run pytest evals/tests/live/test_wording_evolution_live.py -m live -q -s \
      -k comparison

    TYPEVET_WORDING_COMPONENTS=criteria_true,criteria_false \
      TYPEVET_WORDING_SEED_CRITERIA=1 \
      TYPEVET_WORDING_ARTIFACT=$R/wording365_armA_evolution_gemma_llama_cpp.json \
      TYPEVET_WORDING_CHECKPOINT=$HOME/typevet-365/armA.checkpoint.json \
      uv run pytest evals/tests/live/test_wording_evolution_live.py -m live -q -s \
      -k evolution
    ```

See Also:
    - [typevet_evals.wording.runner][]: the evolution runner
    - [typevet_evals.wording.held_out][]: held-out scoring and receipt
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

import httpx
import pytest
from gepa_adk.domain.stopper import StopperState
from gepa_adk.ports.stopper import StopperProtocol
from google.adk.models.lite_llm import LiteLlm
from judgevet.adapters.inbound.settings import ApiSettings, Settings
from judgevet.adapters.outbound.http import HTTPSystemOneAdapter
from judgevet.domain.questions import Noul
from judgevet.domain.spend import SpendCap
from judgevet.providers import ProviderError

from typevet.adapters.inbound.backend_settings import (
    load_backend,
    load_vllm_settings,
    open_judgment,
)
from typevet.adapters.inbound.judgevet import TypevetSystemOnePort
from typevet.adapters.inbound.settings import load_llama_settings
from typevet.adapters.outbound.judgment_scoring import ScoringJudgmentAdapter
from typevet.adapters.outbound.llama_cpp.scoring import LlamaCppCandidateScoringAdapter
from typevet.ports.judgment import JudgmentPort
from typevet_evals.datasets.difraud import (
    IS_SCAM_NOUL_SCHEMA,
    PINNED_REVISION,
    PRIMARY_NOUL_NAME,
    DIFrauDRecord,
    DIFrauDSplits,
    load_splits,
)
from typevet_evals.experiment_identity import (
    PromptSpec,
    RuntimeBuild,
    WorkingTreeState,
    begin_run_identity,
    capture_working_tree_at_run_start,
    finalize_experiment_identity,
    read_baseline_commit,
    snapshot_evaluated_inputs,
    write_receipt_exclusive,
)
from typevet_evals.face_match import ensure_key_free, served_weights_pins
from typevet_evals.runner.live_gate import require_live_enabled
from typevet_evals.wording import (
    WordingRun,
    WordingRunConfig,
    evolution_artifact,
    evolve_wording,
    held_out_receipt,
    score_held_out,
    stratified_subset,
    train_subset,
)
from typevet_evals.wording.calls import TimedJudgePort, call_summary, error_count
from typevet_evals.wording.comparison import (
    ComparisonSubject,
    comparison_receipt,
    evolved_parts_for,
)
from typevet_evals.wording.held_out import (
    DEFAULT_TRAIN_ROWS,
    HeldOutRows,
    ValidationRows,
)
from typevet_evals.wording.parts import (
    CRITERIA_FALSE,
    CRITERIA_TRUE,
    INSTRUCTIONS,
    PART_NAMES,
    WordingParts,
    artifact_parts,
    check_selection,
    seed_mapping,
)
from typevet_evals.wording.served import (
    TEXT_JUDGE,
    probe_llama_template,
    probe_vllm_template,
    require_native_template,
)
from typevet_evals.wording.transport import JudgePort

_REPO_ROOT = Path(__file__).resolve().parents[3]
_WORDING_SRC = _REPO_ROOT / "evals" / "src" / "typevet_evals" / "wording"
_JUDGE = TEXT_JUDGE
_SMOKE_ROWS = 10
_REFLECTOR = "Qwen3.8-27B-UD-Q4_K_M"
_N_VOCAB = 262144
_TIMEOUT = 900.0
SEED_TEXT = str(IS_SCAM_NOUL_SCHEMA["properties"][PRIMARY_NOUL_NAME]["instructions"])
_SEED = Noul(instructions=SEED_TEXT)
SEED_CRITERIA: Final[Mapping[str, str]] = MappingProxyType(
    {
        "true": (
            "The message tries to deceive the reader into money, credentials "
            "or an unsafe action."
        ),
        "false": "The message is an ordinary personal, commercial or informational text.",
    }
)
_LABELS = ("true", "false")


def wording_components(environ: Mapping[str, str]) -> tuple[str, ...]:
    """Read ``TYPEVET_WORDING_COMPONENTS``: the part names that evolve (#365).

    Args:
        environ: The environment.

    Returns:
        The comma-separated names, in order; ``("instructions",)`` when unset.

    Raises:
        ValueError: If a name is empty or not a ``Noul`` part. The message
            does not repeat the value.
    """
    raw = environ.get("TYPEVET_WORDING_COMPONENTS", "").strip()
    if not raw:
        return (INSTRUCTIONS,)
    names = tuple(name.strip() for name in raw.split(","))
    if not all(name in PART_NAMES for name in names):
        msg = f"TYPEVET_WORDING_COMPONENTS names a part outside {', '.join(PART_NAMES)}"
        raise ValueError(msg)
    return names


def wording_seed(environ: Mapping[str, str]) -> Noul:
    """Build the seed; ``TYPEVET_WORDING_SEED_CRITERIA=1`` adds the criteria.

    Args:
        environ: The environment.

    Returns:
        The #252 seed ``Noul``, with ``SEED_CRITERIA`` when the knob is ``1``.

    Raises:
        ValueError: If the knob is set to a value other than ``0`` or ``1``.
    """
    raw = environ.get("TYPEVET_WORDING_SEED_CRITERIA", "").strip()
    if raw not in {"", "0", "1"}:
        raise ValueError("TYPEVET_WORDING_SEED_CRITERIA must be 0 or 1")
    criteria = dict(SEED_CRITERIA) if raw == "1" else None
    return Noul(instructions=SEED_TEXT, criteria=criteria)


def evolution_inputs(environ: Mapping[str, str]) -> tuple[Noul, tuple[str, ...]]:
    """Return the seed and the checked selection, before any call.

    Args:
        environ: The environment.

    Returns:
        The seed and the part names that evolve.
    """
    seed = wording_seed(environ)
    return seed, check_selection(wording_components(environ), seed_mapping(seed))


async def evolve_from_env(
    port: JudgePort,
    environ: Mapping[str, str],
    *,
    train: Sequence[DIFrauDRecord],
    validation: Sequence[DIFrauDRecord],
    config: WordingRunConfig,
) -> WordingRun:
    """Evolve the parts the knobs select from the seed the knobs build.

    Args:
        port: The judge port.
        environ: The environment.
        train: The train records.
        validation: The validation records.
        config: The run settings.

    Returns:
        The run; its ``components``, ``seed_parts`` and ``evolved_parts``
        reach the artifact.
    """
    seed, components = evolution_inputs(environ)
    return await evolve_wording(
        port=port,
        seed=seed,
        key=PRIMARY_NOUL_NAME,
        train=train,
        validation=validation,
        config=config,
        components=components,
    )


def held_out_parts(
    artifact: Mapping[str, Any], environ: Mapping[str, str]
) -> tuple[Noul, WordingParts]:
    """Return the seed and the parts the held-out arms send.

    Args:
        artifact: The evolution artifact.
        environ: The environment; ``TYPEVET_WORDING_SEED_CRITERIA`` must
            match the artifact's seed.

    Returns:
        The seed and ``artifact_parts`` over the seed's parts.
    """
    seed = wording_seed(environ)
    return seed, artifact_parts(artifact, seed_mapping(seed))


def comparison_parts(
    artifact: Mapping[str, Any], seed: Noul, *, backend: str
) -> WordingParts:
    """Return the parts of the comparison judge's own evolution.

    Args:
        artifact: The judge's evolution artifact.
        seed: The seed ``Noul``.
        backend: ``jev``, ``llama_cpp`` or ``vllm``.

    Returns:
        ``evolved_parts_for`` over the seed's parts.
    """
    return evolved_parts_for(artifact, backend=backend, seed_parts=seed_mapping(seed))


def _criteria(mapping: Mapping[str, str]) -> dict[str, str]:
    if CRITERIA_TRUE not in mapping:
        return {}
    return {"true": mapping[CRITERIA_TRUE], "false": mapping[CRITERIA_FALSE]}


def prompt_specs(parts: WordingParts) -> tuple[PromptSpec, PromptSpec]:
    """Return the identity prompts of both arms, with their criteria.

    Args:
        parts: The run's parts.

    Returns:
        The ``seed`` and ``evolved`` prompts; no criteria for a seed without.
    """
    return (
        PromptSpec("seed", _LABELS, parts.seed_text, _criteria(parts.seed)),
        PromptSpec("evolved", _LABELS, parts.evolved_text, _criteria(parts.evolved)),
    )


@dataclass(frozen=True)
class _TextSession:
    """A text-only llama.cpp judgment session.

    Attributes:
        port (JudgmentPort): Scoring-backed judgment pinned to the model.
        model (str): The served model id.
        client (httpx.Client): The router client.
        served_template (str): The served template class.
    """

    port: JudgmentPort
    model: str
    client: httpx.Client
    served_template: str


def _required_path(name: str, *, new: bool) -> Path:
    raw = os.environ.get(name, "").strip()
    if not raw:
        reason = f"{name} not set"
        if require_live_enabled():
            pytest.fail(reason)
        pytest.skip(reason)
    path = Path(raw).expanduser()
    if new and path.exists():
        pytest.fail(f"{name} must name a new file: {path} exists")
    if not new and not path.is_file():
        pytest.fail(f"{name} must name an existing file: {path}")
    return path


def _int_env(name: str, default: int | None) -> int | None:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else default


@contextmanager
def _llama_text_session(
    model: str, environ: Mapping[str, str]
) -> Iterator[_TextSession]:
    """Open a text-only scoring session on the router, pinned to ``model``.

    ``LlamaCppCandidateScoringAdapter`` sends a request once more when the
    router closes a connection before a response (#305). The session refuses
    a template other than native Gemma 4 unless the override is set (#327).

    Yields:
        The session.
    """
    settings = load_llama_settings(environ)
    base = settings.base_url.rstrip("/")
    timeout = max(settings.timeout, _TIMEOUT)
    with httpx.Client(base_url=base, timeout=timeout) as client:
        served = require_native_template(probe_llama_template(client, model), environ)

        def tokenize(text: str) -> tuple[int, ...]:
            body = client.post(
                "/tokenize",
                json={"model": model, "content": text, "add_special": False},
            )
            return tuple(body.raise_for_status().json()["tokens"])

        with LlamaCppCandidateScoringAdapter(
            base_url=base, timeout=timeout, client=client, n_vocab=_N_VOCAB
        ) as scorer:
            port = ScoringJudgmentAdapter(
                scorer,
                tokenize_content=tokenize,
                served_template=served,
                pinned_model=model,
            )
            yield _TextSession(port, model, client, served.value)


def _server_facts(backend: str, client: httpx.Client, model: str) -> dict[str, object]:
    served = client.get("/v1/models").raise_for_status().json()["data"]
    facts: dict[str, object] = {
        "served": [
            {k: item.get(k) for k in ("id", "root", "max_model_len") if k in item}
            for item in served
            if item.get("id") == model
        ]
    }
    if backend == "llama_cpp":
        body = client.get("/props", params={"model": model}).raise_for_status().json()
        facts["build_info"] = body.get("build_info", "unknown")
        facts["model_alias"] = body.get("model_alias")
    else:
        version = client.get("/version")
        facts["build_info"] = (
            version.json().get("version", "unknown")
            if version.status_code == httpx.codes.OK
            else "unknown"
        )
    return facts


def _outside_repo(path: Path) -> Path:
    resolved = path.resolve()
    if resolved.is_relative_to(_REPO_ROOT):
        pytest.fail(f"the checkpoint must lie outside the repository: {resolved}")
    return resolved


def _router_v1(environ: Mapping[str, str]) -> str:
    return load_llama_settings(environ).base_url.rstrip("/") + "/v1"


def _reflector(environ: Mapping[str, str]) -> LiteLlm:
    return LiteLlm(
        model="openai/" + environ.get("TYPEVET_WORDING_REFLECTOR", _REFLECTOR),
        api_base=environ.get("TYPEVET_WORDING_REFLECTOR_BASE", _router_v1(environ)),
        api_key="local-router-no-key",
        timeout=float(environ.get("TYPEVET_WORDING_REFLECTOR_TIMEOUT", "1800")),
    )


_PROVIDERS = ("gemma", "jev")
_DEFAULT_CONCURRENCY = {"gemma": 5, "jev": 1}
_BUDGET_REFUSAL = "JevBudgetExceededError"


def _no_secret_key_free(text: str) -> None:
    ensure_key_free(text, secrets=())


@dataclass(frozen=True)
class _EvolutionJudge:
    """The judge of one evolution and what the artifact records about it.

    Attributes:
        port (JudgePort): The judgevet ``SystemOnePort``.
        model (str): The model name sent with each call.
        facts (dict[str, object]): Judge facts for the artifact.
        stoppers (tuple[StopperProtocol, ...]): gepa-adk stoppers.
        spend (Callable[[], dict[str, object]] | None): The spend so far, or None.
        key_free (Callable[[str], None]): Raises if text holds the key.
    """

    port: JudgePort
    model: str
    facts: dict[str, object]
    stoppers: tuple[StopperProtocol, ...] = ()
    spend: Callable[[], dict[str, object]] | None = None
    key_free: Callable[[str], None] = _no_secret_key_free


def _judge_provider(environ: Mapping[str, str]) -> str:
    provider = environ.get("TYPEVET_WORDING_JUDGE_PROVIDER", "").strip() or "gemma"
    if provider not in _PROVIDERS:
        pytest.fail(f"TYPEVET_WORDING_JUDGE_PROVIDER must be one of {_PROVIDERS}")
    return provider


def _jev_api() -> ApiSettings:
    """Read judgevet's settings; refuse a run without a spend cap or a key.

    Returns:
        The Jev API settings.
    """
    api = Settings().api
    if api.spend_max_attempts is None:
        pytest.fail(
            "JEV_API__SPEND_MAX_ATTEMPTS must be set: the Jev evolution does not "
            "run without a spend cap"
        )
    if not api.has_key_source:
        pytest.fail("the Jev key is not set (JEV_API__KEY or TYPESAFE_API_KEY)")
    return api


def _spend(cap: SpendCap) -> dict[str, object]:
    limit = cap.max_attempts
    return {
        "attempts": cap.attempts,
        "input_tokens": cap.input_tokens,
        "max_attempts": limit,
        "max_input_tokens": cap.max_input_tokens,
        "attempts_exhausted": limit is not None and cap.attempts >= limit,
    }


def _jev_key(api: ApiSettings) -> str:
    """Resolve the Jev key for one use; no caller binds it to a name.

    Returns:
        The key value.
    """
    key = api.resolve_key()
    if key is None:
        pytest.fail("the Jev key source resolved to nothing")
    return key.get_secret_value()


@contextmanager
def _jev_judge(
    api: ApiSettings, environ: Mapping[str, str]
) -> Iterator[_EvolutionJudge]:
    """Build judgevet's HTTP adapter with one spend cap for the run.

    Yields:
        The Jev judge; the adapter closes on exit.
    """
    cap = api.spend_cap
    limit = api.spend_max_attempts
    if cap is None or limit is None:
        pytest.fail("the spend cap is not set")

    def spent(state: StopperState) -> bool:
        return cap.attempts >= limit

    def key_free(text: str) -> None:
        ensure_key_free(text, secrets=(_jev_key(api),))

    with HTTPSystemOneAdapter(
        api_key=_jev_key(api),
        base_url=api.base_url,
        default_model=api.default_model,
        timeout_seconds=api.timeout_seconds,
        retry=api.retry_policy,
        network=api.network_config,
        gateway=api.gateway_config,
        spend_cap=cap,
    ) as adapter:
        yield _EvolutionJudge(
            port=adapter,
            model=environ.get("TYPEVET_WORDING_JUDGE", "").strip() or api.default_model,
            facts={"base_url": api.base_url, "served_template": None},
            stoppers=(spent,),
            spend=lambda: _spend(cap),
            key_free=key_free,
        )


@contextmanager
def _gemma_judge(environ: Mapping[str, str]) -> Iterator[_EvolutionJudge]:
    """Open the llama.cpp text session for the Gemma judge.

    Yields:
        The Gemma judge.
    """
    judge = environ.get("TYPEVET_WORDING_JUDGE", _JUDGE)
    with _llama_text_session(judge, environ) as session:
        yield _EvolutionJudge(
            port=TypevetSystemOnePort(session.port),
            model=judge,
            facts={"served_template": session.served_template},
        )


@pytest.mark.live
def test_wording_evolution_live_artifact() -> None:
    """Evolve the is_scam wording with the chosen judge and write the artifact."""
    artifact_path = _required_path("TYPEVET_WORDING_ARTIFACT", new=True)
    environ = dict(os.environ)
    provider = _judge_provider(environ)
    evolution_inputs(environ)
    jev_api = _jev_api() if provider == "jev" else None
    checkpoint = _outside_repo(
        Path(
            environ.get("TYPEVET_WORDING_CHECKPOINT", "").strip()
            or artifact_path.with_suffix(".checkpoint.json")
        ).expanduser()
    )
    splits = load_splits(seed=0)
    validation = stratified_subset(
        splits.validation, _int_env("TYPEVET_WORDING_VALIDATION_ROWS", 200) or 200
    )
    train = train_subset(
        splits.train, _int_env("TYPEVET_WORDING_TRAIN_ROWS", DEFAULT_TRAIN_ROWS)
    )
    reflector = _reflector(environ)
    concurrency = _int_env("TYPEVET_WORDING_CONCURRENCY", None)
    started = time.monotonic()
    session = _gemma_judge(environ) if jev_api is None else _jev_judge(jev_api, environ)
    with session as judge:
        config = WordingRunConfig(
            reflector=reflector,
            judge_model=judge.model,
            max_iterations=_int_env("TYPEVET_WORDING_MAX_ITERATIONS", 10) or 10,
            reflection_minibatch_size=_int_env("TYPEVET_WORDING_MINIBATCH", 8),
            max_concurrent_evals=concurrency or _DEFAULT_CONCURRENCY[provider],
            checkpoint_path=checkpoint,
            resume=environ.get("TYPEVET_WORDING_RESUME", "") == "1",
            stop_callbacks=judge.stoppers,
        )
        timed = TimedJudgePort(judge.port)
        run = asyncio.run(
            evolve_from_env(
                timed, environ, train=train, validation=validation, config=config
            )
        )
        spend = None if judge.spend is None else judge.spend()
    artifact = evolution_artifact(
        run,
        config=config,
        train=train,
        validation=validation,
        call_records=timed.records,
    )
    summary = artifact["call_summary"]
    artifact["reflector_base"] = environ.get(
        "TYPEVET_WORDING_REFLECTOR_BASE", _router_v1(environ)
    )
    artifact["dataset_revision"] = PINNED_REVISION
    artifact["served_template"] = judge.facts["served_template"]
    artifact["judge_provider"] = provider
    artifact["judge_identity"] = {
        "requested_model": judge.model,
        "reported_models": summary["models"],
        **{k: v for k, v in judge.facts.items() if k != "served_template"},
    }
    artifact["stop_reason"] = run.result.stop_reason.value
    refusals = error_count(timed.records, _BUDGET_REFUSAL)
    artifact["budget_refusals"] = refusals
    artifact["valid"] = refusals == 0
    if spend is not None:
        artifact["spend"] = spend
    artifact["wall_seconds"] = round(time.monotonic() - started, 1)
    artifact["finished_utc"] = datetime.now(UTC).isoformat()
    judge.key_free(json.dumps(artifact, indent=2))
    write_receipt_exclusive(artifact_path, artifact)
    history = [
        (r.iteration_number, r.accepted, r.skip_reason)
        for r in run.result.iteration_history
    ]
    print(
        f"artifact {artifact_path} wall_seconds {artifact['wall_seconds']}\n"
        f"seed {run.seed_text!r}\nevolved {run.evolved_text!r}\n"
        f"valset_score {run.result.valset_score} iterations {history}\n"
        f"stop_reason {artifact['stop_reason']} spend {json.dumps(spend)}\n"
        f"judge {json.dumps(artifact['judge_identity'])}\n"
        f"calls {json.dumps(summary)}\n"
        f"budget_refusals {refusals} valid {artifact['valid']}"
    )
    assert refusals == 0, (
        f"{refusals} judge calls were refused by the spend cap and scored 0; "
        "the run is not a valid comparison (the artifact is kept)"
    )
    if provider == "jev":
        assert summary["failed"] < summary["calls"], "every Jev call failed"
    assert run.result.total_iterations >= 1


def _working_tree() -> WorkingTreeState:
    porcelain = os.environ.get("TYPEVET_GIT_STATUS_PORCELAIN")
    if porcelain is None:
        return WorkingTreeState(
            read_baseline_commit(_REPO_ROOT),
            True,
            ("dirty-unknown-without-porcelain",),
            "unknown",
        )
    return capture_working_tree_at_run_start(_REPO_ROOT, porcelain=porcelain)


@contextmanager
def _held_out_session(
    environ: Mapping[str, str], backend: str
) -> Iterator[tuple[JudgmentPort, str, httpx.Client, str]]:
    """Open the judgment session for ``backend``.

    Yields:
        The judgment port, the model id, the HTTP client and the template.
    """
    if backend == "vllm":
        with open_judgment(environ) as vllm:
            served = probe_vllm_template(vllm.client, vllm.model)
            require_native_template(served, environ)
            yield vllm.port, vllm.model, vllm.client, served.value
        return
    judge = environ.get("TYPEVET_WORDING_JUDGE", _JUDGE)
    with _llama_text_session(judge, environ) as llama:
        yield llama.port, llama.model, llama.client, llama.served_template


@pytest.mark.live
def test_wording_held_out_live_receipt() -> None:
    """Score seed and evolved wording once each on the held-out rows."""
    path = _required_path("TYPEVET_WORDING_HELD_OUT_RECEIPT", new=True)
    evolved_path = _required_path("TYPEVET_WORDING_EVOLVED_ARTIFACT", new=False)
    environ = dict(os.environ)
    backend = load_backend(environ)
    secret = load_vllm_settings(environ).api_key if backend == "vllm" else None
    weights = served_weights_pins(backend, environ)
    evolved = json.loads(evolved_path.read_text(encoding="utf-8"))
    assert evolved["seed_text"] == SEED_TEXT, "artifact seed is not the is_scam seed"
    seed, parts = held_out_parts(evolved, environ)
    splits = load_splits(seed=0)
    assert len(splits.held_out) == 158, "held-out row count drifted"
    tree = _working_tree()
    snapshot = snapshot_evaluated_inputs(
        prompts=prompt_specs(parts),
        code_paths={n: _WORDING_SRC / f"{n}.py" for n in ("held_out", "metrics")},
        fixture_paths={"evolution_artifact": evolved_path},
    )
    started = time.monotonic()
    with _held_out_session(environ, backend) as (port, model, client, template):
        facts = _server_facts(backend, client, model)
        start = begin_run_identity(
            repo_root=_REPO_ROOT,
            runtime=RuntimeBuild(model, template, str(facts["build_info"])),
            working_tree=tree,
        )
        timed = TimedJudgePort(TypevetSystemOnePort(port))
        run = score_held_out(
            timed,
            seed,
            PRIMARY_NOUL_NAME,
            evolved_text=parts,
            rows=HeldOutRows(splits.held_out, splits.prior_measured_ids),
            judge_model=model,
            failures=(ProviderError,),
        )
        run = replace(run, call_records=timed.records)
    identity = finalize_experiment_identity(
        run_start=start, evaluated=snapshot, arm_call_counts={"judgments": run.calls}
    )
    pins = {
        "finished_utc": datetime.now(UTC).isoformat(),
        "wall_seconds": round(time.monotonic() - started, 1),
        "dataset": "difraud/difraud sms",
        "dataset_revision": PINNED_REVISION,
        "split_seed": 0,
        "served_template": template,
        "evolution_artifact_sha256": hashlib.sha256(
            evolved_path.read_bytes()
        ).hexdigest(),
        "server": facts,
        **weights,
    }
    receipt = held_out_receipt(
        run,
        seed_text=parts.seed_text,
        evolved_text=parts.evolved_text,
        backend=backend,
        model=model,
        pins=pins,
        identity=identity.to_receipt_mapping(),
    )
    ensure_key_free(json.dumps(receipt, indent=2), secrets=(secret,))
    write_receipt_exclusive(path, receipt)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    print(
        f"receipt {path} sha256 {digest}\n"
        f"metrics {json.dumps(receipt['metrics'])}\n"
        f"verdict {json.dumps(receipt['verdict'])}\n"
        f"bootstrap {json.dumps(receipt['bootstrap'])}\n"
        f"reference_133 {json.dumps(receipt['reference_133'])}"
    )
    assert run.stopped is None, run.stopped
    assert len(run.pairs) == len(splits.held_out)


@pytest.mark.live
def test_wording_framing_smoke_live_receipt() -> None:
    """Score the seed wording on 10 validation rows and record each call."""
    path = _required_path("TYPEVET_WORDING_SMOKE_RECEIPT", new=True)
    environ = dict(os.environ)
    backend = load_backend(environ)
    secret = load_vllm_settings(environ).api_key if backend == "vllm" else None
    rows = stratified_subset(load_splits(seed=0).validation, _SMOKE_ROWS)
    started = time.monotonic()
    with _held_out_session(environ, backend) as (port, model, client, template):
        facts = _server_facts(backend, client, model)
        timed = TimedJudgePort(TypevetSystemOnePort(port))
        answers = [
            timed.system_one(r.example.text, {PRIMARY_NOUL_NAME: _SEED}, model)
            .nouls[PRIMARY_NOUL_NAME]
            .noul
            for r in rows
        ]
    receipt = {
        "issue": 327,
        "backend": backend,
        "model": model,
        "served_template": template,
        "split": "validation",
        "rows": [
            {"record_id": r.record_id, "label": r.example.label, "seed": p}
            for r, p in zip(rows, answers, strict=True)
        ],
        "per_call": [c.to_mapping() for c in timed.records],
        "call_summary": call_summary(timed.records),
        "wall_seconds": round(time.monotonic() - started, 1),
        "finished_utc": datetime.now(UTC).isoformat(),
        "server": facts,
    }
    ensure_key_free(json.dumps(receipt, indent=2), secrets=(secret,))
    write_receipt_exclusive(path, receipt)
    print(f"receipt {path}\n{json.dumps(receipt['call_summary'])}")
    assert all(r.example.split == "validation" for r in rows)
    assert len(timed.records) == _SMOKE_ROWS


_MAX_COMPARISON_SMOKE_ROWS = 6
_COMPARISON_JUDGE = {"jev": "jev", "llama_cpp": "gemma_llama_cpp", "vllm": "gemma_vllm"}


@dataclass(frozen=True)
class _ComparisonJudge:
    """One judge of the #252 comparison and what its receipt records.

    Attributes:
        port (JudgePort): The judgevet ``SystemOnePort``.
        model (str): The model name sent with each call.
        template (str): The probed served template, or ``jev_http``.
        facts (dict[str, object]): Server, quant or revision facts.
        spend (Callable[[], dict[str, object]] | None): The spend so far, or None.
        key_free (Callable[[str], None]): Raises if text holds a key.
    """

    port: JudgePort
    model: str
    template: str
    facts: dict[str, object]
    spend: Callable[[], dict[str, object]] | None
    key_free: Callable[[str], None]


def _comparison_rows(splits: DIFrauDSplits) -> tuple[HeldOutRows | ValidationRows, str]:
    """Return the held-out rows, or the validation rows of a smoke.

    Returns:
        The checked rows and their split name.
    """
    smoke = _int_env("TYPEVET_WORDING_COMPARISON_SMOKE_ROWS", None)
    if smoke is None:
        assert len(splits.held_out) == 158, "held-out row count drifted"
        return HeldOutRows(splits.held_out, splits.prior_measured_ids), "test"
    if not 1 <= smoke <= _MAX_COMPARISON_SMOKE_ROWS:
        pytest.fail(
            "TYPEVET_WORDING_COMPARISON_SMOKE_ROWS must be 1 to "
            f"{_MAX_COMPARISON_SMOKE_ROWS}"
        )
    return ValidationRows(stratified_subset(splits.validation, smoke)), "validation"


def _weights_facts(
    backend: str, client: httpx.Client, model: str, environ: Mapping[str, str]
) -> dict[str, object]:
    """Return the quant of a llama.cpp model or the vLLM weights revision.

    Returns:
        ``model_ftype`` and ``model_file`` for llama.cpp; the revision pin
        for vLLM.
    """
    if backend == "vllm":
        return dict(served_weights_pins(backend, environ))
    body = client.get("/props", params={"model": model}).raise_for_status().json()
    path = body.get("model_path")
    return {
        "model_ftype": body.get("model_ftype"),
        "model_file": None if path is None else Path(str(path)).name,
    }


@contextmanager
def _comparison_judge(
    backend: str, environ: Mapping[str, str]
) -> Iterator[_ComparisonJudge]:
    """Open the Jev or Gemma judge of the comparison.

    Yields:
        The judge.
    """
    if backend == "jev":
        with _jev_judge(_jev_api(), environ) as jev:
            yield _ComparisonJudge(
                jev.port,
                jev.model,
                "jev_http",
                dict(jev.facts),
                jev.spend,
                jev.key_free,
            )
        return
    secret = load_vllm_settings(environ).api_key if backend == "vllm" else None

    def key_free(text: str) -> None:
        ensure_key_free(text, secrets=(secret,))

    with _held_out_session(environ, backend) as (port, model, client, template):
        facts = {
            **_server_facts(backend, client, model),
            **_weights_facts(backend, client, model, environ),
        }
        yield _ComparisonJudge(
            TypevetSystemOnePort(port), model, template, facts, None, key_free
        )


@pytest.mark.live
def test_wording_comparison_live_receipt() -> None:
    """Score one judge's seed and own evolved wording for the #252 comparison."""
    path = _required_path("TYPEVET_WORDING_COMPARISON_RECEIPT", new=True)
    evolved_path = _required_path("TYPEVET_WORDING_EVOLVED_ARTIFACT", new=False)
    environ = dict(os.environ)
    provider = _judge_provider(environ)
    backend = "jev" if provider == "jev" else load_backend(environ)
    artifact = json.loads(evolved_path.read_text(encoding="utf-8"))
    seed = wording_seed(environ)
    parts = comparison_parts(artifact, seed, backend=backend)
    rows, split = _comparison_rows(load_splits(seed=0))
    tree = _working_tree()
    snapshot = snapshot_evaluated_inputs(
        prompts=prompt_specs(parts),
        code_paths={
            n: _WORDING_SRC / f"{n}.py" for n in ("held_out", "metrics", "comparison")
        },
        fixture_paths={"evolution_artifact": evolved_path},
    )
    started = time.monotonic()
    with _comparison_judge(backend, environ) as judge:
        build = str(judge.facts.get("build_info", "unknown"))
        start = begin_run_identity(
            repo_root=_REPO_ROOT,
            runtime=RuntimeBuild(judge.model, judge.template, build),
            working_tree=tree,
        )
        timed = TimedJudgePort(judge.port)
        run = score_held_out(
            timed,
            seed,
            PRIMARY_NOUL_NAME,
            evolved_text=parts,
            rows=rows,
            judge_model=judge.model,
            failures=(ProviderError,),
        )
        run = replace(run, call_records=timed.records)
        spend = None if judge.spend is None else judge.spend()
    identity = finalize_experiment_identity(
        run_start=start, evaluated=snapshot, arm_call_counts={"judgments": run.calls}
    )
    refusals = error_count(timed.records, _BUDGET_REFUSAL)
    pins = {
        "finished_utc": datetime.now(UTC).isoformat(),
        "wall_seconds": round(time.monotonic() - started, 1),
        "dataset": "difraud/difraud sms",
        "dataset_revision": PINNED_REVISION,
        "split_seed": 0,
        "served_template": None if backend == "jev" else judge.template,
        "evolution_artifact": evolved_path.name,
        "evolution_artifact_sha256": hashlib.sha256(
            evolved_path.read_bytes()
        ).hexdigest(),
        "server": judge.facts,
        "reported_models": call_summary(run.call_records)["models"],
        "budget_refusals": refusals,
        "spend": spend,
    }
    subject = ComparisonSubject(_COMPARISON_JUDGE[backend], backend, judge.model, split)
    receipt = comparison_receipt(
        run,
        subject,
        seed_text=parts.seed_text,
        evolved_text=parts.evolved_text,
        pins=pins,
        identity=identity.to_receipt_mapping(),
    )
    judge.key_free(json.dumps(receipt, indent=2))
    write_receipt_exclusive(path, receipt)
    print(
        f"receipt {path} sha256 {hashlib.sha256(path.read_bytes()).hexdigest()}\n"
        f"judge {subject.judge} split {split} rows {receipt['rows']}\n"
        f"metrics {json.dumps(receipt['metrics'])}\n"
        f"calls {json.dumps(receipt['call_summary'])}\n"
        f"spend {json.dumps(spend)} budget_refusals {refusals}"
    )
    assert refusals == 0, f"{refusals} calls were refused by the spend cap"
    assert run.stopped is None, run.stopped
    assert len(run.pairs) == len(rows.records)
    assert all(r.example.split == split for r in rows.records)
