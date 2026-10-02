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
- ``TYPEVET_WORDING_JUDGE_PROVIDER``: ``gemma`` (default, the session above),
  ``jev`` (#328) or ``ollama`` (#333). ``TYPEVET_WORDING_CONCURRENCY`` sets
  gepa-adk's ``max_concurrent_evals`` (default 5 for Gemma, 1 for Jev and
  Ollama).

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

**Seed (#369).** ``TYPEVET_WORDING_SEED`` selects ``difraud`` (default,
the DIFrauD ``is_scam`` ``Noul`` above) or ``pubmedqa``. ``pubmedqa`` uses the
``PUBMEDQA_SEED`` ``Choice`` under the question name ``answer``, with the
parts ``instructions``, ``criteria_yes``, ``criteria_no`` and
``criteria_maybe``. Its rows are a balanced PubMedQA ``pqa_labeled`` pool,
cut into 180 train, 45 validation and 105 held-out rows (60, 15 and 35
per label, seed 0). ``TYPEVET_WORDING_TRAIN_ROWS`` and
``TYPEVET_WORDING_VALIDATION_ROWS`` override the first two. The PubMedQA card
gives about 110 ``maybe`` rows, so the DIFrauD sizes do not fit. The cut
fails before any judge call when the pool is too small.
``TYPEVET_WORDING_SEED_CRITERIA`` applies to ``difraud`` only. The
comparison test refuses ``pubmedqa``. No live run has used ``pubmedqa``.

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

**Ollama judge** (``TYPEVET_WORDING_JUDGE_PROVIDER=ollama``, #333). The judge
is judgevet's ``HTTPSystemOneAdapter`` at ``TYPEVET_OLLAMA_BASE`` (default
``http://localhost:11434``) with a placeholder key (judgevet #267, #268).
The model is ``TYPEVET_WORDING_JUDGE`` (default ``nimble``).
``TYPEVET_OLLAMA_TIMEOUT`` sets the read timeout (600 seconds). There is no
spend cap and no key check. Before the first judge call, the test reads the
Ollama version from ``/api/version`` and the model digest from ``/api/tags``.
It fails when ``/api/tags`` does not list the model. The artifact records
``judge_provider`` ``ollama`` and adds ``base_url``, ``ollama_version`` and
``model_digest`` to ``judge_identity``. With ``ollama``, the held-out and
comparison tests score with the same judge and need an artifact whose
``judge_provider`` is ``ollama``. Their receipts add the same block as the
``judge_identity`` pin and record ``backend`` ``ollama``.

**Held-out check** (``TYPEVET_WORDING_HELD_OUT_RECEIPT`` names a new receipt;
``TYPEVET_WORDING_EVOLVED_ARTIFACT`` names the evolution artifact). Scores
the seed and evolved wording once each on all 158 held-out rows (316 calls)
and writes a key-free receipt: metrics for both, paired bootstrap 95%
intervals (2,000 resamples, seed 0; context only), the pre-registered
verdict, the #133 comparison, pins and call counts. ``TYPEVET_BACKEND``
selects the backend as in ``open_judgment``: ``llama_cpp`` (default; the
same text-only session as the evolution) or ``vllm``
(``TYPEVET_VLLM__*`` and ``TYPEVET_VLLM_MODEL_REVISION``).
``TYPEVET_WORDING_JUDGE_PROVIDER=ollama`` uses the Ollama judge instead.
``TYPEVET_GIT_STATUS_PORCELAIN`` carries the porcelain status text.

**Jev vs Gemma comparison (#329, parent #252)**
(``TYPEVET_WORDING_COMPARISON_RECEIPT`` names a new receipt;
``TYPEVET_WORDING_EVOLVED_ARTIFACT`` names the judge's own evolution
artifact). One judge scores the seed wording and its own evolved wording once
each on the 158 held-out rows. ``TYPEVET_WORDING_JUDGE_PROVIDER=jev`` uses
the Jev judge above (spend cap required before any call); ``ollama`` uses
the Ollama judge (#333); ``gemma`` uses the ``TYPEVET_BACKEND`` session
(``llama_cpp`` or ``vllm``). The artifact's ``judge_provider`` must be the
judge's (``jev`` for Jev, ``ollama`` for Ollama, ``gemma`` for both Gemma
backends), valid and free of budget refusals. The receipt holds per arm
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
from judgevet.domain.questions import Choice, Noul
from judgevet.domain.spend import SpendCap
from judgevet.providers import ProviderError

from typevet.adapters.inbound.backend_settings import (
    load_backend,
    load_vllm_settings,
    open_judgment,
)
from typevet.adapters.inbound.fake_backend import require_live_session
from typevet.adapters.inbound.judgevet import TypevetSystemOnePort
from typevet.adapters.inbound.settings import load_llama_settings
from typevet.adapters.outbound.judgment_scoring import ScoringJudgmentAdapter
from typevet.adapters.outbound.llama_cpp.scoring import LlamaCppCandidateScoringAdapter
from typevet.ports.judgment import JudgmentPort
from typevet_evals.datasets.difraud import (
    IS_SCAM_NOUL_SCHEMA,
    PINNED_REVISION,
    PRIMARY_NOUL_NAME,
    DIFrauDSplits,
    load_splits,
)
from typevet_evals.datasets.pubmedqa import (
    CHOICE_LABELS,
    PRIMARY_CHOICE_NAME,
    download_labeled_jsonl,
    iter_labeled_rows,
    map_examples,
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
from typevet_evals.wording.calls import (
    CallRecord,
    TimedJudgePort,
    call_summary,
    error_count,
)
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
    part_table,
    question_mapping,
    seed_mapping,
)
from typevet_evals.wording.rows import (
    AnyRow,
    WordingSplits,
    as_row,
    pubmedqa_splits,
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
DIFRAUD: Final[str] = "difraud"
PUBMEDQA: Final[str] = "pubmedqa"
HELD_OUT_ROWS: Final[int] = 158
PUBMEDQA_SIZES: Final[tuple[int, int, int]] = (180, 45, 105)
PUBMEDQA_SEED: Final[Choice] = Choice(
    instructions=(
        "Given the biomedical question and the abstract contexts, "
        "is the answer yes, no or maybe?"
    ),
    criteria={
        "yes": "The contexts support a yes answer.",
        "no": "The contexts support a no answer.",
        "maybe": "The contexts do not settle the answer.",
    },
)


_DATASET_PINS: Final[Mapping[str, tuple[str, str | None]]] = MappingProxyType(
    {
        DIFRAUD: ("difraud/difraud sms", PINNED_REVISION),
        PUBMEDQA: ("qiaojin/PubMedQA pqa_labeled", None),
    }
)


def wording_seed_name(environ: Mapping[str, str]) -> str:
    """Read ``TYPEVET_WORDING_SEED``: ``difraud`` (default) or ``pubmedqa`` (#369).

    Args:
        environ: The environment.

    Returns:
        The seed name.

    Raises:
        ValueError: If the value is another name; the message does not repeat it.
    """
    name = environ.get("TYPEVET_WORDING_SEED", "").strip() or DIFRAUD
    if name not in (DIFRAUD, PUBMEDQA):
        raise ValueError("TYPEVET_WORDING_SEED must be difraud or pubmedqa")
    return name


def wording_question_name(environ: Mapping[str, str]) -> str:
    """Return the question name of the selected seed.

    Args:
        environ: The environment.

    Returns:
        ``is_scam`` for ``difraud``, ``answer`` for ``pubmedqa``.
    """
    if wording_seed_name(environ) == PUBMEDQA:
        return PRIMARY_CHOICE_NAME
    return PRIMARY_NOUL_NAME


def wording_components(environ: Mapping[str, str]) -> tuple[str, ...]:
    """Read ``TYPEVET_WORDING_COMPONENTS``: the part names that evolve (#365).

    Args:
        environ: The environment.

    Returns:
        The comma-separated names, in order; ``("instructions",)`` when unset.

    Raises:
        ValueError: If a name is empty or not a part name of the selected
            seed type: a ``Noul`` part for ``difraud``, a part of
            ``PUBMEDQA_SEED`` for ``pubmedqa``. The message does not repeat
            the value.
    """
    raw = environ.get("TYPEVET_WORDING_COMPONENTS", "").strip()
    if not raw:
        return (INSTRUCTIONS,)
    known = tuple(question_mapping(PUBMEDQA_SEED))
    if wording_seed_name(environ) == DIFRAUD:
        known = PART_NAMES
    names = tuple(name.strip() for name in raw.split(","))
    if not all(name in known for name in names):
        msg = f"TYPEVET_WORDING_COMPONENTS names a part outside {', '.join(known)}"
        raise ValueError(msg)
    return names


def wording_seed(environ: Mapping[str, str]) -> Noul | Choice:
    """Build the seed; ``TYPEVET_WORDING_SEED_CRITERIA=1`` adds the criteria.

    Args:
        environ: The environment.

    Returns:
        The #252 seed ``Noul``, with ``SEED_CRITERIA`` when the knob is ``1``;
        ``PUBMEDQA_SEED`` when ``TYPEVET_WORDING_SEED`` is ``pubmedqa``.

    Raises:
        ValueError: If the criteria knob is set to a value other than ``0``
            or ``1``, or to ``1`` with the ``pubmedqa`` seed.
    """
    raw = environ.get("TYPEVET_WORDING_SEED_CRITERIA", "").strip()
    if raw not in {"", "0", "1"}:
        raise ValueError("TYPEVET_WORDING_SEED_CRITERIA must be 0 or 1")
    if wording_seed_name(environ) == PUBMEDQA:
        if raw == "1":
            raise ValueError("TYPEVET_WORDING_SEED_CRITERIA applies to difraud only")
        return PUBMEDQA_SEED
    criteria = dict(SEED_CRITERIA) if raw == "1" else None
    return Noul(instructions=SEED_TEXT, criteria=criteria)


def wording_split_sizes(environ: Mapping[str, str]) -> tuple[int, int, int]:
    """Return the train, validation and held-out row counts of a run.

    Args:
        environ: The environment.

    Returns:
        ``TYPEVET_WORDING_TRAIN_ROWS``, ``TYPEVET_WORDING_VALIDATION_ROWS`` and
        the held-out count. The defaults are the DIFrauD sizes (1000, 200,
        ``HELD_OUT_ROWS``), or ``PUBMEDQA_SIZES`` for the ``pubmedqa`` seed.
    """
    defaults = (DEFAULT_TRAIN_ROWS, 200, HELD_OUT_ROWS)
    if wording_seed_name(environ) == PUBMEDQA:
        defaults = PUBMEDQA_SIZES
    train = environ.get("TYPEVET_WORDING_TRAIN_ROWS", "").strip()
    validation = environ.get("TYPEVET_WORDING_VALIDATION_ROWS", "").strip()
    return (
        int(train) if train else defaults[0],
        int(validation) if validation else defaults[1],
        defaults[2],
    )


def pubmedqa_wording_splits(
    environ: Mapping[str, str], jsonl_text: str, *, held_out: int | None = None
) -> WordingSplits:
    """Cut the PubMedQA ``pqa_labeled`` JSONL into balanced run splits.

    Args:
        environ: The environment; it sets the train and validation sizes.
        jsonl_text: The ``pqa_labeled`` JSONL text.
        held_out: Held-out rows; None takes the count of
            ``wording_split_sizes``.

    Returns:
        ``pubmedqa_splits`` of the mapped rows, seed 0.
    """
    train, validation, default_held_out = wording_split_sizes(environ)
    examples = map_examples(iter_labeled_rows(jsonl_text))
    return pubmedqa_splits(
        examples,
        train=train,
        validation=validation,
        held_out=default_held_out if held_out is None else held_out,
    )


def evolution_inputs(
    environ: Mapping[str, str],
) -> tuple[Noul | Choice, tuple[str, ...]]:
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
    train: Sequence[AnyRow],
    validation: Sequence[AnyRow],
    config: WordingRunConfig,
) -> WordingRun:
    """Evolve the parts the knobs select from the seed the knobs build.

    Args:
        port: The judge port.
        environ: The environment.
        train: The train records or rows.
        validation: The validation records or rows.
        config: The run settings.

    Returns:
        The run; its ``components``, ``seed_parts``, ``part_table`` and
        ``evolved_parts`` reach the artifact.
    """
    seed, components = evolution_inputs(environ)
    return await evolve_wording(
        port=port,
        seed=seed,
        question_name=wording_question_name(environ),
        train=train,
        validation=validation,
        config=config,
        components=components,
    )


def held_out_parts(
    artifact: Mapping[str, Any], environ: Mapping[str, str]
) -> tuple[Noul | Choice, WordingParts]:
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
    artifact: Mapping[str, Any], seed: Noul | Choice, *, backend: str
) -> WordingParts:
    """Return the parts of the comparison judge's own evolution.

    Args:
        artifact: The judge's evolution artifact.
        seed: The seed ``Noul``.
        backend: ``jev``, ``llama_cpp``, ``vllm`` or ``ollama``.

    Returns:
        ``evolved_parts_for`` over the seed's parts.
    """
    return evolved_parts_for(artifact, backend=backend, seed_parts=seed_mapping(seed))


def _criteria(
    mapping: Mapping[str, str], table: Mapping[str, str | int] | None
) -> dict[str, str]:
    if table is not None:
        return {str(label): mapping[name] for name, label in table.items()}
    if CRITERIA_TRUE not in mapping:
        return {}
    return {"true": mapping[CRITERIA_TRUE], "false": mapping[CRITERIA_FALSE]}


def prompt_specs(
    parts: WordingParts, seed: Choice | None = None
) -> tuple[PromptSpec, PromptSpec]:
    """Return the identity prompts of both arms, with their criteria.

    Args:
        parts: The run's parts.
        seed: The ``Choice`` seed, whose labels and ``part_table`` name the
            criteria; None for the DIFrauD ``Noul``.

    Returns:
        The ``seed`` and ``evolved`` prompts; no criteria for a seed without.
    """
    labels = _LABELS if seed is None else CHOICE_LABELS
    table = None if seed is None else part_table(seed)
    return (
        PromptSpec("seed", labels, parts.seed_text, _criteria(parts.seed, table)),
        PromptSpec(
            "evolved", labels, parts.evolved_text, _criteria(parts.evolved, table)
        ),
    )


def _choice_or_none(seed: Noul | Choice) -> Choice | None:
    return seed if isinstance(seed, Choice) else None


def _run_rows(
    environ: Mapping[str, str],
) -> tuple[Sequence[AnyRow], Sequence[AnyRow], HeldOutRows]:
    """Load the train, validation and held-out rows of the selected seed.

    Returns:
        The train and validation rows and the checked held-out rows.
    """
    if wording_seed_name(environ) == PUBMEDQA:
        splits = pubmedqa_wording_splits(environ, download_labeled_jsonl())
        held = HeldOutRows(splits.held_out, frozenset())
        return splits.train, splits.validation, held
    difraud = load_splits(seed=0)
    train_rows, validation_rows, _ = wording_split_sizes(environ)
    validation = stratified_subset(difraud.validation, validation_rows or 200)
    train = train_subset(difraud.train, train_rows)
    held = HeldOutRows(difraud.held_out, difraud.prior_measured_ids)
    return train, validation, held


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


_PROVIDERS = ("gemma", "jev", "ollama")
_DEFAULT_CONCURRENCY = {"gemma": 5, "jev": 1, "ollama": 1}
_OLLAMA_BASE = "http://localhost:11434"
OLLAMA_PLACEHOLDER_KEY: Final = "ollama-no-key"
_DEFAULT_JUDGE: Final[Mapping[str, str]] = MappingProxyType(
    {"gemma": _JUDGE, "ollama": "nimble"}
)
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


def judge_name(environ: Mapping[str, str], provider: str) -> str:
    """Return the judge model of a Gemma or Ollama run.

    Args:
        environ: The process environment.
        provider: ``gemma`` or ``ollama``.

    Returns:
        ``TYPEVET_WORDING_JUDGE``, or the provider's default when it is unset
        or blank: the Gemma pin for ``gemma`` and ``nimble`` for ``ollama``.
    """
    return environ.get("TYPEVET_WORDING_JUDGE", "").strip() or _DEFAULT_JUDGE[provider]


def ollama_identity(
    get: Callable[[str], Mapping[str, Any]], base_url: str, model: str
) -> dict[str, object]:
    """Read the Ollama version and the judge model's digest.

    Args:
        get: Returns the JSON body of one GET request to a URL.
        base_url: The Ollama server, for example ``http://localhost:11434``.
        model: The judge model; a name without a tag also matches ``:latest``.

    Returns:
        ``ollama_version`` from ``/api/version`` and ``model_digest`` from
        ``/api/tags``.

    Raises:
        ValueError: When ``/api/tags`` does not list the model. The message
            does not hold the listed names.
    """
    base = base_url.rstrip("/")
    version = get(f"{base}/api/version")["version"]
    names = {model, f"{model}:latest"}
    for entry in get(f"{base}/api/tags").get("models", ()):
        if entry.get("name") in names:
            return {"ollama_version": version, "model_digest": entry["digest"]}
    msg = "the judge model is not listed by Ollama /api/tags"
    raise ValueError(msg)


def _http_json(url: str) -> Mapping[str, Any]:
    return httpx.get(url, timeout=30.0).raise_for_status().json()


@contextmanager
def _ollama_judge(
    environ: Mapping[str, str],
    *,
    adapter: Callable[..., Any] = HTTPSystemOneAdapter,
    get: Callable[[str], Mapping[str, Any]] = _http_json,
) -> Iterator[_EvolutionJudge]:
    """Build judgevet's HTTP adapter against a local Ollama (judgevet #267).

    Args:
        environ: The process environment.
        adapter: The adapter class; tests pass a recording stand-in.
        get: The JSON GET used for the identity reads.

    Yields:
        The Ollama judge, with no spend cap and a placeholder key; the
        adapter closes on exit.
    """
    base = environ.get("TYPEVET_OLLAMA_BASE", "").strip().rstrip("/") or _OLLAMA_BASE
    model = judge_name(environ, "ollama")
    identity = ollama_identity(get, base, model)
    timeout = environ.get("TYPEVET_OLLAMA_TIMEOUT", "").strip() or "600"
    with adapter(
        api_key=OLLAMA_PLACEHOLDER_KEY,
        base_url=base,
        default_model=model,
        timeout_seconds=float(timeout),
    ) as port:
        yield _EvolutionJudge(
            port=port,
            model=model,
            facts={"base_url": base, "served_template": None, **identity},
        )


def judge_identity(
    model: str, facts: Mapping[str, object], reported_models: Sequence[str]
) -> dict[str, object]:
    """Return the ``judge_identity`` block of a judge.

    Args:
        model: The requested model.
        facts: The judge facts; ``served_template`` is left out.
        reported_models: The model ids the calls reported.

    Returns:
        ``requested_model``, ``reported_models`` and the other facts, for
        example ``base_url``, ``ollama_version`` and ``model_digest``.
    """
    return {
        "requested_model": model,
        "reported_models": list(reported_models),
        **{k: v for k, v in facts.items() if k != "served_template"},
    }


def judge_pins(
    backend: str,
    model: str,
    facts: Mapping[str, object],
    records: Sequence[CallRecord],
) -> dict[str, object]:
    """Return the identity pins a held-out or comparison receipt adds.

    Args:
        backend: The receipt's backend.
        model: The requested model.
        facts: The judge facts.
        records: The calls of the run.

    Returns:
        A ``judge_identity`` pin for ``ollama``; no pin for another backend,
        so its receipts keep their keys.
    """
    if backend != "ollama":
        return {}
    reported = call_summary(records)["models"]
    return {"judge_identity": judge_identity(model, facts, reported)}


def runtime_build(facts: Mapping[str, object]) -> str:
    """Return the server build for the run identity.

    Args:
        facts: The judge or server facts.

    Returns:
        ``build_info`` when the facts hold it, else ``ollama_version``, else
        ``unknown``.
    """
    if "build_info" in facts:
        return str(facts["build_info"])
    return str(facts.get("ollama_version", "unknown"))


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
    judge = judge_name(environ, "gemma")
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
    train, validation, _ = _run_rows(environ)
    reflector = _reflector(environ)
    concurrency = _int_env("TYPEVET_WORDING_CONCURRENCY", None)
    started = time.monotonic()
    session = (
        _jev_judge(jev_api, environ)
        if jev_api is not None
        else _ollama_judge(environ)
        if provider == "ollama"
        else _gemma_judge(environ)
    )
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
    artifact["dataset_revision"] = _DATASET_PINS[wording_seed_name(environ)][1]
    if wording_seed_name(environ) != DIFRAUD:
        artifact["dataset"] = _DATASET_PINS[wording_seed_name(environ)][0]
    artifact["served_template"] = judge.facts["served_template"]
    artifact["judge_provider"] = provider
    artifact["judge_identity"] = judge_identity(
        judge.model, judge.facts, summary["models"]
    )
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
    if provider != "gemma":
        assert summary["failed"] < summary["calls"], f"every {provider} call failed"
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
        with open_judgment(environ) as opened:
            vllm = require_live_session(opened)
            served = probe_vllm_template(vllm.client, vllm.model)
            require_native_template(served, environ)
            yield vllm.port, vllm.model, vllm.client, served.value
        return
    judge = judge_name(environ, "gemma")
    with _llama_text_session(judge, environ) as llama:
        yield llama.port, llama.model, llama.client, llama.served_template


@contextmanager
def _held_out_judge(
    environ: Mapping[str, str], backend: str
) -> Iterator[tuple[JudgePort, str, str, dict[str, object]]]:
    """Open the held-out judge for ``backend``.

    Yields:
        The ``SystemOnePort``, the model id, the served template
        (``ollama_http`` for Ollama) and the server facts.
    """
    if backend == "ollama":
        with _ollama_judge(environ) as ollama:
            yield ollama.port, ollama.model, "ollama_http", dict(ollama.facts)
        return
    with _held_out_session(environ, backend) as (port, model, client, template):
        facts = _server_facts(backend, client, model)
        yield TypevetSystemOnePort(port), model, template, facts


@pytest.mark.live
def test_wording_held_out_live_receipt() -> None:
    """Score seed and evolved wording once each on the held-out rows."""
    path = _required_path("TYPEVET_WORDING_HELD_OUT_RECEIPT", new=True)
    evolved_path = _required_path("TYPEVET_WORDING_EVOLVED_ARTIFACT", new=False)
    environ = dict(os.environ)
    ollama = _judge_provider(environ) == "ollama"
    backend = "ollama" if ollama else load_backend(environ)
    secret = load_vllm_settings(environ).api_key if backend == "vllm" else None
    weights = {} if ollama else served_weights_pins(backend, environ)
    evolved = json.loads(evolved_path.read_text(encoding="utf-8"))
    expected_seed = wording_seed(environ).instructions
    assert evolved["seed_text"] == expected_seed, "artifact seed is not the seed"
    if ollama:
        assert evolved.get("judge_provider") == "ollama", "not an Ollama artifact"
    seed, parts = held_out_parts(evolved, environ)
    _, _, held = _run_rows(environ)
    expected_rows = wording_split_sizes(environ)[2]
    assert len(held.records) == expected_rows, "held-out row count drifted"
    tree = _working_tree()
    snapshot = snapshot_evaluated_inputs(
        prompts=prompt_specs(parts, _choice_or_none(seed)),
        code_paths={n: _WORDING_SRC / f"{n}.py" for n in ("held_out", "metrics")},
        fixture_paths={"evolution_artifact": evolved_path},
    )
    started = time.monotonic()
    with _held_out_judge(environ, backend) as (port, model, template, facts):
        start = begin_run_identity(
            repo_root=_REPO_ROOT,
            runtime=RuntimeBuild(model, template, runtime_build(facts)),
            working_tree=tree,
        )
        timed = TimedJudgePort(port)
        run = score_held_out(
            timed,
            seed,
            wording_question_name(environ),
            evolved=parts,
            rows=held,
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
        "dataset": _DATASET_PINS[wording_seed_name(environ)][0],
        "dataset_revision": _DATASET_PINS[wording_seed_name(environ)][1],
        "split_seed": 0,
        "served_template": None if ollama else template,
        "evolution_artifact_sha256": hashlib.sha256(
            evolved_path.read_bytes()
        ).hexdigest(),
        "server": facts,
        **weights,
        **judge_pins(backend, model, facts, run.call_records),
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
    assert run.rows == len(held.records)


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
_COMPARISON_JUDGE = {
    "jev": "jev",
    "llama_cpp": "gemma_llama_cpp",
    "vllm": "gemma_vllm",
    "ollama": "ollama",
}
_HTTP_BACKENDS = ("jev", "ollama")


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
    """Open the Jev, Ollama or Gemma judge of the comparison.

    Yields:
        The judge.
    """
    if backend in _HTTP_BACKENDS:
        session = (
            _jev_judge(_jev_api(), environ)
            if backend == "jev"
            else _ollama_judge(environ)
        )
        with session as http:
            yield _ComparisonJudge(
                http.port,
                http.model,
                f"{backend}_http",
                dict(http.facts),
                http.spend,
                http.key_free,
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
    if wording_seed_name(environ) != DIFRAUD:
        pytest.fail("the comparison runs on the difraud seed only")
    provider = _judge_provider(environ)
    backend = provider if provider in _HTTP_BACKENDS else load_backend(environ)
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
        build = runtime_build(judge.facts)
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
            evolved=parts,
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
        "served_template": None if backend in _HTTP_BACKENDS else judge.template,
        "evolution_artifact": evolved_path.name,
        "evolution_artifact_sha256": hashlib.sha256(
            evolved_path.read_bytes()
        ).hexdigest(),
        "server": judge.facts,
        "reported_models": call_summary(run.call_records)["models"],
        "budget_refusals": refusals,
        "spend": spend,
        **judge_pins(backend, judge.model, judge.facts, run.call_records),
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
    assert all(as_row(r).split == split for r in rows.records)
