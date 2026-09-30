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

    TYPEVET_BACKEND=vllm TYPEVET_VLLM_MODEL_REVISION=<sha> \
      TYPEVET_VLLM__BASE_URL=https://<pod>-8000.proxy.runpod.net \
      TYPEVET_VLLM__MODEL=google/gemma-4-31B-it \
      TYPEVET_WORDING_EVOLVED_ARTIFACT=$HOME/typevet-309/evolution.json \
      TYPEVET_WORDING_HELD_OUT_RECEIPT=evals/fixtures/difraud/receipts/wording_held_out_vllm.json \
      uv run pytest evals/tests/live/test_wording_evolution_live.py -m live -q -s \
      -k held_out
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
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest
from google.adk.models.lite_llm import LiteLlm
from judgevet.domain.questions import Noul
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
    WordingRunConfig,
    evolution_artifact,
    evolve_wording,
    held_out_receipt,
    score_held_out,
    stratified_subset,
    train_subset,
)
from typevet_evals.wording.calls import TimedJudgePort, call_summary
from typevet_evals.wording.held_out import DEFAULT_TRAIN_ROWS, HeldOutRows
from typevet_evals.wording.served import (
    TEXT_JUDGE,
    probe_llama_template,
    probe_vllm_template,
    require_native_template,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_WORDING_SRC = _REPO_ROOT / "evals" / "src" / "typevet_evals" / "wording"
_JUDGE = TEXT_JUDGE
_SMOKE_ROWS = 10
_REFLECTOR = "Qwen3.8-27B-UD-Q4_K_M"
_N_VOCAB = 262144
_TIMEOUT = 900.0
_SEED_TEXT = str(IS_SCAM_NOUL_SCHEMA["properties"][PRIMARY_NOUL_NAME]["instructions"])
_SEED = Noul(instructions=_SEED_TEXT)


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


@pytest.mark.live
def test_wording_evolution_live_artifact() -> None:
    """Evolve the is_scam wording on the local router and write the artifact."""
    artifact_path = _required_path("TYPEVET_WORDING_ARTIFACT", new=True)
    environ = dict(os.environ)
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
    judge = environ.get("TYPEVET_WORDING_JUDGE", _JUDGE)
    reflector = _reflector(environ)
    config = WordingRunConfig(
        reflector=reflector,
        judge_model=judge,
        max_iterations=_int_env("TYPEVET_WORDING_MAX_ITERATIONS", 10) or 10,
        reflection_minibatch_size=_int_env("TYPEVET_WORDING_MINIBATCH", 8),
        checkpoint_path=checkpoint,
        resume=environ.get("TYPEVET_WORDING_RESUME", "") == "1",
    )
    started = time.monotonic()
    with _llama_text_session(judge, environ) as session:
        timed = TimedJudgePort(TypevetSystemOnePort(session.port))
        run = asyncio.run(
            evolve_wording(
                port=timed,
                seed=_SEED,
                key=PRIMARY_NOUL_NAME,
                train=train,
                validation=validation,
                config=config,
            )
        )
    artifact = evolution_artifact(
        run,
        config=config,
        train=train,
        validation=validation,
        call_records=timed.records,
    )
    artifact["reflector_base"] = environ.get(
        "TYPEVET_WORDING_REFLECTOR_BASE", _router_v1(environ)
    )
    artifact["dataset_revision"] = PINNED_REVISION
    artifact["served_template"] = session.served_template
    artifact["wall_seconds"] = round(time.monotonic() - started, 1)
    artifact["finished_utc"] = datetime.now(UTC).isoformat()
    text = json.dumps(artifact, indent=2)
    ensure_key_free(text, secrets=())
    write_receipt_exclusive(artifact_path, artifact)
    history = [
        (r.iteration_number, r.accepted, r.skip_reason)
        for r in run.result.iteration_history
    ]
    print(
        f"artifact {artifact_path} wall_seconds {artifact['wall_seconds']}\n"
        f"seed {run.seed_text!r}\nevolved {run.evolved_text!r}\n"
        f"valset_score {run.result.valset_score} iterations {history}"
    )
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
    assert evolved["seed_text"] == _SEED_TEXT, "artifact seed is not the is_scam seed"
    evolved_text = str(evolved["evolved_text"])
    splits = load_splits(seed=0)
    assert len(splits.held_out) == 158, "held-out row count drifted"
    tree = _working_tree()
    snapshot = snapshot_evaluated_inputs(
        prompts=(
            PromptSpec("seed", ("true", "false"), _SEED_TEXT, {}),
            PromptSpec("evolved", ("true", "false"), evolved_text, {}),
        ),
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
            _SEED,
            PRIMARY_NOUL_NAME,
            evolved_text=evolved_text,
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
        seed_text=_SEED_TEXT,
        evolved_text=evolved_text,
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
