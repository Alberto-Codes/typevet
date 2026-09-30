r"""Opt-in live CEDAR signature-match run with a key-free receipt (#319).

Judges the balanced CEDAR slice (60 genuine-genuine, 60 genuine-skilled and
60 genuine-random pairs, seed 0) once on one backend. Each pair is one
two-image judgment with three typed questions. The run stops at the first
backend failure and records it.

The test skips unless ``TYPEVET_SIGNATURE_MATCH_RECEIPT`` names the receipt
file. It fails when ``TYPEVET_REQUIRE_LIVE`` is truthy and that variable is
missing. The receipt path must not exist; the test fails before any network
call when it does. The CEDAR archive comes from the loader cache
(``TYPEVET_CEDAR_CACHE``, else ``~/.cache/typevet/cedar``); the loader
downloads it only when the cache has no verified copy. ``TYPEVET_BACKEND``
selects the backend, as in ``open_judgment``:

- ``llama_cpp`` (default) reads ``TYPEVET_LLAMA__*``. The multimodal model
  defaults to ``gemma-4-31b-kv9-q4km-mm`` and the timeout to 900 seconds.
  The scoring calls send a request once more when the router closes a
  reused connection before a response (#305).
- ``vllm`` reads ``TYPEVET_VLLM__BASE_URL``, ``TYPEVET_VLLM__MODEL``,
  ``TYPEVET_VLLM__API_KEY`` and ``TYPEVET_VLLM__USER_AGENT``. It also
  requires ``TYPEVET_VLLM_MODEL_REVISION``, the served weights revision. The
  receipt pins record it. The test fails before any network call when it is
  missing.

``TYPEVET_SIGNATURE_MATCH_PER_KIND`` sets a smaller slice for a smoke run.
``TYPEVET_IMAGE_CONCURRENCY`` sets how many judgments run at one time
(default 1, one at a time). The receipt pins record it.
The receipt `throughput` block records the rates, the latency percentiles,
the discarded count and, on vLLM, the `/metrics` deltas over the run (#335).
A full run checks the slice ids against
``evals/fixtures/cedar/default_slice_ids.txt``.
``TYPEVET_GIT_STATUS_PORCELAIN`` carries the porcelain status text for the
working-tree fingerprint, as in the CORD smoke. The receipt holds pair ids,
typed answers, metrics by pair kind and pins. It holds no image bytes, no
key and no auth header. The ``same_writer`` probability is model
confidence, not a match percentage or a forensic score.

Examples:
    ```bash
    TYPEVET_SIGNATURE_MATCH_RECEIPT=evals/fixtures/cedar/receipts/signature_match_llama_cpp.json \
      uv run pytest evals/tests/live/test_signature_match_live.py -m live -q -s

    TYPEVET_BACKEND=vllm \
      TYPEVET_VLLM_MODEL_REVISION=<hf-commit-sha> \
      TYPEVET_VLLM__BASE_URL=https://<pod>-8000.proxy.runpod.net \
      TYPEVET_VLLM__MODEL=google/gemma-4-31B-it \
      TYPEVET_SIGNATURE_MATCH_RECEIPT=evals/fixtures/cedar/receipts/signature_match_vllm.json \
      uv run pytest evals/tests/live/test_signature_match_live.py -m live -q -s
    ```

See Also:
    - [typevet_evals.signature_match.runner][]: run, metrics and receipt
    - [typevet_evals.datasets.cedar][]: pinned CEDAR archive and the slice
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from typevet.adapters.inbound.backend_settings import (
    load_backend,
    load_vllm_settings,
    open_judgment,
)
from typevet.adapters.inbound.settings import load_llama_settings
from typevet.adapters.outbound.llama_cpp.gemma_native_vision_factory import (
    GemmaNativeVisionSession,
    open_gemma_native_vision_judgment,
)
from typevet.adapters.outbound.vllm.judgment_factory import VllmJudgmentSession
from typevet.domain import Choice, Noul, Score
from typevet_evals.datasets.cedar import (
    ARCHIVE_SHA256,
    DEFAULT_PER_KIND,
    DEFAULT_SEED,
    CedarPair,
    fetch_cedar_archive,
    read_members,
    select_balanced_slice,
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
from typevet_evals.face_match import (
    ensure_key_free,
    image_concurrency,
    served_weights_pins,
)
from typevet_evals.runner.live_gate import require_live_enabled
from typevet_evals.serving_metrics import read_metrics
from typevet_evals.signature_match import (
    SignatureMatchRequest,
    build_signature_match_receipt,
    build_signature_match_request,
    run_signature_match,
    signature_match_questions,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SIGNATURE_SRC = _REPO_ROOT / "evals" / "src" / "typevet_evals" / "signature_match"
_SLICE_IDS = _REPO_ROOT / "evals" / "fixtures" / "cedar" / "default_slice_ids.txt"
_POOL_SRC = _REPO_ROOT / "evals" / "src" / "typevet_evals" / "face_match" / "pool.py"
_SERVING_METRICS_SRC = (
    _REPO_ROOT / "evals" / "src" / "typevet_evals" / "serving_metrics.py"
)
_RECEIPT_ENV = "TYPEVET_SIGNATURE_MATCH_RECEIPT"
_PER_KIND_ENV = "TYPEVET_SIGNATURE_MATCH_PER_KIND"
_LLAMA_MODEL = "gemma-4-31b-kv9-q4km-mm"
_LLAMA_TIMEOUT = "900"


def _receipt_path() -> Path:
    raw = os.environ.get(_RECEIPT_ENV, "").strip()
    if not raw:
        reason = f"{_RECEIPT_ENV} not set"
        if require_live_enabled():
            pytest.fail(reason)
        pytest.skip(reason)
    path = Path(raw)
    if path.exists():
        pytest.fail(f"{_RECEIPT_ENV} must name a new file: {path} exists")
    return path


def _environ() -> dict[str, str]:
    environ = dict(os.environ)
    environ.setdefault("TYPEVET_LLAMA__MULTIMODAL_MODEL", _LLAMA_MODEL)
    environ.setdefault("TYPEVET_LLAMA__TIMEOUT", _LLAMA_TIMEOUT)
    return environ


def _slice(per_kind: int) -> tuple[CedarPair, ...]:
    pairs = select_balanced_slice(per_kind=per_kind, seed=DEFAULT_SEED)
    if per_kind == DEFAULT_PER_KIND:
        expected = _SLICE_IDS.read_text(encoding="utf-8").split()
        assert [p.pair_id for p in pairs] == expected, "slice ids drifted"
    return pairs


def _requests(pairs: Sequence[CedarPair]) -> list[SignatureMatchRequest]:
    archive = fetch_cedar_archive()
    images = read_members(archive, [p for pair in pairs for p in pair.member_paths])
    return [
        build_signature_match_request(
            pair,
            reference_image=images[pair.reference.member_path],
            questioned_image=images[pair.questioned.member_path],
        )
        for pair in pairs
    ]


def _prompt_specs() -> tuple[PromptSpec, ...]:
    specs = []
    for name, question in signature_match_questions().items():
        raw = question.criteria
        if isinstance(question, Score):
            assert isinstance(raw, Sequence)
            criteria = {str(i): str(text) for i, text in enumerate(raw)}
        else:
            assert isinstance(question, Choice | Noul)
            assert isinstance(raw, Mapping)
            criteria = {str(k): str(v) for k, v in raw.items()}
        specs.append(
            PromptSpec(name, tuple(criteria), str(question.instructions), criteria)
        )
    return tuple(specs)


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
def _open_session(
    environ: Mapping[str, str], backend: str
) -> Iterator[GemmaNativeVisionSession | VllmJudgmentSession]:
    """Open the judgment session for ``backend``.

    The llama.cpp branch opens the Gemma native vision session on the
    factory's pooled client. The scoring calls send a request once more when
    the router closes a reused connection before a response (#305).

    Yields:
        The judgment session.
    """
    if backend == "vllm":
        with open_judgment(environ) as vllm_session:
            yield vllm_session
        return
    settings = load_llama_settings(environ)
    with open_gemma_native_vision_judgment(settings=settings) as llama_session:
        yield llama_session


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
        props = client.get("/props", params={"model": model}).raise_for_status()
        body = props.json()
        facts["build_info"] = body.get("build_info", "unknown")
        facts["model_alias"] = body.get("model_alias")
        facts["n_ctx"] = body.get("default_generation_settings", {}).get("n_ctx")
        facts["http_keepalive"] = True
    else:
        version = client.get("/version")
        facts["build_info"] = (
            version.json().get("version", "unknown")
            if version.status_code == httpx.codes.OK
            else "unknown"
        )
    return facts


def _summary(receipt: Mapping[str, object], digest: str, path: Path) -> str:
    metrics = receipt["metrics"]
    assert isinstance(metrics, Mapping)
    keys = (
        "pairs",
        "accuracy",
        "kind_accuracy",
        "roc_auc",
        "ece",
        "cannot_tell_rate",
        "skilled_false_accept",
        "noul_choice_agreement",
        "mean_latency_seconds",
    )
    shown = {k: metrics[k] for k in keys}
    by_kind = metrics["by_kind"]
    assert isinstance(by_kind, Mapping)
    brief = {
        kind: {k: row[k] for k in ("pairs", "verdict_counts", "noul_accept_rate")}
        for kind, row in by_kind.items()
    }
    return (
        f"receipt {path} sha256 {digest}\n"
        f"metrics {json.dumps(shown)}\n"
        f"by_kind {json.dumps(brief)}\n"
        f"wall_seconds {receipt['wall_seconds']} stopped {receipt['stopped']}\n"
        f"throughput {json.dumps(receipt['throughput'])}\n"
        "same_writer values are model confidence, not a match percentage."
    )


@pytest.mark.live
def test_signature_match_live_receipt() -> None:
    """Judge the CEDAR slice once and write the key-free receipt."""
    path = _receipt_path()
    environ = _environ()
    backend = load_backend(environ)
    secret = load_vllm_settings(environ).api_key if backend == "vllm" else None
    weights = served_weights_pins(backend, environ)
    per_kind = int(environ.get(_PER_KIND_ENV, str(DEFAULT_PER_KIND)))
    concurrency = image_concurrency(environ)
    pairs = _slice(per_kind)
    requests = _requests(pairs)
    tree = _working_tree()
    evaluated = snapshot_evaluated_inputs(
        prompts=_prompt_specs(),
        code_paths={
            **{
                name: _SIGNATURE_SRC / f"{name}.py"
                for name in ("request", "metrics", "runner")
            },
            "pool": _POOL_SRC,
            "serving_metrics": _SERVING_METRICS_SRC,
        },
        fixture_paths={"default_slice_ids": _SLICE_IDS},
    )

    with _open_session(environ, backend) as session:
        facts = _server_facts(backend, session.client, session.model)
        served_template = getattr(session, "served", None)
        start = begin_run_identity(
            repo_root=_REPO_ROOT,
            runtime=RuntimeBuild(
                session.model,
                served_template.value if served_template else "vllm_chat",
                str(facts["build_info"]),
            ),
            working_tree=tree,
        )
        client = session.client
        run = run_signature_match(
            session.port,
            requests,
            session.model,
            concurrency=concurrency,
            server_metrics=(
                (lambda: read_metrics(client)) if backend == "vllm" else None
            ),
        )

    identity = finalize_experiment_identity(
        run_start=start,
        evaluated=evaluated,
        arm_call_counts={"judgments": len(run.outcomes)},
    )
    slice_text = "\n".join(p.pair_id for p in pairs) + "\n"
    pins = {
        "finished_utc": datetime.now(UTC).isoformat(),
        "dataset": "CEDAR offline signatures",
        "archive_sha256": ARCHIVE_SHA256,
        "slice_seed": DEFAULT_SEED,
        "slice_per_kind": per_kind,
        "concurrency": concurrency,
        "slice_ids_sha256": hashlib.sha256(slice_text.encode()).hexdigest(),
        "server": facts,
        **weights,
    }
    receipt = build_signature_match_receipt(
        run,
        backend=backend,
        model=session.model,
        pins=pins,
        identity=identity.to_receipt_mapping(),
    )
    text = json.dumps(receipt, indent=2)
    ensure_key_free(text, secrets=(secret,))
    write_receipt_exclusive(path, receipt)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    print(_summary(receipt, digest, path))

    assert run.stopped is None, run.stopped
    assert len(run.outcomes) == len(requests) == 3 * per_kind
