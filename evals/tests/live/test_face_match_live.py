r"""Opt-in live LFW face-match run with a key-free receipt (#301).

Judges the balanced LFW View 2 slice (100 same-person and 100
different-person pairs, seed 0) once on one backend. Each pair is one
two-image judgment with three typed questions. The run stops at the first
backend failure and records it.

The test skips unless ``TYPEVET_FACE_MATCH_RECEIPT`` names the receipt file.
It fails when ``TYPEVET_REQUIRE_LIVE`` is truthy and that variable is
missing. The receipt path must not exist; the test fails before any network
call when it does. ``TYPEVET_BACKEND`` selects the backend, as in
``open_judgment``:

- ``llama_cpp`` (default) reads ``TYPEVET_LLAMA__*``. The multimodal model
  defaults to ``gemma-4-31b-kv9-q4km-mm`` and the timeout to 900 seconds.
  The HTTP client keeps no idle connection between calls.
- ``vllm`` reads ``TYPEVET_VLLM__BASE_URL``, ``TYPEVET_VLLM__MODEL``,
  ``TYPEVET_VLLM__API_KEY`` and ``TYPEVET_VLLM__USER_AGENT``.

``TYPEVET_FACE_MATCH_PER_CLASS`` sets a smaller slice for a smoke run.
``TYPEVET_GIT_STATUS_PORCELAIN`` carries the porcelain status text for the
working-tree fingerprint, as in the CORD smoke. The
receipt holds pair ids, typed answers, metrics and pins. It holds no image
bytes, no key and no auth header. The ``same_person`` probability is model
confidence, not a match percentage.

Examples:
    ```bash
    TYPEVET_FACE_MATCH_RECEIPT=evals/fixtures/lfw/receipts/face_match_llama_cpp_receipt.json \
      uv run pytest evals/tests/live/test_face_match_live.py -m live -q -s

    TYPEVET_BACKEND=vllm \
      TYPEVET_VLLM__BASE_URL=https://<pod>-8000.proxy.runpod.net \
      TYPEVET_VLLM__MODEL=google/gemma-4-31B-it \
      TYPEVET_FACE_MATCH_RECEIPT=evals/fixtures/lfw/receipts/face_match_vllm_receipt.json \
      uv run pytest evals/tests/live/test_face_match_live.py -m live -q -s
    ```

See Also:
    - [typevet_evals.face_match.runner][]: run, metrics and receipt
    - [typevet_evals.datasets.lfw][]: pinned LFW files and the slice
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
from typevet_evals.datasets.lfw import (
    ARCHIVE_SHA256,
    DEFAULT_PER_CLASS,
    DEFAULT_SEED,
    PAIRS_SHA256,
    fetch_lfw_files,
    load_pairs,
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
    FaceMatchRequest,
    build_face_match_receipt,
    build_face_match_request,
    ensure_key_free,
    face_match_questions,
    run_face_match,
)
from typevet_evals.runner.live_gate import require_live_enabled

_REPO_ROOT = Path(__file__).resolve().parents[3]
_FACE_MATCH_SRC = _REPO_ROOT / "evals" / "src" / "typevet_evals" / "face_match"
_RECEIPT_ENV = "TYPEVET_FACE_MATCH_RECEIPT"
_PER_CLASS_ENV = "TYPEVET_FACE_MATCH_PER_CLASS"
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


def _requests(per_class: int) -> list[FaceMatchRequest]:
    files = fetch_lfw_files()
    pairs = select_balanced_slice(load_pairs(files.pairs_path), per_class=per_class)
    images = read_members(
        files.archive_path, [p for pair in pairs for p in pair.member_paths]
    )
    return [
        build_face_match_request(
            pair,
            left_image=images[pair.left.member_path],
            right_image=images[pair.right.member_path],
        )
        for pair in pairs
    ]


def _prompt_specs() -> tuple[PromptSpec, ...]:
    specs = []
    for name, question in face_match_questions().items():
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

    The llama.cpp branch opens the Gemma native vision session on a client
    without keep-alive. Two earlier runs on a pooled client stopped with
    "Server disconnected without sending a response" while the model process
    stayed up, which points at a closed keep-alive connection.

    Yields:
        The judgment session.
    """
    if backend == "vllm":
        with open_judgment(environ) as vllm_session:
            yield vllm_session
        return
    settings = load_llama_settings(environ)
    with (
        httpx.Client(
            base_url=settings.base_url.rstrip("/"),
            timeout=settings.timeout,
            limits=httpx.Limits(max_keepalive_connections=0),
        ) as client,
        open_gemma_native_vision_judgment(
            settings=settings, http_client=client
        ) as llama_session,
    ):
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
        facts["http_keepalive"] = False
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
    keys = ("pairs", "accuracy", "roc_auc", "ece", "cannot_tell_rate")
    shown = {k: metrics[k] for k in keys}
    shown["mean_latency_seconds"] = metrics["mean_latency_seconds"]
    return (
        f"receipt {path} sha256 {digest}\n"
        f"metrics {json.dumps(shown)}\n"
        f"score_distribution {json.dumps(metrics['score_distribution'])}\n"
        f"wall_seconds {receipt['wall_seconds']} stopped {receipt['stopped']}\n"
        "same_person values are model confidence, not a match percentage."
    )


@pytest.mark.live
def test_face_match_live_receipt() -> None:
    """Judge the LFW slice once and write the key-free receipt."""
    path = _receipt_path()
    environ = _environ()
    backend = load_backend(environ)
    secret = load_vllm_settings(environ).api_key if backend == "vllm" else None
    per_class = int(environ.get(_PER_CLASS_ENV, str(DEFAULT_PER_CLASS)))
    requests = _requests(per_class)
    tree = _working_tree()
    evaluated = snapshot_evaluated_inputs(
        prompts=_prompt_specs(),
        code_paths={
            name: _FACE_MATCH_SRC / f"{name}.py"
            for name in ("request", "metrics", "runner")
        },
        fixture_paths={},
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
        run = run_face_match(session.port, requests, session.model)

    identity = finalize_experiment_identity(
        run_start=start,
        evaluated=evaluated,
        arm_call_counts={"judgments": len(run.outcomes)},
    )
    pins = {
        "finished_utc": datetime.now(UTC).isoformat(),
        "dataset": "LFW View 2",
        "pairs_sha256": PAIRS_SHA256,
        "archive_sha256": ARCHIVE_SHA256,
        "slice_seed": DEFAULT_SEED,
        "slice_per_class": per_class,
        "server": facts,
    }
    receipt = build_face_match_receipt(
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
    assert len(run.outcomes) == len(requests) == 2 * per_class
