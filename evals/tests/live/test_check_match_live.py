r"""Opt-in live check-versus-register run with a key-free receipt (#316).

Judges the synthetic check slice (20 register rows times 7 variants, seed 0,
140 cases) once on one backend. Each case is one one-image judgment with the
register row as text and four typed questions. The run stops at the first
backend failure and records it.

The test skips unless ``TYPEVET_CHECK_MATCH_RECEIPT`` names the receipt file.
It fails when ``TYPEVET_REQUIRE_LIVE`` is truthy and that variable is
missing. The receipt path must not exist; the test fails before any network
call when it does. ``TYPEVET_BACKEND`` selects the backend, as in
``open_judgment``:

- ``llama_cpp`` (default) reads ``TYPEVET_LLAMA__*``. The multimodal model
  defaults to ``gemma-4-31b-kv9-q4km-mm`` and the timeout to 900 seconds.
  The scoring calls send a request once more when the router closes a
  reused connection before a response (#305).
- ``vllm`` reads ``TYPEVET_VLLM__BASE_URL``, ``TYPEVET_VLLM__MODEL``,
  ``TYPEVET_VLLM__API_KEY`` and ``TYPEVET_VLLM__USER_AGENT``. It also
  requires ``TYPEVET_VLLM_MODEL_REVISION``, the served weights revision. The
  test fails before any network call when it is missing.

``TYPEVET_CHECK_MATCH_ROWS`` sets fewer register rows for a smoke run; each
row still gives all 7 variants. ``TYPEVET_GIT_STATUS_PORCELAIN`` carries the
porcelain status text for the working-tree fingerprint. The receipt holds
case ids, typed answers, render SHA-256 digests, metrics and pins (generator
seed, Pillow version, model, revision, git fingerprint). It holds no image
bytes, no key and no auth header. Generated checks are not evidence about
real checks.

Examples:
    ```bash
    TYPEVET_CHECK_MATCH_RECEIPT=evals/fixtures/checks/receipts/check_match_llama_cpp_receipt.json \
      uv run pytest evals/tests/live/test_check_match_live.py -m live -q -s

    TYPEVET_BACKEND=vllm \
      TYPEVET_VLLM_MODEL_REVISION=<hf-commit-sha> \
      TYPEVET_VLLM__BASE_URL=https://<pod>-8000.proxy.runpod.net \
      TYPEVET_VLLM__MODEL=google/gemma-4-31B-it \
      TYPEVET_CHECK_MATCH_RECEIPT=evals/fixtures/checks/receipts/check_match_vllm_receipt.json \
      uv run pytest evals/tests/live/test_check_match_live.py -m live -q -s
    ```

See Also:
    - [typevet_evals.check_match.runner][]: run, metrics and receipt
    - [typevet_evals.check_match.cases][]: register rows and variants
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
import PIL
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
from typevet_evals.check_match import (
    DEFAULT_SEED,
    ROW_COUNT,
    CheckMatchRequest,
    build_check_match_receipt,
    build_check_match_request,
    check_cases,
    check_match_questions,
    render_check,
    run_check_match,
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

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CHECK_MATCH_SRC = _REPO_ROOT / "evals" / "src" / "typevet_evals" / "check_match"
_RECEIPT_ENV = "TYPEVET_CHECK_MATCH_RECEIPT"
_ROWS_ENV = "TYPEVET_CHECK_MATCH_ROWS"
_LLAMA_MODEL = "gemma-4-31b-kv9-q4km-mm"
_LLAMA_TIMEOUT = "900"
_CODE_MODULES = ("cases", "render", "words", "request", "metrics", "runner")


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


def _requests(rows: int) -> list[CheckMatchRequest]:
    return [
        build_check_match_request(case, image=render_check(case))
        for case in check_cases(DEFAULT_SEED, rows)
    ]


def _prompt_specs() -> tuple[PromptSpec, ...]:
    specs = []
    for name, question in check_match_questions().items():
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
    else:
        version = client.get("/version")
        facts["build_info"] = (
            version.json().get("version", "unknown")
            if version.status_code == httpx.codes.OK
            else "unknown"
        )
    return facts


def _slice_sha256(requests: Sequence[CheckMatchRequest]) -> str:
    digest = hashlib.sha256()
    for request in requests:
        digest.update(request.case_id.encode())
        digest.update(hashlib.sha256(request.media[0].data).digest())
    return digest.hexdigest()


def _summary(receipt: Mapping[str, object], digest: str, path: Path) -> str:
    metrics = receipt["metrics"]
    assert isinstance(metrics, Mapping)
    keys = ("cases", "accuracy", "false_clear_rate", "cannot_tell_rate")
    shown = {k: metrics[k] for k in keys}
    shown["legibility_gap"] = metrics["legibility_gap_clean_minus_low"]
    shown["agreement"] = metrics["noul_choice_agreement"]
    shown["mean_latency_seconds"] = metrics["mean_latency_seconds"]
    return (
        f"receipt {path} sha256 {digest}\n"
        f"metrics {json.dumps(shown)}\n"
        f"accuracy_by_variant {json.dumps(metrics['accuracy_by_variant'])}\n"
        f"wall_seconds {receipt['wall_seconds']} stopped {receipt['stopped']}\n"
        "Noul values are model confidence; generated checks are not real checks."
    )


@pytest.mark.live
def test_check_match_live_receipt() -> None:
    """Judge the synthetic check slice once and write the key-free receipt."""
    path = _receipt_path()
    environ = _environ()
    backend = load_backend(environ)
    secret = load_vllm_settings(environ).api_key if backend == "vllm" else None
    weights = served_weights_pins(backend, environ)
    rows = int(environ.get(_ROWS_ENV, str(ROW_COUNT)))
    if not 1 <= rows <= ROW_COUNT:
        pytest.fail(f"{_ROWS_ENV} must be 1 to {ROW_COUNT}: {rows}")
    requests = _requests(rows)
    tree = _working_tree()
    evaluated = snapshot_evaluated_inputs(
        prompts=_prompt_specs(),
        code_paths={name: _CHECK_MATCH_SRC / f"{name}.py" for name in _CODE_MODULES},
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
        run = run_check_match(session.port, requests, session.model)

    identity = finalize_experiment_identity(
        run_start=start,
        evaluated=evaluated,
        arm_call_counts={"judgments": len(run.outcomes)},
    )
    pins = {
        "finished_utc": datetime.now(UTC).isoformat(),
        "dataset": "synthetic checks (#315 generator)",
        "generator_seed": DEFAULT_SEED,
        "register_rows": rows,
        "variants_per_row": len(requests) // rows,
        "pillow_version": PIL.__version__,
        "slice_sha256": _slice_sha256(requests),
        "server": facts,
        **weights,
    }
    receipt = build_check_match_receipt(
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
    assert len(run.outcomes) == len(requests) == 7 * rows
