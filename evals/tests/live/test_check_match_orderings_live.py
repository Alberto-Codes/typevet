r"""Opt-in live option-order study of the check verdict (#105).

Scores only the six-label verdict ``Choice`` of the seed-1 synthetic checks,
register rows r00 to r02 and all 7 variants: 21 cases. Each case is scored
once per ordering of the balanced K=6 design, 126 requests in total, one at
a time. Ordering 0 is today's option order. The study follows TypeLLM's rule:
the arithmetic mean of post-softmax probabilities over the orderings.

The test skips unless ``TYPEVET_CHECK_ORDERINGS_RECEIPT`` names the receipt
file. It fails when ``TYPEVET_REQUIRE_LIVE`` is truthy and that variable is
missing. The receipt path must not exist. Local llama.cpp only: the test
fails before any network call for another ``TYPEVET_BACKEND``. The seed is 1
(``TYPEVET_CHECK_MATCH_SEED`` defaults to 1 and another value fails), because
the comparison receipt
``evals/fixtures/checks/receipts/check_match_llama_cpp_seed1_receipt.json``
is seed 1. Each render must equal the digest in that receipt.

The llama.cpp settings are those of the main check-match live test
(``TYPEVET_LLAMA__*``; model ``gemma-4-31b-kv9-q4km-mm``, timeout 900
seconds). ``TYPEVET_GIT_STATUS_PORCELAIN`` carries the porcelain status for
the working-tree fingerprint. The receipt holds the per-case numbers, the
four statistics, the decision label, pins and identity. It holds no image
bytes and no key. Generated checks are not evidence about real checks.

Examples:
    ```bash
    RECEIPTS=evals/fixtures/checks/receipts
    TYPEVET_CHECK_ORDERINGS_RECEIPT=$RECEIPTS/check_match_orderings_llama_cpp_seed1.json \
      TYPEVET_GIT_STATUS_PORCELAIN="$(git status --porcelain)" \
      uv run pytest evals/tests/live/test_check_match_orderings_live.py -m live -q -s
    ```

See Also:
    - [typevet_evals.check_match.orderings][]: design, mean and statistics
    - [typevet_evals.check_match.request][]: the verdict ``Choice``
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

import httpx
import PIL
import pytest

from typevet.adapters.inbound.backend_settings import load_backend
from typevet.adapters.inbound.settings import load_llama_settings
from typevet.adapters.outbound.llama_cpp.gemma_native_vision_factory import (
    open_gemma_native_vision_judgment,
)
from typevet.domain import Choice
from typevet_evals.check_match import (
    SEED_ENV,
    VERDICT,
    VERDICT_LABELS,
    check_match_questions,
    check_match_seed,
    check_match_slice,
)
from typevet_evals.check_match.orderings import (
    balanced_orders,
    build_orderings_receipt,
    run_orderings,
    single_order_cases,
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
from typevet_evals.face_match import ensure_key_free
from typevet_evals.runner.live_gate import require_live_enabled
from typevet_evals.throughput.server_args import server_args_block, stated_server_args

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CHECK_MATCH_SRC = _REPO_ROOT / "evals" / "src" / "typevet_evals" / "check_match"
_RECEIPTS = _REPO_ROOT / "evals" / "fixtures" / "checks" / "receipts"
_SINGLE_ORDER_RECEIPT = _RECEIPTS / "check_match_llama_cpp_seed1_receipt.json"
_RECEIPT_ENV = "TYPEVET_CHECK_ORDERINGS_RECEIPT"
_LLAMA_MODEL = "gemma-4-31b-kv9-q4km-mm"
_LLAMA_TIMEOUT = "900"
_SEED = 1
_ROWS = 3
_CODE_MODULES = ("cases", "render", "words", "request", "orderings")


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
    environ.setdefault(SEED_ENV, str(_SEED))
    return environ


def _prompt_specs() -> tuple[PromptSpec, ...]:
    question = check_match_questions()[VERDICT]
    assert isinstance(question, Choice)
    criteria = {str(k): str(v) for k, v in question.criteria.items()}
    return (PromptSpec(VERDICT, tuple(criteria), str(question.instructions), criteria),)


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


def _server_facts(client: httpx.Client, model: str) -> dict[str, object]:
    served = client.get("/v1/models").raise_for_status().json()["data"]
    props = client.get("/props", params={"model": model}).raise_for_status().json()
    return {
        "served": [
            {k: item.get(k) for k in ("id", "root", "max_model_len") if k in item}
            for item in served
            if item.get("id") == model
        ],
        "build_info": props.get("build_info", "unknown"),
        "model_alias": props.get("model_alias"),
        "n_ctx": props.get("default_generation_settings", {}).get("n_ctx"),
    }


def _summary(receipt: Mapping[str, object], digest: str, path: Path) -> str:
    return (
        f"receipt {path} sha256 {digest}\n"
        f"statistics {json.dumps(receipt['statistics'])}\n"
        f"requests {receipt['requests']} wall_seconds {receipt['wall_seconds']} "
        f"stopped {receipt['stopped']}\n"
        "Probabilities are model confidence; generated checks are not real checks."
    )


@pytest.mark.live
def test_check_match_orderings_live_receipt() -> None:
    """Score the verdict under each balanced ordering and write the receipt."""
    path = _receipt_path()
    environ = _environ()
    backend = load_backend(environ)
    if backend != "llama_cpp":
        pytest.fail(f"the orderings study runs on local llama_cpp only: {backend}")
    if check_match_seed(environ) != _SEED:
        pytest.fail(f"{SEED_ENV} must be {_SEED}: the comparison receipt is seed 1")
    made = check_match_slice(environ, _ROWS)
    requests = made.requests
    comparison = json.loads(_SINGLE_ORDER_RECEIPT.read_text(encoding="utf-8"))
    single = single_order_cases(comparison, requests, seed=_SEED)
    orders = balanced_orders(len(VERDICT_LABELS))
    tree = _working_tree()
    evaluated = snapshot_evaluated_inputs(
        prompts=_prompt_specs(),
        code_paths={name: _CHECK_MATCH_SRC / f"{name}.py" for name in _CODE_MODULES},
        fixture_paths={"single_order_receipt": _SINGLE_ORDER_RECEIPT},
    )

    settings = load_llama_settings(environ)
    with open_gemma_native_vision_judgment(settings=settings) as session:
        facts = _server_facts(session.client, session.model)
        start = begin_run_identity(
            repo_root=_REPO_ROOT,
            runtime=RuntimeBuild(
                session.model, session.served.value, str(facts["build_info"])
            ),
            working_tree=tree,
        )
        run = run_orderings(session.port, requests, session.model, orders, single)

    identity = finalize_experiment_identity(
        run_start=start,
        evaluated=evaluated,
        arm_call_counts={"verdict_requests": run.requests},
    )
    pins = {
        "finished_utc": datetime.now(UTC).isoformat(),
        **made.pins,
        "concurrency": 1,
        "orderings": [[VERDICT_LABELS[i] for i in order] for order in orders],
        "single_order_receipt": _SINGLE_ORDER_RECEIPT.name,
        "single_order_receipt_sha256": hashlib.sha256(
            _SINGLE_ORDER_RECEIPT.read_bytes()
        ).hexdigest(),
        "pillow_version": PIL.__version__,
        "server": facts,
        "server_args": server_args_block(None, stated_server_args(environ)),
    }
    receipt = build_orderings_receipt(
        run,
        backend=backend,
        model=session.model,
        pins=pins,
        identity=identity.to_receipt_mapping(),
    )
    ensure_key_free(json.dumps(receipt, indent=2), secrets=())
    write_receipt_exclusive(path, receipt)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    print(_summary(receipt, digest, path))

    assert run.stopped is None, run.stopped
    assert len(run.records) == len(requests) == 7 * _ROWS
    assert run.requests == len(orders) * len(requests) == 126
