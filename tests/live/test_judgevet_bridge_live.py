"""Live judgevet bridge receipts, one per typevet backend (#289, #445).

judgevet's ``SystemOnePort`` drives a real typevet session through
``provider_factory`` / public ``open_judgment`` and asks one ``Noul``, one
``Choice`` and one ``Score`` in a single ``system_one`` call. Each backend is
one parametrized case:

- ``llama_cpp`` runs when ``TYPEVET_LLAMA__BASE_URL`` is set. Model selection
  uses ``TYPEVET_LLAMA__DEFAULT_MODEL`` (or legacy ``TYPEVET_GEMMA_MODEL``),
  with the historical Gemma pin only when unframed and unset. Explicit
  ``TYPEVET_LIVE_LLAMA_FRAMING=chatml`` selects ``ChatMLThinkingFraming`` and
  the empty-think control replay (#445).
- ``vllm`` runs when ``TYPEVET_VLLM__BASE_URL`` is set. It opens
  ``open_judgment`` with ``TYPEVET_BACKEND=vllm`` and the
  ``TYPEVET_VLLM__*`` settings.

A case skips when its variable is absent, or fails when
``TYPEVET_REQUIRE_LIVE`` is truthy. The test asserts structure only: typed
judgevet answers, the model id equal to a served id, and finite
probabilities. One run is not a calibration claim. Set
``TYPEVET_LIVE_RECEIPT_DIR`` to write each receipt JSON. Receipts carry no
key.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import pytest
from judgevet import (
    ChoiceAnswer,
    NoulAnswer,
    Score,
    ScoreAnswer,
    SystemOneResponse,
)

from tests.live.gate import gate_live
from tests.live.judgevet_llama_receipt import (
    HISTORICAL_GEMMA_FALLBACK,
    QUESTIONS,
    STATE,
    RecordingTransport,
    run_llama_bridge_receipt,
    scrub_secrets,
)
from typevet.adapters.inbound.backend_settings import load_vllm_settings, open_judgment
from typevet.adapters.inbound.fake_backend import require_live_session
from typevet.adapters.inbound.judgevet import JudgmentSession, provider_factory
from typevet.adapters.inbound.settings import load_llama_settings
from typevet_evals.runner.live_gate import require_live_enabled

_HISTORICAL_GEMMA = HISTORICAL_GEMMA_FALLBACK
_STATE = STATE
_QUESTIONS = QUESTIONS
_TOLERANCE = 1e-6
_BACKEND_NAMES = ("llama_cpp", "vllm")


@dataclass
class _Backend:
    """One backend case: how to open its session and what it serves.

    Attributes:
        name (str): Backend name in the receipt.
        model (str): Model id sent through judgevet.
        open_session (Callable): Opens a typevet ``JudgmentSession``.
        served (list[dict[str, object]]): ``/v1/models`` entries the server lists.
        facts (dict[str, object]): Server facts for the receipt.
        secret (str | None): Key that must not appear in the receipt.
    """

    name: str
    model: str
    open_session: Any
    served: list[dict[str, object]]
    facts: dict[str, object] = field(default_factory=dict)
    secret: str | None = None


def _served_ids(client: httpx.Client) -> list[dict[str, object]]:
    body = client.get("/v1/models").raise_for_status().json()
    return [
        {k: item.get(k) for k in ("id", "root", "max_model_len") if k in item}
        for item in body["data"]
    ]


def _require_env(name: str) -> None:
    if os.environ.get(name):
        return
    reason = f"{name} not set"
    if require_live_enabled():
        pytest.fail(reason)
    pytest.skip(reason)


def _vllm_backend() -> _Backend:
    _require_env("TYPEVET_VLLM__BASE_URL")
    environ = {**os.environ, "TYPEVET_BACKEND": "vllm"}
    settings = load_vllm_settings(environ)
    served_ids: list[dict[str, object]] = []

    @contextmanager
    def _open_vllm() -> Iterator[JudgmentSession]:
        with open_judgment(environ) as opened:
            session = require_live_session(opened)
            assert session.model == settings.model
            served_ids.extend(_served_ids(session.client))
            yield session

    return _Backend(
        "vllm",
        settings.model,
        _open_vllm,
        served_ids,
        {"base_url": settings.base_url},
        settings.api_key,
    )


def _answers_summary(response: SystemOneResponse) -> dict[str, object]:
    noul = response.answers["duplicate_charge"]
    choice = response.answers["route"]
    score = response.answers["urgency"]
    assert isinstance(noul, NoulAnswer)
    assert isinstance(choice, ChoiceAnswer)
    assert isinstance(score, ScoreAnswer)
    true_p = float(noul.noul)
    return {
        "duplicate_charge": {
            "type": "noul",
            "noul": noul.noul,
            "probabilities": {"0": 1.0 - true_p, "1": true_p},
        },
        "route": {
            "type": "choice",
            "choice": choice.choice,
            "confidence": choice.confidence,
            "probabilities": choice.probabilities,
        },
        "urgency": {
            "type": "score",
            "score": score.score,
            "confidence": score.confidence,
            "legend": {str(k): v for k, v in score.legend.items()},
            "probabilities": {str(k): v for k, v in score.probabilities.items()},
        },
    }


def _assert_distribution(probabilities: Mapping[Any, float], keys: set[Any]) -> None:
    assert set(probabilities) == keys
    values = list(probabilities.values())
    assert all(math.isfinite(p) and 0.0 <= p <= 1.0 for p in values)
    assert abs(sum(values) - 1.0) < _TOLERANCE


def _write_receipt(name: str, text: str) -> None:
    out_dir = os.environ.get("TYPEVET_LIVE_RECEIPT_DIR")
    if not out_dir:
        return
    path = Path(out_dir) / f"judgevet_bridge_{name}_receipt.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")


def _stdout_receipt(receipt: dict[str, object]) -> dict[str, object]:
    """Return a receipt copy safe for stdout (no full top-N dump)."""
    summary = dict(receipt)
    first_top = summary.pop("first_top_n", None)
    if isinstance(first_top, list):
        summary["first_top_n_count"] = len(first_top)
    control = summary.get("control")
    if isinstance(control, dict):
        control_summary = dict(control)
        control_summary.pop("response", None)
        control_summary.pop("request", None)
        summary["control"] = control_summary
    return summary


def _receipt_text(receipt: Mapping[str, object], *, secret: str | None) -> str:
    """Serialize a receipt after scrubbing configured secrets."""
    secrets = (secret,) if secret else ()
    cleaned = scrub_secrets(dict(receipt), secrets)
    text = json.dumps(cleaned, indent=2, sort_keys=True)
    if secret:
        assert secret not in text
    return text


def _assert_llama_receipt(receipt: dict[str, object], *, secret: str | None) -> None:
    """Assert structure on a llama.cpp harness receipt."""
    view = scrub_secrets(receipt, (secret,) if secret else ())
    assert isinstance(view, dict)
    if secret:
        assert secret not in json.dumps(view)
    assert view["model"] == view["requested_model"]
    served = view["served"]
    assert isinstance(served, list)
    served_ids = {entry["id"] for entry in served}
    assert view["model"] in served_ids
    answers = view["answers"]
    assert isinstance(answers, dict)
    assert set(answers) == set(_QUESTIONS)
    noul = answers["duplicate_charge"]
    choice = answers["route"]
    score = answers["urgency"]
    assert isinstance(noul, dict)
    assert isinstance(choice, dict)
    assert isinstance(score, dict)
    assert math.isfinite(float(noul["noul"])) and 0.0 <= float(noul["noul"]) <= 1.0
    _assert_distribution(noul["probabilities"], {"0", "1"})
    assert noul["probabilities"]["1"] == pytest.approx(
        float(noul["noul"]), abs=_TOLERANCE
    )
    _assert_distribution(choice["probabilities"], {"billing", "technical", "sales"})
    assert choice["choice"] in choice["probabilities"]
    urgency = _QUESTIONS["urgency"]
    assert isinstance(urgency, Score)
    assert len(score["legend"]) == len(urgency.criteria)
    _assert_distribution(score["probabilities"], set(score["legend"]))
    levels = sorted(int(level) for level in score["probabilities"])
    assert (
        math.isfinite(float(score["score"]))
        and levels[0] <= float(score["score"]) <= levels[-1]
    )


def _run_llama_live_case(
    *,
    transport: httpx.BaseTransport | None = None,
) -> dict[str, object]:
    """Run the llama.cpp live harness and assert the receipt."""
    _require_env("TYPEVET_LLAMA__BASE_URL")
    settings = load_llama_settings()
    gate_live(settings)
    recorder = (
        transport
        if isinstance(transport, RecordingTransport)
        else RecordingTransport(
            transport or httpx.HTTPTransport(retries=0),
            max_completions=4,
        )
    )
    receipt: dict[str, object] = {}
    try:
        run_llama_bridge_receipt(os.environ, transport=recorder, receipt=receipt)
        _assert_llama_receipt(receipt, secret=settings.api_key)
    except BaseException as exc:
        if receipt and "failure" not in receipt:
            receipt["failure"] = {
                "type": type(exc).__name__,
                "reason": str(exc),
                "completion_attempts": recorder.completion_attempts,
            }
        raise
    finally:
        if receipt:
            text = _receipt_text(receipt, secret=settings.api_key)
            _write_receipt("llama_cpp", text)
            stdout = scrub_secrets(
                _stdout_receipt(receipt),
                (settings.api_key,) if settings.api_key else (),
            )
            print(json.dumps(stdout, indent=2, sort_keys=True))
    return receipt


def _run_vllm_live_case() -> None:
    """Run the vLLM live path and assert the receipt."""
    backend = _vllm_backend()
    factory = provider_factory(backend.open_session)
    with factory() as port:
        started = time.perf_counter()
        response = port.system_one(_STATE, _QUESTIONS, backend.model)
        wall_seconds = time.perf_counter() - started

    answers = _answers_summary(response)
    receipt = {
        "issue": 289,
        "backend": backend.name,
        "requested_model": backend.model,
        "model": response.model,
        "served": backend.served,
        "server": backend.facts,
        "usage": {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        },
        "answers": answers,
        "receipts": dict(response.receipts),
        "wall_seconds": round(wall_seconds, 3),
    }
    text = _receipt_text(receipt, secret=backend.secret)
    _write_receipt(backend.name, text)
    print(text)

    served_ids = {entry["id"] for entry in backend.served}
    assert response.model == backend.model
    assert response.model in served_ids
    assert set(response.answers) == set(_QUESTIONS)
    noul_answer = response.answers["duplicate_charge"]
    choice_answer = response.answers["route"]
    score_answer = response.answers["urgency"]
    assert isinstance(noul_answer, NoulAnswer)
    assert isinstance(choice_answer, ChoiceAnswer)
    assert isinstance(score_answer, ScoreAnswer)
    assert math.isfinite(noul_answer.noul) and 0.0 <= noul_answer.noul <= 1.0
    _assert_distribution(choice_answer.probabilities, {"billing", "technical", "sales"})
    assert choice_answer.choice in choice_answer.probabilities
    urgency = _QUESTIONS["urgency"]
    assert isinstance(urgency, Score)
    assert len(score_answer.legend) == len(urgency.criteria)
    _assert_distribution(score_answer.probabilities, set(score_answer.legend))
    levels = sorted(score_answer.legend)
    assert (
        math.isfinite(score_answer.score)
        and levels[0] <= score_answer.score <= levels[-1]
    )


@pytest.mark.live
@pytest.mark.parametrize("backend_name", _BACKEND_NAMES)
def test_judgevet_bridge_live_receipt(backend_name: str) -> None:
    """One judgevet ``system_one`` call returns typed answers on a real backend."""
    if backend_name == "llama_cpp":
        _run_llama_live_case()
        return
    _run_vllm_live_case()
