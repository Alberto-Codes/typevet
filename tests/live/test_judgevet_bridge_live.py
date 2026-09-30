"""Live judgevet bridge receipts, one per typevet backend (#289).

judgevet's ``SystemOnePort`` drives a real typevet session through
``provider_factory`` and asks one ``Noul``, one ``Choice`` and one ``Score``
in a single ``system_one`` call. Each backend is one parametrized case:

- ``llama_cpp`` runs when ``TYPEVET_LLAMA__BASE_URL`` is set. It builds a
  text-only session (``ScoringJudgmentAdapter`` over
  ``LlamaCppCandidateScoringAdapter``) on the pinned Gemma 4 router model,
  because ``open_judgment`` on llama.cpp opens the vision session.
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
from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import httpx
import pytest
from judgevet import (
    Choice,
    ChoiceAnswer,
    Noul,
    NoulAnswer,
    Score,
    ScoreAnswer,
    SystemOneResponse,
)

from tests.live.gate import gate_live
from typevet.adapters.inbound.backend_settings import load_vllm_settings, open_judgment
from typevet.adapters.inbound.judgevet import JudgmentSession, provider_factory
from typevet.adapters.inbound.settings import load_llama_settings
from typevet.adapters.outbound.gemma import classify_served_template
from typevet.adapters.outbound.judgment_scoring import ScoringJudgmentAdapter
from typevet.adapters.outbound.llama_cpp.scoring import LlamaCppCandidateScoringAdapter
from typevet.ports.judgment import JudgmentPort
from typevet_evals.runner.live_gate import require_live_enabled

_PINNED_LLAMA_MODEL = "gemma-4-31b-24gib-kv11-decoder"
_N_VOCAB = 262144
_TOLERANCE = 1e-6
_STATE = (
    "Support ticket: I was charged twice for my March subscription. "
    "Please refund the duplicate payment as soon as possible."
)
_URGENCY_LEVELS = (
    "Not urgent",
    "Slightly urgent",
    "Moderately urgent",
    "Very urgent",
    "Critical",
)
_QUESTIONS = {
    "duplicate_charge": Noul(
        instructions="Does the ticket report a duplicate charge?",
    ),
    "route": Choice(
        instructions="Which team should handle this ticket?",
        criteria={
            "billing": "Payments, refunds and invoices.",
            "technical": "Faults, errors and outages.",
            "sales": "New plans, upgrades and quotes.",
        },
    ),
    "urgency": Score(
        instructions="How urgent is this ticket?",
        criteria=_URGENCY_LEVELS,
    ),
}


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
    open_session: Callable[[], AbstractContextManager[JudgmentSession]]
    served: list[dict[str, object]]
    facts: dict[str, object] = field(default_factory=dict)
    secret: str | None = None


@dataclass(frozen=True)
class _TextSession:
    """Text-only llama.cpp session with the ``port`` the bridge reads.

    Attributes:
        port (JudgmentPort): Scoring-backed judgment pinned to one model.
    """

    port: JudgmentPort


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


def _llama_backend() -> _Backend:
    _require_env("TYPEVET_LLAMA__BASE_URL")
    settings = replace(load_llama_settings(), default_model=_PINNED_LLAMA_MODEL)
    gate_live(settings)
    base = settings.base_url.rstrip("/")
    timeout = max(settings.timeout, 600.0)
    model = _PINNED_LLAMA_MODEL
    served_ids: list[dict[str, object]] = []
    facts: dict[str, object] = {}

    @contextmanager
    def _open_llama() -> Iterator[JudgmentSession]:
        with httpx.Client(base_url=base, timeout=timeout) as client:
            served_ids.extend(e for e in _served_ids(client) if e["id"] == model)
            props = client.get("/props", params={"model": model})
            props_body = props.raise_for_status().json()
            facts["model_alias"] = props_body.get("model_alias")
            facts["build_info"] = props_body.get("build_info")
            rendered = client.post(
                "/apply-template",
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": "hello"}],
                    "add_generation_prompt": True,
                },
            )
            served = classify_served_template(
                rendered.raise_for_status().json()["prompt"]
            )
            facts["served_template_class"] = served.value

            def tokenize(text: str) -> tuple[int, ...]:
                body = client.post(
                    "/tokenize",
                    json={"model": model, "content": text, "add_special": False},
                )
                return tuple(body.raise_for_status().json()["tokens"])

            with LlamaCppCandidateScoringAdapter(
                base_url=base, timeout=timeout, client=client, n_vocab=_N_VOCAB
            ) as scorer:
                yield _TextSession(
                    ScoringJudgmentAdapter(
                        scorer,
                        tokenize_content=tokenize,
                        served_template=served,
                        pinned_model=model,
                    )
                )

    return _Backend("llama_cpp", model, _open_llama, served_ids, facts)


def _vllm_backend() -> _Backend:
    _require_env("TYPEVET_VLLM__BASE_URL")
    environ = {**os.environ, "TYPEVET_BACKEND": "vllm"}
    settings = load_vllm_settings(environ)
    served_ids: list[dict[str, object]] = []

    @contextmanager
    def _open_vllm() -> Iterator[JudgmentSession]:
        with open_judgment(environ) as session:
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


_BACKENDS: dict[str, Callable[[], _Backend]] = {
    "llama_cpp": _llama_backend,
    "vllm": _vllm_backend,
}


def _answers_summary(response: SystemOneResponse) -> dict[str, object]:
    noul = response.answers["duplicate_charge"]
    choice = response.answers["route"]
    score = response.answers["urgency"]
    assert isinstance(noul, NoulAnswer)
    assert isinstance(choice, ChoiceAnswer)
    assert isinstance(score, ScoreAnswer)
    return {
        "duplicate_charge": {"type": "noul", "noul": noul.noul},
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


@pytest.mark.live
@pytest.mark.parametrize("backend_name", sorted(_BACKENDS))
def test_judgevet_bridge_live_receipt(backend_name: str) -> None:
    """One judgevet ``system_one`` call returns typed answers on a real backend."""
    backend = _BACKENDS[backend_name]()
    factory = provider_factory(backend.open_session)
    with factory() as port:
        started = time.perf_counter()
        response = port.system_one(_STATE, _QUESTIONS, backend.model)
        wall_seconds = time.perf_counter() - started

    answers = _answers_summary(response)
    receipt: dict[str, object] = {
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
        "wall_seconds": round(wall_seconds, 3),
    }
    text = json.dumps(receipt, indent=2, sort_keys=True)
    if backend.secret:
        assert backend.secret not in text
    _write_receipt(backend.name, text)
    print(text)

    served_ids = {entry["id"] for entry in backend.served}
    assert response.model == backend.model
    assert response.model in served_ids
    assert set(response.answers) == set(_QUESTIONS)
    noul = response.answers["duplicate_charge"]
    choice = response.answers["route"]
    score = response.answers["urgency"]
    assert isinstance(noul, NoulAnswer)
    assert isinstance(choice, ChoiceAnswer)
    assert isinstance(score, ScoreAnswer)
    assert math.isfinite(noul.noul) and 0.0 <= noul.noul <= 1.0
    _assert_distribution(choice.probabilities, {"billing", "technical", "sales"})
    assert choice.choice in choice.probabilities
    assert len(score.legend) == len(_URGENCY_LEVELS)
    _assert_distribution(score.probabilities, set(score.legend))
    levels = sorted(score.legend)
    assert math.isfinite(score.score) and levels[0] <= score.score <= levels[-1]
