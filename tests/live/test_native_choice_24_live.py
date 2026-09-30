"""Live 24-option native Choice on local llama.cpp with a receipt (#288).

One opt-in call runs a 24-option native ``Choice`` end to end through
``ScoringJudgmentAdapter`` on the pinned text-only Gemma 4 router model. The
state names option 20, whose control is the letter ``"K"``. The test reads
each option's control from its option line in the rendered prompt and the
vocabulary size from ``meta.n_vocab`` of the model's ``/v1/models`` entry.
The test asserts structure only: the answer is one of the 24 labels and the
24 probabilities are finite and sum to 1. It does not assert the answer, and one run is not a
calibration claim. Set ``TYPEVET_LIVE_RECEIPT_DIR`` to write the receipt JSON.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import httpx
import pytest

from tests.live.gate import gate_live
from typevet.adapters.inbound.settings import load_llama_settings
from typevet.adapters.outbound.gemma import (
    ServedTemplateClass,
    classify_served_template,
)
from typevet.adapters.outbound.judgment_scoring import ScoringJudgmentAdapter
from typevet.adapters.outbound.llama_cpp.scoring import LlamaCppCandidateScoringAdapter
from typevet.domain.candidate_scoring_request import CandidateScoringRequest
from typevet.domain.candidate_scoring_response import CandidateScoringResult
from typevet.domain.judgment_questions import Choice

_LLAMA = load_llama_settings()
_PINNED_MODEL = "gemma-4-31b-24gib-kv11-decoder"
_CRITERIA = {
    "groceries": "Food and household staples bought for the kitchen.",
    "clothing": "Garments, shoes and fashion accessories.",
    "electronics": "Phones, laptops, televisions and similar devices.",
    "furniture": "Tables, chairs, sofas, beds and shelving.",
    "books": "Printed or digital books and magazines.",
    "toys": "Games and playthings for children.",
    "sporting_goods": "Equipment for sports and fitness.",
    "beauty": "Cosmetics, skin care and fragrance.",
    "pharmacy": "Medicines and health supplies.",
    "automotive": "Car parts, tyres and vehicle care.",
    "garden": "Plants, seeds, soil and outdoor tools.",
    "pet_supplies": "Food, toys and care products for pets.",
    "office_supplies": "Paper, pens, printers and desk items.",
    "jewelry": "Rings, necklaces, watches and earrings.",
    "music_instruments": "Guitars, pianos, drums and accessories.",
    "baby_products": "Diapers, strollers and infant care.",
    "hardware_tools": "Hammers, drills, screws and building tools.",
    "kitchenware": "Pots, pans, knives and cooking utensils.",
    "luggage": "Suitcases, backpacks and travel bags.",
    "video_games": "Consoles and game software.",
    "camping_gear": "Tents, sleeping bags, camp stoves and hiking packs.",
    "art_supplies": "Paints, brushes, canvas and drawing materials.",
    "cleaning_products": "Detergents, mops and household cleaners.",
    "stationery_cards": "Greeting cards, gift wrap and envelopes.",
}
_TARGET = "camping_gear"
_STATE = (
    "Order note: the customer bought a two-person tent, a down sleeping bag "
    "and a portable camp stove for a weekend hiking trip in the mountains."
)
_QUESTION = Choice(
    instructions="Which product category best fits this order?",
    criteria=_CRITERIA,
)
_TOLERANCE = 1e-6


@dataclass
class _RecordingPort:
    """Forward scoring calls and keep each request for the receipt.

    Attributes:
        inner (LlamaCppCandidateScoringAdapter): Real llama.cpp scorer.
        requests (list[CandidateScoringRequest]): Requests in call order.
    """

    inner: LlamaCppCandidateScoringAdapter
    requests: list[CandidateScoringRequest] = field(default_factory=list)

    def score_candidates(
        self, request: CandidateScoringRequest
    ) -> CandidateScoringResult:
        """Record ``request`` and score it on the real adapter.

        Returns:
            The real adapter result, unchanged.
        """
        self.requests.append(request)
        return self.inner.score_candidates(request)


@pytest.fixture
def gemma4_choice_model() -> str:
    """Require the router and the pinned Gemma 4 id in the catalog."""
    gate_live(replace(_LLAMA, default_model=_PINNED_MODEL))
    return _PINNED_MODEL


def _router_props(client: httpx.Client, model: str) -> dict[str, object]:
    return client.get("/props", params={"model": model}).raise_for_status().json()


def _props_summary(props: dict[str, object]) -> dict[str, object]:
    return {
        "model_alias": props.get("model_alias"),
        "model_ftype": props.get("model_ftype"),
        "model_path": props.get("model_path"),
        "build_info": props.get("build_info"),
        "modalities": props.get("modalities"),
        "chat_template_sha256": hashlib.sha256(
            str(props.get("chat_template", "")).encode()
        ).hexdigest(),
    }


def _models_n_vocab(models: dict[str, object], model: str) -> int:
    """Read ``meta.n_vocab`` from the ``/v1/models`` entry for ``model``.

    Args:
        models: Parsed ``GET /v1/models`` body from the router.
        model: Served model id.

    Returns:
        The positive ``n_vocab`` of that entry.

    Raises:
        AssertionError: When no single entry has id ``model``, or its
            ``meta.n_vocab`` is absent or not a positive int.
    """
    data = models.get("data")
    rows = data if isinstance(data, list) else []
    entries = [e for e in rows if isinstance(e, dict) and e.get("id") == model]
    assert len(entries) == 1, f"/v1/models has {len(entries)} entries for {model}"
    meta = entries[0].get("meta")
    size = meta.get("n_vocab") if isinstance(meta, dict) else None
    assert type(size) is int and size > 0, f"{model} meta.n_vocab is {size!r}"
    return size


def _prompt_controls(prompt: str, labels: Sequence[str]) -> dict[str, str]:
    """Read each label's control from its option line in the rendered prompt.

    An option line is ``<control> → <label>`` or ``Control <control> →
    <label>``, with an optional ``: <description>`` after the label.

    Args:
        prompt: Rendered prompt that the scorer received.
        labels: Option labels in question order.

    Returns:
        Label to control, in ``labels`` order.

    Raises:
        AssertionError: When a label has no option line or more than one.
    """
    controls: dict[str, str] = {}
    for label in labels:
        line = re.compile(
            rf"^(?:Control )?([0-9A-Z]) → {re.escape(label)}(?::|$)", re.MULTILINE
        )
        found = line.findall(prompt)
        assert len(found) == 1, f"prompt has {len(found)} option lines for {label}"
        controls[label] = found[0]
    return controls


def _served_class(client: httpx.Client, model: str) -> ServedTemplateClass:
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
    return classify_served_template(rendered)


def _write_receipt(receipt: dict[str, object]) -> None:
    out_dir = os.environ.get("TYPEVET_LIVE_RECEIPT_DIR")
    if not out_dir:
        return
    path = Path(out_dir) / "native_choice_24_receipt.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")


@pytest.mark.live
def test_native_choice_24_options_live_receipt(gemma4_choice_model: str) -> None:
    """One 24-option native Choice call returns a valid full distribution."""
    labels = list(_CRITERIA)
    assert len(labels) == 24
    base = _LLAMA.base_url.rstrip("/")
    timeout = max(_LLAMA.timeout, 600.0)
    with httpx.Client(base_url=base, timeout=timeout) as client:
        props = _props_summary(_router_props(client, gemma4_choice_model))
        models = client.get("/v1/models").raise_for_status().json()
        n_vocab = _models_n_vocab(models, gemma4_choice_model)
        served = _served_class(client, gemma4_choice_model)

        def tokenize(text: str) -> tuple[int, ...]:
            body = client.post(
                "/tokenize",
                json={
                    "model": gemma4_choice_model,
                    "content": text,
                    "add_special": False,
                },
            )
            return tuple(body.raise_for_status().json()["tokens"])

        with LlamaCppCandidateScoringAdapter(
            base_url=base, timeout=timeout, client=client, n_vocab=n_vocab
        ) as scorer:
            recorder = _RecordingPort(scorer)
            port = ScoringJudgmentAdapter(
                recorder,
                tokenize_content=tokenize,
                served_template=served,
                pinned_model=gemma4_choice_model,
            )
            started = time.perf_counter()
            response = port.judge(_STATE, {"category": _QUESTION}, gemma4_choice_model)
            wall_seconds = time.perf_counter() - started

    answer = response.choices["category"]
    distribution = dict(answer.probabilities)
    (request,) = recorder.requests
    controls = _prompt_controls(request.prefix, labels)
    digits = [label for label in labels if controls[label].isdigit()]
    ranked = sorted(distribution.items(), key=lambda item: item[1], reverse=True)
    receipt: dict[str, object] = {
        "issue": 288,
        "model": response.model,
        "router_props": props,
        "n_vocab": n_vocab,
        "served_template_class": served.value,
        "prompt_sha256": hashlib.sha256(request.prefix.encode()).hexdigest(),
        "prompt_chars": len(request.prefix),
        "candidate_token_ids": {
            spec.label: list(spec.token_ids) for spec in request.candidates
        },
        "target": _TARGET,
        "target_control": controls[_TARGET],
        "answer": answer.choice,
        "answer_control": controls[answer.choice],
        "confidence": answer.confidence,
        "distribution": {label: distribution[label] for label in labels},
        "top5": [[label, controls[label], prob] for label, prob in ranked[:5]],
        "digit_mass": sum(distribution[label] for label in digits),
        "letter_mass": sum(
            distribution[label] for label in labels if label not in digits
        ),
        "wall_seconds": round(wall_seconds, 3),
    }
    _write_receipt(receipt)
    print(json.dumps(receipt, sort_keys=True))

    assert answer.choice in labels
    assert sorted(distribution) == sorted(labels)
    probs = list(distribution.values())
    assert all(math.isfinite(p) and 0.0 <= p <= 1.0 for p in probs)
    assert abs(sum(probs) - 1.0) < _TOLERANCE
    assert [spec.label for spec in request.candidates] == labels
