"""Opt-in live image-conditioned candidate scoring smoke (#142).

Skips when the router or the multimodal model id is unavailable. **Fails**
when the router serves that id text-only, because a text-only pass would not
prove image conditioning.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import httpx
import pytest

from tests.fixtures.synthetic_images import solid_image
from typevet.adapters.inbound.settings import load_llama_settings
from typevet.adapters.outbound.gemma import (
    ServedTemplateClass,
    classify_served_template,
)
from typevet.adapters.outbound.judgment_scoring import ScoringJudgmentAdapter
from typevet.adapters.outbound.llama_cpp_multimodal import fetch_media_capability
from typevet.adapters.outbound.llama_cpp_scoring import LlamaCppCandidateScoringAdapter
from typevet.domain.judgment_questions import Choice
from typevet.domain.judgment_response import JudgmentResponse
from typevet.evaluation.runner.live_gate import live_skip_reason

_LLAMA = load_llama_settings()
_N_VOCAB = 262144
_STATE = "Look at the attached image."
_QUESTIONS = {
    "fill": Choice(
        criteria={
            "red": "The image is filled with red.",
            "green": "The image is filled with green.",
            "blue": "The image is filled with blue.",
        },
        instructions="What single colour fills the attached image?",
    )
}
_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "scratchpad" / "multimodal"
# One Gemma 3 image costs 256 prompt tokens. A request that silently drops the
# image evaluates the marker as text and grows by tens of tokens, not hundreds.
_MIN_IMAGE_TOKENS = 200


@pytest.fixture
def live_multimodal_model() -> str:
    """Skip unless the router catalog serves the multimodal model id."""
    model = _LLAMA.multimodal_model
    reason = live_skip_reason(replace(_LLAMA, default_model=model))
    if reason is not None:
        pytest.skip(reason)
    return model


def _tokenizer(client: httpx.Client, model: str):
    def tokenize_content(text: str) -> tuple[int, ...]:
        payload = (
            client.post(
                "/tokenize",
                json={"model": model, "content": text, "add_special": False},
            )
            .raise_for_status()
            .json()
        )
        return tuple(payload["tokens"])

    return tokenize_content


def _served_template(client: httpx.Client, model: str) -> ServedTemplateClass:
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


def _probabilities(response: JudgmentResponse) -> dict[str, float]:
    return dict(response.choices["fill"].probabilities)


@pytest.mark.live
def test_image_conditioned_scoring_changes_the_judgment(
    live_multimodal_model: str,
) -> None:
    """Prove the image, not the upload, moves the scored distribution."""
    settings = replace(_LLAMA, timeout=max(_LLAMA.timeout, 900.0))
    base = settings.base_url.rstrip("/")
    results: dict[str, JudgmentResponse] = {}
    with httpx.Client(base_url=base, timeout=settings.timeout) as client:
        capability = fetch_media_capability(client, f"{base}/", live_multimodal_model)
        assert capability.vision, (
            f"{live_multimodal_model} reports text-only input modalities; "
            "a text-only router cannot prove image conditioning"
        )
        with LlamaCppCandidateScoringAdapter(
            base_url=base,
            timeout=settings.timeout,
            n_vocab=_N_VOCAB,
        ) as scoring:
            port = ScoringJudgmentAdapter(
                scoring,
                tokenize_content=_tokenizer(client, live_multimodal_model),
                served_template=_served_template(client, live_multimodal_model),
            )
            results["omitted"] = port.judge(_STATE, _QUESTIONS, live_multimodal_model)
            for colour in ("red", "green", "blue"):
                results[colour] = port.judge(
                    _STATE,
                    _QUESTIONS,
                    live_multimodal_model,
                    media=(solid_image(colour),),
                )

    omitted = _probabilities(results["omitted"])
    text_tokens = results["omitted"].usage.input_tokens
    assert text_tokens is not None, "router did not report tokens_evaluated"
    for colour in ("red", "green", "blue"):
        answer = results[colour].choices["fill"]
        # Control -> label binding: answers stay on original labels, not digits.
        assert set(answer.probabilities) == {"red", "green", "blue"}
        assert answer.choice == colour
        assert _probabilities(results[colour]) != omitted
        # Fail loud on a silent drop (#155): HTTP 200 and a valid distribution
        # say nothing about attachment; only the prompt token count does.
        image_tokens = results[colour].usage.input_tokens
        assert image_tokens is not None
        assert image_tokens - text_tokens >= _MIN_IMAGE_TOKENS, (
            f"{colour} evaluated {image_tokens} prompt tokens against a "
            f"{text_tokens}-token text baseline; the image was not attached"
        )

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (_OUTPUT_DIR / "live_receipt.json").write_text(
        json.dumps(
            {
                "model": live_multimodal_model,
                "media_marker": capability.marker,
                "vision": capability.vision,
                "choices": {
                    name: {
                        "choice": response.choices["fill"].choice,
                        "probabilities": _probabilities(response),
                        "tokens_evaluated": response.usage.input_tokens,
                    }
                    for name, response in results.items()
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
