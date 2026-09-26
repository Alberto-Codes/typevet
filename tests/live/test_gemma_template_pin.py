"""Live receipt: pin served template class from local ``/apply-template`` (#118)."""

from __future__ import annotations

import httpx
import pytest

from tests.live.gate import gate_live
from typevet.adapters.inbound.settings import load_llama_settings
from typevet.gemma_answer_binding import resolve_answer_anchor
from typevet.gemma_served_template import ServedTemplateClass

_SETTINGS = load_llama_settings()
_PINNED_MODEL = "gemma-4-31b-24gib-kv11-decoder"


@pytest.mark.live
def test_gemma4_served_template_class_pin() -> None:
    gate_live(_SETTINGS)
    base = _SETTINGS.base_url.rstrip("/")
    with httpx.Client(base_url=base, timeout=30.0) as client:
        rendered = (
            client.post(
                "/apply-template",
                json={
                    "model": _PINNED_MODEL,
                    "messages": [{"role": "user", "content": "Classify emotion."}],
                    "add_generation_prompt": True,
                },
            )
            .raise_for_status()
            .json()["prompt"]
        )

        def tokenize_with_special(text: str) -> list[int]:
            payload = (
                client.post(
                    "/tokenize",
                    json={"model": _PINNED_MODEL, "content": text, "add_special": True},
                )
                .raise_for_status()
                .json()
            )
            return payload["tokens"]

        anchor = resolve_answer_anchor(
            rendered,
            tokenize_with_special=tokenize_with_special,
        )
    assert anchor.template_class is ServedTemplateClass.DEGRADED_CHATML
    assert anchor.prefix_token_count > 0
