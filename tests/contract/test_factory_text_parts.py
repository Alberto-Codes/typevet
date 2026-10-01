"""Contract tests: both judgment factories pass ``text_parts`` through (#373).

The vLLM factory and the llama.cpp Gemma native vision factory take a
``text_parts`` keyword. The templates reach the scoring adapter: the scored
prefix holds the substituted text, and the response receipt holds each
template digest. A scripted fake replaces the scoring transport, so no test
sends a scoring request.

Examples:
    ```bash
    uv run pytest -q tests/contract/test_factory_text_parts.py
    ```

See Also:
    - [typevet.adapters.outbound.vllm.judgment_factory][]: vLLM factory
    - [typevet.adapters.outbound.llama_cpp.gemma_native_vision_factory][]:
      llama.cpp factory
    - [typevet.domain.text_parts][]: Templates and receipt
"""

from __future__ import annotations

import hashlib
import math

import httpx
import pytest

from tests.fixtures.judgment_scoring_contract import SequentialScoringFake
from typevet.adapters.inbound.settings import LlamaSettings
from typevet.adapters.outbound.gemma.served_template import GEMMA4_NO_THINKING_PREFILL
from typevet.adapters.outbound.llama_cpp.gemma_native_vision_factory import (
    open_gemma_native_vision_judgment,
)
from typevet.adapters.outbound.vllm.judgment_factory import open_vllm_judgment
from typevet.domain.judgment_questions import Noul
from typevet.domain.text_parts import CONTEXT_TEMPLATE, OPTION_BLOCK, TextParts
from typevet.ports.scoring import CandidateScoringPort

pytestmark = pytest.mark.contract

_MODEL = "factory-judge"
_STATE = "Charged twice."
_QUESTIONS = {"flagged": Noul(instructions="Billing issue?")}
_OPTION_BLOCK = (
    "Field {name}. {question}\n- {control} → {label}{description}\n{answer_rule}"
)
_CONTEXT_TEMPLATE = "Text under review:\n{context}\n---\n{field_block}\nDecide."
_PARTS = TextParts(option_block=_OPTION_BLOCK, context_template=_CONTEXT_TEMPLATE)
_GEMMA4_RENDERED = "<|turn>user\nhello<turn|>\n<|turn>model\n"


def _tokenize(text: str) -> tuple[int, ...]:
    return (ord(text[0]),) if text else ()


def _digest(template: str) -> str:
    return hashlib.sha256(template.encode("utf-8")).hexdigest()


def _router(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/apply-template":
        return httpx.Response(200, json={"prompt": _GEMMA4_RENDERED})
    if request.url.path == "/props":
        return httpx.Response(
            200,
            json={
                "modalities": {"vision": True, "audio": False},
                "media_marker": "<__media_contract__>",
            },
        )
    return httpx.Response(404, json={"error": "unexpected path"})


def _fake() -> SequentialScoringFake:
    return SequentialScoringFake([{"True": math.log(0.7), "False": math.log(0.3)}])


def _assert_parts_reached_the_adapter(
    fake: SequentialScoringFake, receipt: dict[str, str]
) -> str:
    assert receipt == {
        OPTION_BLOCK: _digest(_OPTION_BLOCK),
        CONTEXT_TEMPLATE: _digest(_CONTEXT_TEMPLATE),
    }
    [call] = fake.calls
    assert f"Text under review:\n{_STATE}\n---\nField flagged. " in call.prefix
    assert "\nDecide." in call.prefix
    return call.prefix


def test_vllm_factory_passes_text_parts_to_the_adapter_and_the_receipt() -> None:
    fake = _fake()

    def wrap(port: CandidateScoringPort) -> CandidateScoringPort:
        del port
        return fake

    with (
        httpx.Client(
            transport=httpx.MockTransport(_router), base_url="http://vllm.test"
        ) as client,
        open_vllm_judgment(
            client=client,
            model=_MODEL,
            tokenize_content=_tokenize,
            scoring_port_wrapper=wrap,
            text_parts=_PARTS,
        ) as session,
    ):
        response = session.port.judge(_STATE, _QUESTIONS, _MODEL)
    prefix = _assert_parts_reached_the_adapter(fake, response.text_parts)
    assert prefix.endswith("\nDecide.")


def test_llama_cpp_factory_passes_text_parts_to_the_adapter_and_the_receipt() -> None:
    fake = _fake()

    def wrap(port: CandidateScoringPort) -> CandidateScoringPort:
        del port
        return fake

    settings = LlamaSettings(base_url="http://offline-router", timeout=30.0)
    with (
        httpx.Client(
            transport=httpx.MockTransport(_router), base_url="http://offline-router"
        ) as client,
        open_gemma_native_vision_judgment(
            settings=settings,
            model=_MODEL,
            http_client=client,
            tokenize_content=_tokenize,
            scoring_port_wrapper=wrap,
            text_parts=_PARTS,
        ) as session,
    ):
        response = session.port.judge(_STATE, _QUESTIONS, _MODEL)
    prefix = _assert_parts_reached_the_adapter(fake, response.text_parts)
    assert prefix.endswith(GEMMA4_NO_THINKING_PREFILL)
