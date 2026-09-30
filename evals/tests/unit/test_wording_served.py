"""Unit checks for the served-template probes and the native-template rule (#327).

Mocked transports stand in for the llama.cpp router and the vLLM server, so
the probes run offline.

Examples:
    ```bash
    uv run pytest -q evals/tests/unit/test_wording_served.py
    ```

See Also:
    - [typevet_evals.wording.served][]: the probes and the rule
"""

from __future__ import annotations

import json
from collections.abc import Mapping

import httpx
import pytest

from typevet.adapters.outbound.gemma import ServedTemplateClass
from typevet_evals.wording.served import (
    ALLOW_DEGRADED_ENV,
    TEXT_JUDGE,
    probe_llama_template,
    probe_vllm_template,
    require_native_template,
)

pytestmark = pytest.mark.unit

NATIVE = "<|turn>system\n<|think|>\n<turn|>\n<|turn>user\nhello<turn|>\n<|turn>model\n"
CHATML = "<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n"


def _client(reply: Mapping[str, object], seen: list[httpx.Request]) -> httpx.Client:
    """Return a client whose every request gets ``reply``."""

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=dict(reply))

    return httpx.Client(base_url="http://server", transport=httpx.MockTransport(handle))


def test_text_judge_is_the_alias_that_serves_the_native_template() -> None:
    assert TEXT_JUDGE == "gemma-4-31b-kv9-q4km-mm"


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        (NATIVE, ServedTemplateClass.NATIVE_GEMMA4_TURN),
        (CHATML, ServedTemplateClass.DEGRADED_CHATML),
    ],
)
def test_llama_probe_classifies_the_apply_template_prompt(
    prompt: str, expected: ServedTemplateClass
) -> None:
    seen: list[httpx.Request] = []
    with _client({"prompt": prompt}, seen) as client:
        served = probe_llama_template(client, "gemma")

    assert served is expected
    assert seen[0].url.path == "/apply-template"
    body = json.loads(seen[0].content)
    assert body["model"] == "gemma"
    assert body["add_generation_prompt"] is True
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["messages"] == [{"role": "user", "content": "hello"}]


@pytest.mark.parametrize(
    ("token_strs", "expected"),
    [
        (
            ["<bos>", "<|turn>", "user", "\n", "hello", "<turn|>", "\n", "<|turn>"],
            ServedTemplateClass.NATIVE_GEMMA4_TURN,
        ),
        (
            ["<|im_start|>", "user", "\n", "hello", "<|im_end|>"],
            ServedTemplateClass.DEGRADED_CHATML,
        ),
    ],
)
def test_vllm_probe_classifies_the_tokenized_chat_prompt(
    token_strs: list[str], expected: ServedTemplateClass
) -> None:
    seen: list[httpx.Request] = []
    reply = {"count": 3, "max_model_len": 8, "tokens": [1, 2, 3]}
    with _client({**reply, "token_strs": token_strs}, seen) as client:
        served = probe_vllm_template(client, "google/gemma-4-31B-it")

    assert served is expected
    assert seen[0].url.path == "/tokenize"
    body = json.loads(seen[0].content)
    assert body["model"] == "google/gemma-4-31B-it"
    assert body["messages"] == [{"role": "user", "content": "hello"}]
    assert body["add_generation_prompt"] is True
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["return_token_strs"] is True


def test_vllm_probe_refuses_a_reply_without_token_strings() -> None:
    reply = {"count": 1, "max_model_len": 8, "tokens": [1], "token_strs": None}
    with (
        _client(reply, []) as client,
        pytest.raises(TypeError, match="token_strs"),
    ):
        probe_vllm_template(client, "gemma")


def test_native_gemma4_passes_without_the_override() -> None:
    served = require_native_template(ServedTemplateClass.NATIVE_GEMMA4_TURN, {})

    assert served is ServedTemplateClass.NATIVE_GEMMA4_TURN


@pytest.mark.parametrize(
    "served",
    [
        ServedTemplateClass.DEGRADED_CHATML,
        ServedTemplateClass.UNSUPPORTED,
        ServedTemplateClass.NATIVE_GEMMA3_TURN,
    ],
)
def test_other_templates_are_refused_without_the_override(
    served: ServedTemplateClass,
) -> None:
    with pytest.raises(ValueError, match=ALLOW_DEGRADED_ENV):
        require_native_template(served, {ALLOW_DEGRADED_ENV: "0"})


def test_the_override_lets_degraded_chatml_through() -> None:
    served = require_native_template(
        ServedTemplateClass.DEGRADED_CHATML, {ALLOW_DEGRADED_ENV: "1"}
    )

    assert served is ServedTemplateClass.DEGRADED_CHATML
