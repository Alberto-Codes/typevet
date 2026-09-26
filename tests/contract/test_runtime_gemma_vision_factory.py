"""Contract tests: runtime Gemma native vision factory ([#174][i174]).

Examples:
    ```bash
    uv run pytest -q tests/contract/test_runtime_gemma_vision_factory.py
    ```

See Also:
    - [typevet.runtime.llama_cpp_gemma_vision][]: factory helpers
"""

from __future__ import annotations

import json

import httpx
import pytest

from typevet.adapters.inbound.settings import LlamaSettings
from typevet.runtime.llama_cpp_gemma_vision import (
    open_gemma_native_vision_judgment,
    probe_gemma_native_vision_support,
)

pytestmark = pytest.mark.contract

_GEMMA4_RENDERED = "<|turn>user\nhello<turn|>\n<|turn>model\n"
_MODEL = "gemma-4-31b-kv9-q4km-mm"
_ROUTER_MARKER = "<__media_contract__>"


class _Router:
    """Offline router stub for factory contract tests.

    Attributes:
        paths (list[str]): Request paths observed in call order.
    """

    def __init__(self) -> None:
        self.paths: list[str] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        """Return canned llama.cpp responses for template, props, and completion.

        Returns:
            Synthetic ``httpx.Response`` for supported routes.
        """
        path = request.url.path
        self.paths.append(path)
        if path == "/apply-template":
            return httpx.Response(200, json={"prompt": _GEMMA4_RENDERED})
        if path == "/props":
            return httpx.Response(
                200,
                json={
                    "modalities": {"vision": True, "audio": False},
                    "media_marker": _ROUTER_MARKER,
                },
            )
        if path == "/tokenize":
            content = json.loads(request.content.decode())["content"]
            token_id = 100 + (sum(map(ord, content)) % 50)
            return httpx.Response(200, json={"tokens": [token_id]})
        if path == "/completion":
            body = json.loads(request.content.decode())
            wanted = body["logit_bias"][0]["id"] if body.get("logit_bias") else 100
            top = [
                {"id": wanted, "logprob": -0.2},
                {"id": wanted + 1, "logprob": -1.5},
            ]
            return httpx.Response(
                200,
                json={
                    "completion_probabilities": [{"top_logprobs": top}],
                    "tokens_evaluated": 12,
                },
            )
        return httpx.Response(404)


def _settings() -> LlamaSettings:
    return LlamaSettings(base_url="http://offline-router", timeout=30.0)


@pytest.mark.unit
def test_probe_and_open_require_gemma4_native_turn() -> None:
    """Factory accepts Gemma 4 native template and exposes a judgment port."""
    router = _Router()
    transport = httpx.MockTransport(router.handle)
    with httpx.Client(transport=transport, base_url="http://offline-router") as client:
        meta = probe_gemma_native_vision_support(
            settings=_settings(),
            model=_MODEL,
            http_client=client,
        )
        assert meta["ok"] is True
        assert meta["served"] == "native_gemma4_turn"
        ctx = open_gemma_native_vision_judgment(
            settings=_settings(),
            model=_MODEL,
            http_client=client,
        )
        with ctx as session:
            assert session.served.value == "native_gemma4_turn"
            assert session.port is not None


@pytest.mark.unit
def test_open_rejects_chatml_template() -> None:
    """Unsupported template families fail before scoring."""

    def handle(request: httpx.Request) -> httpx.Response:
        """Return vision-capable props and a ChatML template render.

        Returns:
            Synthetic ``httpx.Response`` for supported routes.
        """
        if request.url.path == "/props":
            return httpx.Response(
                200,
                json={
                    "modalities": {"vision": True, "audio": False},
                    "media_marker": _ROUTER_MARKER,
                },
            )
        if request.url.path == "/apply-template":
            return httpx.Response(
                200,
                json={"prompt": "<|im_start|>user\nhello"},
            )
        return httpx.Response(404)

    transport = httpx.MockTransport(handle)
    with (
        httpx.Client(transport=transport, base_url="http://offline-router") as client,
        pytest.raises(ValueError, match="NATIVE_GEMMA4_TURN"),
    ):
        probe_gemma_native_vision_support(
            settings=_settings(),
            model=_MODEL,
            http_client=client,
        )
