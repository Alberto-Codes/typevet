"""Run the multimodal how-to library recipe against an offline router (#175).

The recipe block in ``docs/how-to/run-a-multimodal-live-smoke.md`` is executed
as written. A mock transport stands in for llama.cpp, so the test needs no
network and no model.
"""

from __future__ import annotations

import base64
import json
import re
import runpy
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest

from typevet.domain import JudgmentResponse, JudgmentValidationError

pytestmark = pytest.mark.contract

_HOWTO = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "how-to"
    / "run-a-multimodal-live-smoke.md"
)
_RECIPE_DEF = "def judge_image("
_TEMPLATE_KWARG = "served_template=served,"
_ROUTER_MARKER = "<__media_Rc175offline__>"
_MODEL = "gemma-3-4b-it-q4km-mm"
_PNG = b"\x89PNG\r\n\x1a\nhowto-recipe"
_GEMMA3_RENDERED = "<start_of_turn>user\nhello<end_of_turn>\n<start_of_turn>model\n"

type Recipe = Callable[[httpx.Client, str, bytes], JudgmentResponse]


def _recipe_source() -> str:
    text = _HOWTO.read_text(encoding="utf-8")
    blocks = [
        block
        for block in re.findall(r"```python\n(.*?)```", text, flags=re.DOTALL)
        if _RECIPE_DEF in block
    ]
    assert len(blocks) == 1, (
        f"expected one recipe block defining judge_image in {_HOWTO}"
    )
    return blocks[0]


def _load_recipe(source: str, tmp_path: Path) -> Recipe:
    module = tmp_path / "howto_recipe.py"
    module.write_text(source, encoding="utf-8")
    return runpy.run_path(str(module))["judge_image"]


class _Router:
    """Answer the four llama.cpp routes the recipe touches, in call order."""

    def __init__(self) -> None:
        self.paths: list[str] = []
        self.completion_bodies: list[dict[str, Any]] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.paths.append(path)
        if path == "/apply-template":
            return httpx.Response(200, json={"prompt": _GEMMA3_RENDERED})
        if path == "/tokenize":
            content = json.loads(request.content.decode())["content"]
            return httpx.Response(200, json={"tokens": [ord(c) for c in content]})
        if path == "/props":
            return httpx.Response(
                200,
                json={
                    "modalities": {"vision": True, "audio": False},
                    "media_marker": _ROUTER_MARKER,
                },
            )
        if path == "/completion":
            self.completion_bodies.append(json.loads(request.content.decode()))
            top = [
                {"id": ord(str(digit)), "logprob": -0.5 - digit} for digit in range(10)
            ]
            return httpx.Response(
                200,
                json={
                    "completion_probabilities": [{"top_logprobs": top}],
                    "tokens_evaluated": 362,
                },
            )
        return httpx.Response(404, text=f"unexpected route {path}")


def _client(router: _Router) -> httpx.Client:
    return httpx.Client(
        base_url="http://router.invalid",
        transport=httpx.MockTransport(router.handle),
    )


def test_recipe_sends_native_gemma3_media_prompt_without_prompt_cache(
    tmp_path: Path,
) -> None:
    router = _Router()
    judge_image = _load_recipe(_recipe_source(), tmp_path)

    with _client(router) as client:
        response = judge_image(client, _MODEL, _PNG)

    assert router.paths[0] == "/apply-template"
    assert router.completion_bodies, "recipe never reached /completion"
    for body in router.completion_bodies:
        assert body["cache_prompt"] is False
        prompt = body["prompt"]
        assert isinstance(prompt, dict), "image sent without the nested prompt"
        assert prompt["prompt_string"].startswith("<start_of_turn>user\n")
        assert "<|im_start|>" not in prompt["prompt_string"]
        assert prompt["prompt_string"].count(_ROUTER_MARKER) == 1
        assert prompt["multimodal_data"] == [base64.b64encode(_PNG).decode("ascii")]
    (answer,) = response.choices.values()
    assert answer.choice in answer.probabilities
    assert response.usage.input_tokens is not None


def test_recipe_without_template_setup_fails_closed_before_scoring(
    tmp_path: Path,
) -> None:
    source = _recipe_source()
    assert source.count(_TEMPLATE_KWARG) == 1, "recipe must pass served_template"
    router = _Router()
    judge_image = _load_recipe(source.replace(_TEMPLATE_KWARG, ""), tmp_path)

    with (
        _client(router) as client,
        pytest.raises(JudgmentValidationError, match="native Gemma 3"),
    ):
        judge_image(client, _MODEL, _PNG)

    assert router.completion_bodies == []
