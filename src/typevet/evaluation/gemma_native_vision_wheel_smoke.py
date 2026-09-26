"""Offline wheel smoke for ``open_gemma_native_vision_judgment`` ([#177][i177], [#196][i196]).

Examples:
    ```python
    from typevet.evaluation.gemma_native_vision_wheel_smoke import run_wheel_smoke

    assert run_wheel_smoke() == 0
    ```

See Also:
    - [typevet.adapters.outbound.gemma_native_vision_factory][]: factory entry

[i177]: https://github.com/Alberto-Codes/typevet/issues/177
[i196]: https://github.com/Alberto-Codes/typevet/issues/196
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import httpx

from typevet.adapters.inbound.settings import LlamaSettings
from typevet.adapters.outbound.gemma_native_vision_factory import (
    open_gemma_native_vision_judgment,
)
from typevet.domain.errors import JudgmentValidationError
from typevet.domain.judgment_answers import NoulAnswer
from typevet.domain.judgment_questions import Noul
from typevet.domain.media import ImageInput

_GEMMA4_RENDERED = "<|turn>user\nhello<turn|>\n<|turn>model\n"
_ROUTER_MARKER = "<__wheel_smoke_media__>"
_PINNED = "wheel-smoke-model"
_OTHER = "other-model"


@dataclass
class _Router:
    """Minimal llama.cpp stub for isolated wheel smoke.

    Attributes:
        paths (list[str]): Request paths observed in order.
        completion_calls (int): Count of ``/completion`` responses served.

    Examples:
        ```python
        router = _Router(paths=[])
        assert router.completion_calls == 0
        ```
    """

    paths: list[str]
    completion_calls: int = 0

    def handle(self, request: httpx.Request) -> httpx.Response:
        """Return canned router responses.

        Returns:
            Synthetic response for supported routes.
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
            self.completion_calls += 1
            top = [
                {"id": token_id, "logprob": -0.2 - (token_id % 5) * 0.15}
                for token_id in range(100, 180)
            ]
            return httpx.Response(
                200,
                json={
                    "completion_probabilities": [{"top_logprobs": top}],
                    "tokens_evaluated": 8,
                },
            )
        return httpx.Response(404)


def run_wheel_smoke() -> int:
    """Exercise the public factory from an installed wheel without checkout imports.

    Returns:
        ``0`` when text, image, and negative model checks pass; ``1`` otherwise.
    """
    router = _Router(paths=[])
    transport = httpx.MockTransport(router.handle)
    settings = LlamaSettings(base_url="http://wheel-smoke", timeout=5.0)
    with (
        httpx.Client(transport=transport, base_url=settings.base_url) as client,
        open_gemma_native_vision_judgment(
            settings=settings,
            model=_PINNED,
            http_client=client,
        ) as session,
    ):
        text_resp = session.port.judge(
            "text state",
            {"q": Noul(instructions="Question?")},
            _PINNED,
        )
        answer = text_resp.answers["q"]
        if not isinstance(answer, NoulAnswer):
            return 1
        png = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32
        media = (ImageInput(data=png, mime_type="image/png"),)
        image_resp = session.port.judge(
            "image state",
            {"q": Noul(instructions="Image question?")},
            _PINNED,
            media=media,
        )
        if not isinstance(image_resp.answers["q"], NoulAnswer):
            return 1
        before = len(router.paths)
        try:
            session.port.judge("x", {"q": Noul()}, _OTHER)
        except JudgmentValidationError:
            pass
        else:
            return 1
        if "/tokenize" in router.paths[before:]:
            return 1
    print(
        json.dumps({"factory_smoke": "ok", "completion_calls": router.completion_calls})
    )
    return 0
