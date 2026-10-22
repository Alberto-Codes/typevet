"""Unit tests: optional ChatML framing on ``open_judgment`` ([#444][i444]).

Explicit ``framing`` on the public factory and the llama.cpp factory hook
requires a served ``DEGRADED_CHATML`` family and ``require_vision=False``.
The path scores with an empty-think ChatML prefix, refuses images before any
completion, and rejects framing on fake or vLLM. Without framing, Gemma
defaults stay the same.

Examples:
    ```bash
    uv run pytest -q tests/unit/test_open_judgment_chatml_framing.py
    ```

See Also:
    - [typevet.adapters.inbound.backend_settings][]: ``open_judgment``
    - [typevet.adapters.outbound.llama_cpp.gemma_native_vision_factory][]:
      Factory hooks
    - [typevet.adapters.outbound.gemma.chatml_framing][]: ``ChatMLThinkingFraming``

[i444]: https://github.com/Alberto-Codes/typevet/issues/444
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from tests.fixtures.gemma_template_contract import (
    degraded_chatml_rendered,
    native_gemma4_rendered,
)
from typevet.adapters.inbound.backend_settings import open_judgment
from typevet.adapters.inbound.settings import load_llama_settings
from typevet.adapters.outbound.gemma import (
    CHATML_ASSISTANT_HEADER,
    CHATML_EMPTY_THINK_PREFILLS,
    CHATML_IM_END,
    CHATML_IM_START,
    ChatMLThinkingFraming,
)
from typevet.adapters.outbound.llama_cpp.gemma_native_vision_factory import (
    GemmaNativeVisionSession,
    open_gemma_native_vision_judgment,
)
from typevet.domain.errors import (
    GemmaTemplateError,
    ScoringUnsupportedCapabilityError,
)
from typevet.domain.judgment_answers import NoulAnswer
from typevet.domain.judgment_questions import Noul
from typevet.domain.media import ImageInput

pytestmark = pytest.mark.unit

_MODEL = "qwen-chatml-text"
_ENVIRON = {
    "TYPEVET_BACKEND": "llama_cpp",
    "TYPEVET_LLAMA__MULTIMODAL_MODEL": _MODEL,
    "TYPEVET_LLAMA__DEFAULT_MODEL": _MODEL,
}
_CHATML_RENDERED = degraded_chatml_rendered("hello")
_GEMMA4_RENDERED = native_gemma4_rendered("hello")
_MIXED_RENDERED = _CHATML_RENDERED + "<|turn>"
_PLAIN_RENDERED = "plain prompt"
_PNG = b"\x89PNG\r\n\x1a\nchatml-frame"
_EMPTY_THINK = CHATML_EMPTY_THINK_PREFILLS[0]


class _BareChatMLFraming:
    """Caller framing that stops at the ChatML assistant header."""

    def compose_prefix(self, *, user_text: str, media: tuple[ImageInput, ...]) -> str:
        """Return a ChatML prefix without an empty think block.

        Returns:
            Prefix that ends with the ChatML assistant header.
        """
        del media
        return f"{CHATML_IM_START}user\n{user_text}{CHATML_IM_END}\n{CHATML_ASSISTANT_HEADER}"


class _Router:
    """Offline llama.cpp router that records paths and completion prompts.

    Attributes:
        props (dict[str, Any]): Body served on ``/props``.
        template (str): Prompt string served on ``/apply-template``.
        paths (list[str]): Request paths in call order.
        completion_prompts (list[str]): Prompt strings from ``/completion``.

    Examples:
        ```python
        router = _Router({"modalities": {"vision": False}}, _CHATML_RENDERED)
        assert router.paths == []
        ```
    """

    def __init__(self, props: dict[str, Any], template: str) -> None:
        self.props = props
        self.template = template
        self.paths: list[str] = []
        self.completion_prompts: list[str] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        """Serve props, template, tokenize and completion bodies.

        Args:
            request: The request the client sent.

        Returns:
            A synthetic llama.cpp response, or 404 for another path.
        """
        path = request.url.path
        self.paths.append(path)
        if path == "/props":
            return httpx.Response(200, json=self.props)
        if path == "/apply-template":
            return httpx.Response(200, json={"prompt": self.template})
        if path == "/tokenize":
            content = json.loads(request.content.decode())["content"]
            return httpx.Response(
                200, json={"tokens": [100 + sum(map(ord, content)) % 50]}
            )
        if path == "/completion":
            body = json.loads(request.content.decode())
            prompt = body["prompt"]
            assert isinstance(prompt, str)
            self.completion_prompts.append(prompt)
            top = [{"id": i, "logprob": -0.2 - (i % 5) * 0.15} for i in range(100, 180)]
            return httpx.Response(
                200, json={"completion_probabilities": [{"top_logprobs": top}]}
            )
        return httpx.Response(404)


def _text_props(*, vision: bool = False) -> dict[str, Any]:
    props: dict[str, Any] = {"modalities": {"vision": vision, "audio": False}}
    if vision:
        props["media_marker"] = "<__media_444__>"
    return props


def test_factory_hook_accepts_chatml_with_framing() -> None:
    """Lower factory with ChatML framing opens and scores an empty-think prefix."""
    router = _Router(_text_props(), _CHATML_RENDERED)
    settings = load_llama_settings(_ENVIRON)
    with (
        httpx.Client(
            base_url="http://router.test", transport=httpx.MockTransport(router.handle)
        ) as client,
        open_gemma_native_vision_judgment(
            settings=settings,
            require_gemma4=False,
            require_vision=False,
            framing=ChatMLThinkingFraming(),
            http_client=client,
        ) as session,
    ):
        assert session.served.value == "degraded_chatml"
        response = session.port.judge("Charged twice.", {"q": Noul()}, _MODEL)
    assert isinstance(response.answers["q"], NoulAnswer)
    assert router.completion_prompts[0].endswith(
        f"{CHATML_ASSISTANT_HEADER}{_EMPTY_THINK}"
    )


def test_open_judgment_chatml_framing_scores_with_empty_think_prefix() -> None:
    """Public ``open_judgment`` with framing completes on an empty-think prefix."""
    router = _Router(_text_props(), _CHATML_RENDERED)
    with open_judgment(
        _ENVIRON,
        transport=httpx.MockTransport(router.handle),
        framing=ChatMLThinkingFraming(),
    ) as session:
        assert isinstance(session, GemmaNativeVisionSession)
        assert session.served.value == "degraded_chatml"
        assert session.capability.vision is False
        response = session.port.judge("Charged twice.", {"q": Noul()}, _MODEL)
    assert isinstance(response.answers["q"], NoulAnswer)
    assert router.paths.count("/completion") == 1
    assert router.completion_prompts[0].endswith(
        f"{CHATML_ASSISTANT_HEADER}{_EMPTY_THINK}"
    )


def test_bare_chatml_framing_is_refused_before_completion() -> None:
    """A bare ChatML framing hits the prefill guard before any completion."""
    router = _Router(_text_props(), _CHATML_RENDERED)
    with (
        open_judgment(
            _ENVIRON,
            transport=httpx.MockTransport(router.handle),
            framing=_BareChatMLFraming(),
        ) as session,
        pytest.raises(GemmaTemplateError, match="_BareChatMLFraming"),
    ):
        session.port.judge("Charged twice.", {"q": Noul()}, _MODEL)
    assert "/completion" not in router.paths
    assert router.completion_prompts == []


def test_no_framing_keeps_gemma_chatml_refusal() -> None:
    """Without framing, ChatML stays refused by the native family guard."""
    router = _Router(_text_props(vision=True), _CHATML_RENDERED)
    with (
        pytest.raises(ValueError, match=r"unsupported served template|NATIVE_GEMMA4"),
        open_judgment(_ENVIRON, transport=httpx.MockTransport(router.handle)),
    ):
        pass


def test_framing_rejects_gemma_served_family() -> None:
    """Framing on a Gemma served template is refused."""
    router = _Router(_text_props(), _GEMMA4_RENDERED)
    with (
        pytest.raises(ValueError, match=r"degraded_chatml|framing"),
        open_judgment(
            _ENVIRON,
            transport=httpx.MockTransport(router.handle),
            framing=ChatMLThinkingFraming(),
        ),
    ):
        pass
    assert "/completion" not in router.paths


def test_framing_rejects_mixed_served_family() -> None:
    """Framing on a mixed served template is refused."""
    router = _Router(_text_props(), _MIXED_RENDERED)
    with (
        pytest.raises(ValueError, match=r"degraded_chatml|framing|unsupported"),
        open_judgment(
            _ENVIRON,
            transport=httpx.MockTransport(router.handle),
            framing=ChatMLThinkingFraming(),
        ),
    ):
        pass


def test_framing_rejects_unsupported_served_family() -> None:
    """Framing on an unsupported served template is refused."""
    router = _Router(_text_props(), _PLAIN_RENDERED)
    with (
        pytest.raises(ValueError, match=r"degraded_chatml|framing|unsupported"),
        open_judgment(
            _ENVIRON,
            transport=httpx.MockTransport(router.handle),
            framing=ChatMLThinkingFraming(),
        ),
    ):
        pass


def test_framed_port_refuses_images_when_server_reports_vision() -> None:
    """A framed port refuses images before completion even when vision is on."""
    router = _Router(_text_props(vision=True), _CHATML_RENDERED)
    image = ImageInput(data=_PNG, mime_type="image/png")
    with open_judgment(
        _ENVIRON,
        transport=httpx.MockTransport(router.handle),
        framing=ChatMLThinkingFraming(),
    ) as session:
        assert isinstance(session, GemmaNativeVisionSession)
        assert session.capability.vision is True
        opened = list(router.paths)
        with pytest.raises(ScoringUnsupportedCapabilityError):
            session.port.judge("Receipt.", {"q": Noul()}, _MODEL, media=(image,))
    assert router.paths == opened
    assert "/completion" not in router.paths


def test_framing_rejected_on_fake_backend() -> None:
    """Framing on the fake backend raises ``ValueError``."""
    with (
        pytest.raises(ValueError, match="framing"),
        open_judgment({"TYPEVET_BACKEND": "fake"}, framing=ChatMLThinkingFraming()),
    ):
        pass


def test_framing_rejected_on_vllm_backend() -> None:
    """Framing on the vLLM backend raises ``ValueError``."""
    environ = {
        "TYPEVET_BACKEND": "vllm",
        "TYPEVET_VLLM__BASE_URL": "http://vllm.test",
        "TYPEVET_VLLM__MODEL": "served",
    }
    with (
        pytest.raises(ValueError, match="framing"),
        open_judgment(environ, framing=ChatMLThinkingFraming()),
    ):
        pass


def test_factory_framing_requires_require_vision_false() -> None:
    """Framing with ``require_vision=True`` fails before any router request."""
    router = _Router(_text_props(), _CHATML_RENDERED)
    settings = load_llama_settings(_ENVIRON)
    with (
        httpx.Client(
            base_url="http://router.test", transport=httpx.MockTransport(router.handle)
        ) as client,
        pytest.raises(ValueError, match="require_vision"),
        open_gemma_native_vision_judgment(
            settings=settings,
            require_gemma4=False,
            framing=ChatMLThinkingFraming(),
            http_client=client,
        ),
    ):
        pass
    assert router.paths == []
