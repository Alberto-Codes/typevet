"""Unit tests: the llama.cpp judgment accepts a text-only Gemma 4 model ([#438][i438]).

``open_judgment`` with ``TYPEVET_BACKEND=llama_cpp`` opens a session on a model
whose ``/props`` reports no image input. Text judgments then score as before.
An image judgment raises ``ScoringUnsupportedCapabilityError`` before any
request leaves the process, and the judgevet bridge maps it to
``ProviderCapabilityError``. A vision model keeps its image path. A fake
transport serves every response.

Examples:
    ```bash
    uv run pytest -q tests/unit/test_open_judgment_text_only.py
    ```

See Also:
    - [typevet.adapters.inbound.backend_settings][]: ``open_judgment``
    - [typevet.adapters.outbound.llama_cpp.gemma_native_vision_factory][]: Probe
    - [typevet.adapters.inbound.judgevet][]: ``provider_factory``

[i438]: https://github.com/Alberto-Codes/typevet/issues/438
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from judgevet.domain.media import ImageAttachment, ImageEvidence, MediaCapabilities
from judgevet.domain.questions import Noul as JevNoul
from judgevet.providers import ProviderCapabilityError

from typevet.adapters.inbound.backend_settings import open_judgment
from typevet.adapters.inbound.judgevet import TypevetMediaSystemOnePort
from typevet.adapters.inbound.settings import load_llama_settings
from typevet.adapters.outbound.llama_cpp.gemma_native_vision_factory import (
    GemmaNativeVisionSession,
    probe_gemma_native_vision_support,
)
from typevet.domain.errors import ScoringUnsupportedCapabilityError
from typevet.domain.judgment_answers import NoulAnswer
from typevet.domain.judgment_questions import Noul
from typevet.domain.media import ImageInput

pytestmark = pytest.mark.unit

_MODEL = "gemma-4-31b-kv9-text"
_ENVIRON = {"TYPEVET_BACKEND": "llama_cpp", "TYPEVET_LLAMA__MULTIMODAL_MODEL": _MODEL}
_GEMMA4_RENDERED = "<|turn>user\nhello<turn|>\n<|turn>model\n"
_TEXT_PROPS: dict[str, Any] = {"modalities": {"vision": False, "audio": False}}
_VISION_PROPS: dict[str, Any] = {
    "modalities": {"vision": True, "audio": False},
    "media_marker": "<__media_438__>",
}
_PNG = b"\x89PNG\r\n\x1a\ntext-only"


class _Router:
    """Offline llama.cpp router that records each request path.

    Attributes:
        props (dict[str, Any]): Body served on ``/props``.
        paths (list[str]): Request paths in call order.

    Examples:
        ```python
        router = _Router(_TEXT_PROPS)
        assert router.paths == []
        ```
    """

    def __init__(self, props: dict[str, Any]) -> None:
        self.props = props
        self.paths: list[str] = []

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
            return httpx.Response(200, json={"prompt": _GEMMA4_RENDERED})
        if path == "/tokenize":
            content = json.loads(request.content.decode())["content"]
            return httpx.Response(
                200, json={"tokens": [100 + sum(map(ord, content)) % 50]}
            )
        if path == "/completion":
            top = [{"id": i, "logprob": -0.2 - (i % 5) * 0.15} for i in range(100, 180)]
            body = {"completion_probabilities": [{"top_logprobs": top}]}
            return httpx.Response(200, json=body)
        return httpx.Response(404)


def test_text_only_model_opens_and_serves_text_judgment() -> None:
    """A text-only Gemma 4 model opens and answers a text question."""
    router = _Router(_TEXT_PROPS)
    with open_judgment(
        _ENVIRON, transport=httpx.MockTransport(router.handle)
    ) as session:
        assert isinstance(session, GemmaNativeVisionSession)
        assert session.capability.vision is False
        assert session.served.value == "native_gemma4_turn"
        response = session.port.judge("Charged twice.", {"q": Noul()}, _MODEL)
    assert isinstance(response.answers["q"], NoulAnswer)
    assert router.paths.count("/completion") == 1


def test_text_only_model_refuses_an_image_before_any_request() -> None:
    """An image judgment on a text-only session sends no request."""
    router = _Router(_TEXT_PROPS)
    image = ImageInput(data=_PNG, mime_type="image/png")
    with open_judgment(
        _ENVIRON, transport=httpx.MockTransport(router.handle)
    ) as session:
        opened = list(router.paths)
        with pytest.raises(ScoringUnsupportedCapabilityError, match=_MODEL):
            session.port.judge("Receipt.", {"q": Noul()}, _MODEL, media=(image,))
    assert router.paths == opened
    assert "/completion" not in router.paths


def test_bridge_maps_the_image_refusal_to_provider_capability_error() -> None:
    """The judgevet media bridge maps the refusal and sends no scoring request."""
    router = _Router(_TEXT_PROPS)
    evidence = ImageEvidence([ImageAttachment("a", _PNG, "image/png")], {"q": ["a"]})
    with open_judgment(
        _ENVIRON, transport=httpx.MockTransport(router.handle)
    ) as session:
        port = TypevetMediaSystemOnePort(
            session.port, media=MediaCapabilities({"image/png"})
        )
        opened = list(router.paths)
        with pytest.raises(ProviderCapabilityError):
            port.system_one_media(
                "Receipt.", {"q": JevNoul()}, _MODEL, evidence=evidence
            )
    assert router.paths == opened


def test_vision_model_keeps_the_image_path() -> None:
    """A vision model still sends the image to ``/completion``."""
    router = _Router(_VISION_PROPS)
    image = ImageInput(data=_PNG, mime_type="image/png")
    with open_judgment(
        _ENVIRON, transport=httpx.MockTransport(router.handle)
    ) as session:
        assert isinstance(session, GemmaNativeVisionSession)
        assert session.capability.vision is True
        response = session.port.judge("Receipt.", {"q": Noul()}, _MODEL, media=(image,))
    assert isinstance(response.answers["q"], NoulAnswer)
    assert router.paths.count("/completion") == 1


def test_probe_reports_text_only_when_vision_is_not_required() -> None:
    """The probe accepts a text-only model only when the caller opts in."""
    router = _Router(_TEXT_PROPS)
    settings = load_llama_settings(_ENVIRON)
    with httpx.Client(
        base_url="http://router.test", transport=httpx.MockTransport(router.handle)
    ) as client:
        meta = probe_gemma_native_vision_support(
            settings=settings, require_vision=False, http_client=client
        )
        assert meta == {
            "ok": True,
            "model": _MODEL,
            "served": "native_gemma4_turn",
            "vision": False,
        }
        with pytest.raises(ValueError, match="text-only input modalities"):
            probe_gemma_native_vision_support(settings=settings, http_client=client)
