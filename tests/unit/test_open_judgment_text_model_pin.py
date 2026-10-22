"""Unit tests: framed ``open_judgment`` pins ``settings.default_model`` ([#444][i444]).

With ``framing``, the public factory pins every model-bearing request to
``TYPEVET_LLAMA__DEFAULT_MODEL`` (or legacy ``TYPEVET_GEMMA_MODEL``). A missing
or empty pin raises ``ValueError`` that names that setting before any request.
Without framing, ``TYPEVET_LLAMA__MULTIMODAL_MODEL`` stays the pin. A fake
transport records each request model.

Examples:
    ```bash
    uv run pytest -q tests/unit/test_open_judgment_text_model_pin.py
    ```

See Also:
    - [typevet.adapters.inbound][]: Public ``open_judgment``
    - [typevet.adapters.inbound.backend_settings][]: Composition root
    - [typevet.adapters.inbound.settings][]: ``default_model`` load path

[i444]: https://github.com/Alberto-Codes/typevet/issues/444
"""

from __future__ import annotations

import json
import re
from typing import Any

import httpx
import pytest

from tests.fixtures.gemma_template_contract import (
    degraded_chatml_rendered,
    native_gemma4_rendered,
)
from typevet.adapters import inbound
from typevet.adapters.outbound.gemma import ChatMLThinkingFraming
from typevet.adapters.outbound.llama_cpp.gemma_native_vision_factory import (
    GemmaNativeVisionSession,
)
from typevet.domain.errors import JudgmentValidationError
from typevet.domain.judgment_answers import NoulAnswer
from typevet.domain.judgment_questions import Noul

pytestmark = pytest.mark.unit

_TEXT = "qwen-chatml-text"
_VISION = "gemma-4-vision-mm"
_CHATML_RENDERED = degraded_chatml_rendered("hello")
_GEMMA4_RENDERED = native_gemma4_rendered("hello")
_DEFAULT_VAR = "TYPEVET_LLAMA__DEFAULT_MODEL"
_MULTIMODAL_VAR = "TYPEVET_LLAMA__MULTIMODAL_MODEL"
_LEGACY_VAR = "TYPEVET_GEMMA_MODEL"
_SETTINGS_LABEL = f"settings.default_model ({_DEFAULT_VAR})"
_SETTINGS_MATCH = re.escape(_SETTINGS_LABEL)


class _Router:
    """Offline llama.cpp router that records the model on each path.

    Attributes:
        props (dict[str, Any]): Body served on ``/props``.
        template (str): Prompt string served on ``/apply-template``.
        paths (list[str]): Request paths in call order.
        models (dict[str, list[str]]): Model ids recorded per path.

    Examples:
        ```python
        router = _Router({"modalities": {"vision": False}}, _CHATML_RENDERED)
        assert router.models == {}
        ```
    """

    def __init__(self, props: dict[str, Any], template: str) -> None:
        self.props = props
        self.template = template
        self.paths: list[str] = []
        self.models: dict[str, list[str]] = {}

    def _record(self, path: str, model: str) -> None:
        self.models.setdefault(path, []).append(model)

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
            model = request.url.params.get("model", "")
            assert isinstance(model, str)
            self._record(path, model)
            return httpx.Response(200, json=self.props)
        if path == "/apply-template":
            body = json.loads(request.content.decode())
            self._record(path, body["model"])
            return httpx.Response(200, json={"prompt": self.template})
        if path == "/tokenize":
            body = json.loads(request.content.decode())
            self._record(path, body["model"])
            content = body["content"]
            return httpx.Response(
                200, json={"tokens": [100 + sum(map(ord, content)) % 50]}
            )
        if path == "/completion":
            body = json.loads(request.content.decode())
            self._record(path, body["model"])
            top = [{"id": i, "logprob": -0.2 - (i % 5) * 0.15} for i in range(100, 180)]
            return httpx.Response(
                200, json={"completion_probabilities": [{"top_logprobs": top}]}
            )
        return httpx.Response(404)


def _text_props() -> dict[str, Any]:
    return {"modalities": {"vision": False, "audio": False}}


def _vision_props() -> dict[str, Any]:
    return {
        "modalities": {"vision": True, "audio": False},
        "media_marker": "<__media_444c__>",
    }


def _framed_environ(**extra: str) -> dict[str, str]:
    environ = {
        "TYPEVET_BACKEND": "llama_cpp",
        _MULTIMODAL_VAR: _VISION,
        _DEFAULT_VAR: _TEXT,
    }
    environ.update(extra)
    return environ


def test_framed_open_judgment_pins_default_model_on_session_and_requests() -> None:
    """Framed public open pins session and every model-bearing request."""
    router = _Router(_text_props(), _CHATML_RENDERED)
    with inbound.open_judgment(
        _framed_environ(),
        transport=httpx.MockTransport(router.handle),
        framing=ChatMLThinkingFraming(),
    ) as session:
        assert isinstance(session, GemmaNativeVisionSession)
        assert session.model == _TEXT
        response = session.port.judge("Charged twice.", {"q": Noul()}, _TEXT)
    assert isinstance(response.answers["q"], NoulAnswer)
    for path in ("/props", "/apply-template", "/tokenize", "/completion"):
        recorded = router.models.get(path, [])
        assert recorded, path
        assert set(recorded) == {_TEXT}, path
    assert _VISION not in {m for ids in router.models.values() for m in ids}


def test_missing_default_model_refuses_before_any_request() -> None:
    """Framing without a text pin raises and sends no request."""
    router = _Router(_text_props(), _CHATML_RENDERED)
    environ = {
        "TYPEVET_BACKEND": "llama_cpp",
        _MULTIMODAL_VAR: _VISION,
    }
    with (
        pytest.raises(ValueError, match=_SETTINGS_MATCH) as info,
        inbound.open_judgment(
            environ,
            transport=httpx.MockTransport(router.handle),
            framing=ChatMLThinkingFraming(),
        ),
    ):
        pass
    assert _DEFAULT_VAR in str(info.value)
    assert router.paths == []
    assert router.models == {}


def test_empty_default_model_refuses_before_any_request() -> None:
    """An empty nested text pin is refused before any request."""
    router = _Router(_text_props(), _CHATML_RENDERED)
    environ = {
        "TYPEVET_BACKEND": "llama_cpp",
        _MULTIMODAL_VAR: _VISION,
        _DEFAULT_VAR: "   ",
    }
    with (
        pytest.raises(ValueError, match=_SETTINGS_MATCH),
        inbound.open_judgment(
            environ,
            transport=httpx.MockTransport(router.handle),
            framing=ChatMLThinkingFraming(),
        ),
    ):
        pass
    assert router.paths == []


def test_legacy_gemma_model_fallback_pins_framed_session() -> None:
    """Legacy ``TYPEVET_GEMMA_MODEL`` pins framing when the nested name is unset."""
    router = _Router(_text_props(), _CHATML_RENDERED)
    environ = {
        "TYPEVET_BACKEND": "llama_cpp",
        _MULTIMODAL_VAR: _VISION,
        _LEGACY_VAR: _TEXT,
    }
    with inbound.open_judgment(
        environ,
        transport=httpx.MockTransport(router.handle),
        framing=ChatMLThinkingFraming(),
    ) as session:
        assert session.model == _TEXT
        session.port.judge("Charged twice.", {"q": Noul()}, _TEXT)
    assert router.models["/props"] == [_TEXT]
    assert router.models["/completion"] == [_TEXT]


def test_default_model_precedes_legacy_fallback() -> None:
    """Nested ``DEFAULT_MODEL`` wins over legacy ``TYPEVET_GEMMA_MODEL``."""
    router = _Router(_text_props(), _CHATML_RENDERED)
    environ = {
        "TYPEVET_BACKEND": "llama_cpp",
        _MULTIMODAL_VAR: _VISION,
        _LEGACY_VAR: "legacy-unused-text",
        _DEFAULT_VAR: _TEXT,
    }
    with inbound.open_judgment(
        environ,
        transport=httpx.MockTransport(router.handle),
        framing=ChatMLThinkingFraming(),
    ) as session:
        assert session.model == _TEXT
        session.port.judge("Charged twice.", {"q": Noul()}, _TEXT)
    assert "legacy-unused-text" not in {
        m for ids in router.models.values() for m in ids
    }


def test_mismatched_judgment_model_fails_before_further_requests() -> None:
    """A judge call with another model id fails without new requests."""
    router = _Router(_text_props(), _CHATML_RENDERED)
    with inbound.open_judgment(
        _framed_environ(),
        transport=httpx.MockTransport(router.handle),
        framing=ChatMLThinkingFraming(),
    ) as session:
        opened = list(router.paths)
        with pytest.raises(
            JudgmentValidationError, match="does not match pinned model"
        ):
            session.port.judge("Charged twice.", {"q": Noul()}, _VISION)
    assert router.paths == opened
    assert "/completion" not in router.paths


def test_no_framing_keeps_multimodal_pin_when_both_are_set() -> None:
    """Without framing, the multimodal pin stays even when a text pin is set."""
    router = _Router(_vision_props(), _GEMMA4_RENDERED)
    with inbound.open_judgment(
        _framed_environ(),
        transport=httpx.MockTransport(router.handle),
    ) as session:
        assert isinstance(session, GemmaNativeVisionSession)
        assert session.model == _VISION
        response = session.port.judge("Charged twice.", {"q": Noul()}, _VISION)
    assert isinstance(response.answers["q"], NoulAnswer)
    for path in ("/props", "/apply-template", "/tokenize", "/completion"):
        recorded = router.models.get(path, [])
        assert recorded, path
        assert set(recorded) == {_VISION}, path
    assert _TEXT not in {m for ids in router.models.values() for m in ids}
