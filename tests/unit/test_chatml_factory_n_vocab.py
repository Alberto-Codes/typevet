"""Unit tests: factory ``n_vocab`` default with ChatML framing ([#444][i444]).

With framing, an omitted ``n_vocab`` resolves to a catalog lookup so a
200-token vocabulary can report finite off-option mass. Explicit values,
including ``None``, win. Without framing, Gemma keeps ``262144`` and skips
the catalog.

Examples:
    ```bash
    uv run pytest -q tests/unit/test_chatml_factory_n_vocab.py
    ```

See Also:
    - [typevet.adapters.outbound.llama_cpp.gemma_native_vision_factory][]:
      Factory under test
    - [typevet.adapters.outbound.llama_cpp.scoring][]: Post-completion catalog
      read

[i444]: https://github.com/Alberto-Codes/typevet/issues/444
"""

from __future__ import annotations

import json
import math
from typing import Any

import httpx
import pytest

from tests.fixtures.gemma_template_contract import (
    degraded_chatml_rendered,
    native_gemma4_rendered,
)
from typevet.adapters.inbound.backend_settings import open_judgment
from typevet.adapters.inbound.settings import load_llama_settings
from typevet.adapters.outbound.gemma import ChatMLThinkingFraming
from typevet.adapters.outbound.llama_cpp.gemma_native_vision_factory import (
    open_gemma_native_vision_judgment,
)
from typevet.adapters.outbound.llama_cpp.scoring import DEFAULT_N_VOCAB
from typevet.domain.judgment_answers import NoulAnswer
from typevet.domain.judgment_questions import Noul

pytestmark = pytest.mark.unit

_QWEN = "qwen-chatml-text"
_GEMMA = "gemma-4-31b-kv9-text"
_QWEN_ENV = {
    "TYPEVET_BACKEND": "llama_cpp",
    "TYPEVET_LLAMA__MULTIMODAL_MODEL": _QWEN,
    "TYPEVET_LLAMA__DEFAULT_MODEL": _QWEN,
}
_GEMMA_ENV = {
    "TYPEVET_BACKEND": "llama_cpp",
    "TYPEVET_LLAMA__MULTIMODAL_MODEL": _GEMMA,
}
_CHATML_RENDERED = degraded_chatml_rendered("hello")
_GEMMA4_RENDERED = native_gemma4_rendered("hello")
_VOCAB = 200
_CAND_FALSE = 100 + ord("0") % 50
_CAND_TRUE = 100 + ord("1") % 50


def _text_props() -> dict[str, Any]:
    return {"modalities": {"vision": False, "audio": False}}


def _normalized_top(count: int) -> list[dict[str, float | int]]:
    """Return a full-vocabulary top_logprobs list with mass 1.0.

    Controls ``0`` and ``1`` map to ``_CAND_FALSE`` and ``_CAND_TRUE``. Those
    tokens take 0.5 and 0.3; the rest share 0.2.
    """
    rest = 0.2 / (count - 2)
    entries: list[dict[str, float | int]] = []
    for token_id in range(count):
        if token_id == _CAND_FALSE:
            prob = 0.5
        elif token_id == _CAND_TRUE:
            prob = 0.3
        else:
            prob = rest
        entries.append({"id": token_id, "logprob": math.log(prob)})
    return entries


def _models_body(n_vocab: object | None, *, model_id: str) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    if n_vocab is not None:
        meta["n_vocab"] = n_vocab
    return {"object": "list", "data": [{"id": model_id, "meta": meta}]}


class _Router:
    """Offline llama.cpp router with catalog, completion and path records.

    Attributes:
        props (dict[str, Any]): Body served on ``/props``.
        template (str): Prompt string served on ``/apply-template``.
        model_id (str): Model id listed on ``/v1/models``.
        models_n_vocab (object | None): ``meta.n_vocab``, or omit the field
            when ``None`` and ``models_omit_meta`` is true.
        models_omit_meta (bool): When true, the models entry has no ``meta``.
        models_status (int): HTTP status for ``/v1/models``.
        completion_count (int): Entry count returned on ``/completion``.
        paths (list[str]): Request paths in call order.
        n_probs (list[int]): ``n_probs`` values from each ``/completion``.

    Examples:
        ```python
        router = _Router(_text_props(), _CHATML_RENDERED, _QWEN)
        assert router.paths == []
        ```
    """

    def __init__(
        self,
        props: dict[str, Any],
        template: str,
        model_id: str,
        *,
        models_n_vocab: object | None = _VOCAB,
        models_omit_meta: bool = False,
        models_status: int = 200,
        completion_count: int = _VOCAB,
    ) -> None:
        self.props = props
        self.template = template
        self.model_id = model_id
        self.models_n_vocab = models_n_vocab
        self.models_omit_meta = models_omit_meta
        self.models_status = models_status
        self.completion_count = completion_count
        self.paths: list[str] = []
        self.n_probs: list[int] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        """Serve props, template, tokenize, completion and models bodies.

        Args:
            request: The request the client sent.

        Returns:
            A synthetic llama.cpp response, or 404 for another path.
        """
        path = request.url.path
        self.paths.append(path)
        response: httpx.Response | None = None
        if path == "/props":
            response = httpx.Response(200, json=self.props)
        elif path == "/apply-template":
            response = httpx.Response(200, json={"prompt": self.template})
        elif path == "/tokenize":
            content = json.loads(request.content.decode())["content"]
            response = httpx.Response(
                200, json={"tokens": [100 + sum(map(ord, content)) % 50]}
            )
        elif path == "/completion":
            response = self._completion(request)
        elif path == "/v1/models":
            response = self._models()
        return response if response is not None else httpx.Response(404)

    def _completion(self, request: httpx.Request) -> httpx.Response:
        """Build the ``/completion`` response for ``request``.

        Args:
            request: The ``/completion`` request.

        Returns:
            A 200 response with a normalized top_logprobs list.
        """
        body = json.loads(request.content.decode())
        self.n_probs.append(body["n_probs"])
        top = _normalized_top(self.completion_count)
        return httpx.Response(
            200, json={"completion_probabilities": [{"top_logprobs": top}]}
        )

    def _models(self) -> httpx.Response:
        """Build the ``/v1/models`` response from the router catalog options.

        Returns:
            The configured models list, or an error status body.
        """
        if self.models_status != 200:
            return httpx.Response(self.models_status, json={"error": "boom"})
        if self.models_omit_meta:
            payload: dict[str, Any] = {
                "object": "list",
                "data": [{"id": self.model_id}],
            }
        else:
            payload = _models_body(self.models_n_vocab, model_id=self.model_id)
        return httpx.Response(200, json=payload)


def test_framed_omitted_n_vocab_reads_catalog_and_reports_mass() -> None:
    """Omitted ``n_vocab`` with framing yields finite mass and one catalog read."""
    router = _Router(_text_props(), _CHATML_RENDERED, _QWEN)
    settings = load_llama_settings(_QWEN_ENV)
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
        response = session.port.judge("Charged twice.", {"q": Noul()}, _QWEN)
    assert isinstance(response.answers["q"], NoulAnswer)
    assert response.off_option["q"].off_option_mass == pytest.approx(0.2, abs=1e-12)
    assert router.paths.count("/v1/models") == 1


def test_public_framed_omitted_n_vocab_reads_catalog_and_reports_mass() -> None:
    """Public ``open_judgment`` with framing reports mass from the catalog size."""
    router = _Router(_text_props(), _CHATML_RENDERED, _QWEN)
    with open_judgment(
        _QWEN_ENV,
        transport=httpx.MockTransport(router.handle),
        framing=ChatMLThinkingFraming(),
    ) as session:
        response = session.port.judge("Charged twice.", {"q": Noul()}, _QWEN)
    assert isinstance(response.answers["q"], NoulAnswer)
    assert response.off_option["q"].off_option_mass == pytest.approx(0.2, abs=1e-12)
    assert router.paths.count("/v1/models") == 1


def test_framed_missing_catalog_metadata_keeps_result_with_unknown_mass() -> None:
    """Missing catalog ``meta.n_vocab`` still answers with unknown mass."""
    router = _Router(
        _text_props(),
        _CHATML_RENDERED,
        _QWEN,
        models_omit_meta=True,
    )
    settings = load_llama_settings(_QWEN_ENV)
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
        response = session.port.judge("Charged twice.", {"q": Noul()}, _QWEN)
    assert isinstance(response.answers["q"], NoulAnswer)
    assert response.off_option["q"].off_option_mass is None
    assert router.paths.count("/v1/models") == 1


def test_framed_invalid_catalog_metadata_keeps_result_with_unknown_mass() -> None:
    """Invalid catalog ``meta.n_vocab`` still answers with unknown mass."""
    router = _Router(
        _text_props(),
        _CHATML_RENDERED,
        _QWEN,
        models_n_vocab="not-an-int",
    )
    settings = load_llama_settings(_QWEN_ENV)
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
        response = session.port.judge("Charged twice.", {"q": Noul()}, _QWEN)
    assert isinstance(response.answers["q"], NoulAnswer)
    assert response.off_option["q"].off_option_mass is None
    assert router.paths.count("/v1/models") == 1


def test_framed_explicit_integer_n_vocab_skips_catalog() -> None:
    """An explicit integer ``n_vocab`` reports mass without a catalog read."""
    router = _Router(_text_props(), _CHATML_RENDERED, _QWEN)
    settings = load_llama_settings(_QWEN_ENV)
    with (
        httpx.Client(
            base_url="http://router.test", transport=httpx.MockTransport(router.handle)
        ) as client,
        open_gemma_native_vision_judgment(
            settings=settings,
            require_gemma4=False,
            require_vision=False,
            framing=ChatMLThinkingFraming(),
            n_vocab=_VOCAB,
            http_client=client,
        ) as session,
    ):
        response = session.port.judge("Charged twice.", {"q": Noul()}, _QWEN)
    assert isinstance(response.answers["q"], NoulAnswer)
    assert response.off_option["q"].off_option_mass == pytest.approx(0.2, abs=1e-12)
    assert "/v1/models" not in router.paths
    assert router.n_probs == [_VOCAB]


def test_explicit_none_n_vocab_on_gemma_path_reads_catalog() -> None:
    """Explicit ``n_vocab=None`` on the Gemma path reads the catalog for mass."""
    router = _Router(_text_props(), _GEMMA4_RENDERED, _GEMMA)
    settings = load_llama_settings(_GEMMA_ENV)
    with (
        httpx.Client(
            base_url="http://router.test", transport=httpx.MockTransport(router.handle)
        ) as client,
        open_gemma_native_vision_judgment(
            settings=settings,
            require_vision=False,
            n_vocab=None,
            http_client=client,
        ) as session,
    ):
        response = session.port.judge("Charged twice.", {"q": Noul()}, _GEMMA)
    assert isinstance(response.answers["q"], NoulAnswer)
    assert response.off_option["q"].off_option_mass == pytest.approx(0.2, abs=1e-12)
    assert router.paths.count("/v1/models") == 1


def test_default_gemma_sends_default_n_probs_without_catalog() -> None:
    """Omitted ``n_vocab`` without framing keeps Gemma ``n_probs`` and no catalog."""
    router = _Router(_text_props(), _GEMMA4_RENDERED, _GEMMA)
    settings = load_llama_settings(_GEMMA_ENV)
    with (
        httpx.Client(
            base_url="http://router.test", transport=httpx.MockTransport(router.handle)
        ) as client,
        open_gemma_native_vision_judgment(
            settings=settings,
            require_vision=False,
            http_client=client,
        ) as session,
    ):
        response = session.port.judge("Charged twice.", {"q": Noul()}, _GEMMA)
    assert isinstance(response.answers["q"], NoulAnswer)
    assert router.n_probs == [DEFAULT_N_VOCAB]
    assert "/v1/models" not in router.paths
    assert response.off_option["q"].off_option_mass is None
