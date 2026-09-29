"""Unit tests: runtime limits from release contract r1 section 2 ([#202][i202]).

These tests characterize current behaviour. They do not change it. Client
closure on factory exit paths is proved in
``tests/contract/test_runtime_gemma_vision_factory.py`` (commit ``71275a4``).

Examples:
    ```bash
    uv run pytest -q tests/unit/test_runtime_limits.py
    ```

See Also:
    - [typevet.adapters.outbound.llama_cpp.gemma_native_vision_factory][]: Factory client
    - [typevet.adapters.outbound.llama_cpp.scoring][]: Media marker cache
    - [typevet.domain.media][]: ``ImageInput`` validation

[i202]: https://github.com/Alberto-Codes/typevet/issues/202
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

import httpx
import pytest

from typevet.adapters.inbound.settings import load_llama_settings
from typevet.adapters.outbound.llama_cpp.gemma_native_vision_factory import (
    GemmaNativeVisionSession,
    open_gemma_native_vision_judgment,
)
from typevet.adapters.outbound.llama_cpp.scoring import LlamaCppCandidateScoringAdapter
from typevet.domain.candidate_scoring_request import (
    CandidateScoringRequest,
    CandidateTokenSpec,
)
from typevet.domain.errors import TransportError
from typevet.domain.judgment_questions import Noul
from typevet.domain.media import MEDIA_MARKER, ImageInput

pytestmark = pytest.mark.unit

_MODEL = "gemma-4-limits"
_GEMMA4_RENDERED = "<|turn>user\nhello<turn|>\n<|turn>model\n"
_PNG = b"\x89PNG\r\n\x1a\nfake"
_EIGHT_MIB = 8 * 1024 * 1024


class _Router:
    """Offline llama.cpp router stub with a mutable media marker.

    Attributes:
        marker (str): Marker that ``/props`` reports now.
        props_calls (int): Count of ``/props`` requests served.
        completion_bodies (list[dict[str, Any]]): ``/completion`` JSON bodies.
        completion_error (httpx.HTTPError | None): Raised on ``/completion``.

    Examples:
        ```python
        router = _Router()
        assert router.props_calls == 0
        ```
    """

    def __init__(self) -> None:
        self.marker = "<__media_a__>"
        self.props_calls = 0
        self.completion_bodies: list[dict[str, Any]] = []
        self.completion_error: httpx.HTTPError | None = None

    def handle(self, request: httpx.Request) -> httpx.Response:
        """Return canned responses for template, props, tokenize and completion.

        Returns:
            Synthetic ``httpx.Response`` for supported routes.

        Raises:
            httpx.HTTPError: The configured ``completion_error`` on ``/completion``.
        """
        path = request.url.path
        if path == "/apply-template":
            return httpx.Response(200, json={"prompt": _GEMMA4_RENDERED})
        if path == "/props":
            self.props_calls += 1
            return httpx.Response(
                200,
                json={"modalities": {"vision": True}, "media_marker": self.marker},
            )
        if path == "/tokenize":
            content = json.loads(request.content.decode())["content"]
            token_id = 100 + sum(map(ord, content)) % 50
            return httpx.Response(200, json={"tokens": [token_id]})
        if path == "/completion":
            if self.completion_error is not None:
                raise self.completion_error
            self.completion_bodies.append(json.loads(request.content.decode()))
            top = [{"id": i, "logprob": -0.5 - i * 0.001} for i in range(100, 150)]
            return httpx.Response(
                200, json={"completion_probabilities": [{"top_logprobs": top}]}
            )
        return httpx.Response(404)


def _media_request(count: int = 1) -> CandidateScoringRequest:
    images = tuple(ImageInput(data=_PNG, mime_type="image/png") for _ in range(count))
    return CandidateScoringRequest(
        model=_MODEL,
        prefix=f"{MEDIA_MARKER * count} Colour?",
        candidates=(CandidateTokenSpec("a", (101,)), CandidateTokenSpec("b", (102,))),
        media=images,
    )


@contextmanager
def _open_with_env_timeout(
    router: _Router, captured: list[dict[str, Any]]
) -> Iterator[GemmaNativeVisionSession]:
    """Open the factory with an owned client built from environment settings.

    Yields:
        The factory session; the patched constructor records its kwargs.
    """
    real_client = httpx.Client

    def build(**kwargs: Any) -> httpx.Client:
        """Record constructor kwargs and attach the offline transport.

        Other Parameters:
            **kwargs: Keyword arguments the factory passes to ``httpx.Client``.

        Returns:
            A real ``httpx.Client`` over ``router``.
        """
        captured.append(kwargs)
        return real_client(transport=httpx.MockTransport(router.handle), **kwargs)

    settings = load_llama_settings(
        {
            "TYPEVET_LLAMA__BASE_URL": "http://offline-router",
            "TYPEVET_LLAMA__TIMEOUT": "12.5",
        }
    )
    with (
        patch(
            "typevet.adapters.outbound.llama_cpp.gemma_native_vision_factory.httpx.Client",
            side_effect=build,
        ),
        open_gemma_native_vision_judgment(settings=settings, model=_MODEL) as session,
    ):
        yield session


def test_env_timeout_reaches_factory_http_client() -> None:
    """``TYPEVET_LLAMA__TIMEOUT`` sets the timeout of the factory-owned client."""
    router = _Router()
    captured: list[dict[str, Any]] = []
    with _open_with_env_timeout(router, captured) as session:
        assert session.client.timeout == httpx.Timeout(12.5)
    assert [call["timeout"] for call in captured] == [12.5]


def test_transport_timeout_during_scoring_raises_transport_error() -> None:
    """A transport timeout on ``/completion`` surfaces as ``TransportError`` only."""
    router = _Router()
    router.completion_error = httpx.ReadTimeout("deadline exceeded")
    with (
        _open_with_env_timeout(router, []) as session,
        pytest.raises(TransportError, match="deadline exceeded") as info,
    ):
        session.port.judge(
            "Charged twice.", {"billing": Noul(instructions="?")}, _MODEL
        )
    assert type(info.value) is TransportError
    assert isinstance(info.value.__cause__, httpx.ReadTimeout)


def test_reused_adapter_keeps_cached_marker_after_router_change() -> None:
    """A reused adapter keeps its first marker; a new adapter reads the new one."""
    router = _Router()
    client = httpx.Client(transport=httpx.MockTransport(router.handle))
    reused = LlamaCppCandidateScoringAdapter("http://offline-router", client=client)
    reused.score_candidates(_media_request())
    router.marker = "<__media_b__>"
    reused.score_candidates(_media_request())
    fresh = LlamaCppCandidateScoringAdapter("http://offline-router", client=client)
    fresh.score_candidates(_media_request())
    prompts = [body["prompt"]["prompt_string"] for body in router.completion_bodies]
    assert [p.split()[0] for p in prompts] == [
        "<__media_a__>",
        "<__media_a__>",
        "<__media_b__>",
    ]
    assert router.props_calls == 2
    client.close()


def test_image_input_accepts_eight_mib_payload() -> None:
    """``ImageInput`` has no domain byte cap; an 8 MiB payload is accepted."""
    data = _PNG + b"\x00" * _EIGHT_MIB
    image = ImageInput(data=data, mime_type="image/png")
    assert len(image.data) > _EIGHT_MIB


def test_scoring_sends_every_image_without_count_cap() -> None:
    """The scoring adapter sends all images of one request; no count cap applies."""
    router = _Router()
    client = httpx.Client(transport=httpx.MockTransport(router.handle))
    adapter = LlamaCppCandidateScoringAdapter("http://offline-router", client=client)
    adapter.score_candidates(_media_request(count=16))
    body = router.completion_bodies[0]["prompt"]
    assert len(body["multimodal_data"]) == 16
    assert body["prompt_string"].count(router.marker) == 16
    client.close()
