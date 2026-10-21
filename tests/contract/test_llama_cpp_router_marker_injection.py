"""Contract: router-reported media markers in caller text bind no image (#436).

``GET /props`` reports a per-instance media marker. The adapter substitutes
that marker for ``MEDIA_MARKER``. Caller text that already holds the same
marker must not bind an extra image.

Examples:
    ```bash
    uv run pytest -q tests/contract/test_llama_cpp_router_marker_injection.py
    ```

See Also:
    - [typevet.adapters.outbound.llama_cpp.multimodal][]: ``media_prompt_field``
    - [typevet.domain.media][]: ``neutralize_media_markers``
"""

from __future__ import annotations

import base64
import json
from typing import Any

import httpx
import pytest

from typevet.adapters.outbound.llama_cpp.scoring import LlamaCppCandidateScoringAdapter
from typevet.domain.candidate_scoring_request import (
    CandidateScoringRequest,
    CandidateTokenSpec,
)
from typevet.domain.errors import GenerationError
from typevet.domain.media import MEDIA_MARKER, ImageInput

pytestmark = pytest.mark.contract

_ROUTER_MARKER = "<__media_NaJbOnuq8__>"
_UNESCAPABLE_MARKER = "<media>"
_PNG = b"\x89PNG\r\n\x1a\nsynthetic-bytes"
_PNG_B64 = base64.b64encode(_PNG).decode("ascii")
_NEUTRALIZED_ROUTER = r"<\_\_media\_NaJbOnuq8\_\_>"


def _image() -> ImageInput:
    return ImageInput(data=_PNG, mime_type="image/png")


def _request(
    *,
    media: tuple[ImageInput, ...] = (),
    prefix: str | None = None,
) -> CandidateScoringRequest:
    marker_block = MEDIA_MARKER * len(media)
    return CandidateScoringRequest(
        model="gemma-mm",
        prefix=prefix if prefix is not None else f"{marker_block}Answer:",
        candidates=(
            CandidateTokenSpec("billing", (101,)),
            CandidateTokenSpec("technical", (202,)),
        ),
        media=media,
    )


def _completion_body() -> dict[str, object]:
    return {
        "completion_probabilities": [
            {
                "top_logprobs": [
                    {"id": 101, "logprob": -0.5},
                    {"id": 202, "logprob": -1.2},
                ]
            }
        ]
    }


def _props_body(*, marker: str) -> dict[str, object]:
    return {
        "modalities": {"vision": True, "audio": False},
        "media_marker": marker,
    }


def _prompt_object(body: dict[str, Any]) -> dict[str, Any]:
    prompt = body["prompt"]
    assert isinstance(prompt, dict)
    return prompt


class _Router:
    """Record every request and answer ``/props`` and ``/completion``."""

    def __init__(
        self,
        *,
        props: object | None = None,
    ) -> None:
        self.props = props if props is not None else _props_body(marker=_ROUTER_MARKER)
        self.completion = _completion_body()
        self.paths: list[str] = []
        self.completion_bodies: list[dict[str, Any]] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.paths.append(request.url.path)
        if request.url.path.endswith("/props"):
            return httpx.Response(200, json=self.props)
        self.completion_bodies.append(json.loads(request.content.decode()))
        return httpx.Response(200, json=self.completion)

    def adapter(self) -> LlamaCppCandidateScoringAdapter:
        client = httpx.Client(
            transport=httpx.MockTransport(self.handle),
            base_url="http://test",
        )
        return LlamaCppCandidateScoringAdapter(
            base_url="http://test",
            client=client,
            n_vocab=262144,
        )


def test_router_marker_in_caller_text_binds_one_image() -> None:
    router = _Router(props=_props_body(marker=_ROUTER_MARKER))
    prefix = f"{MEDIA_MARKER}diff holds {_ROUTER_MARKER} Answer:"
    router.adapter().score_candidates(_request(media=(_image(),), prefix=prefix))
    prompt_string = _prompt_object(router.completion_bodies[0])["prompt_string"]
    assert prompt_string.count(_ROUTER_MARKER) == 1
    assert _NEUTRALIZED_ROUTER in prompt_string
    assert _prompt_object(router.completion_bodies[0])["multimodal_data"] == [_PNG_B64]


def test_unescapable_router_marker_in_caller_text_raises() -> None:
    router = _Router(props=_props_body(marker=_UNESCAPABLE_MARKER))
    prefix = f"{MEDIA_MARKER}holds {_UNESCAPABLE_MARKER} Answer:"
    with pytest.raises(GenerationError):
        router.adapter().score_candidates(_request(media=(_image(),), prefix=prefix))


def test_text_request_with_router_marker_sends_prefix_unchanged() -> None:
    router = _Router(props=_props_body(marker=_ROUTER_MARKER))
    prefix = f"diff holds {_ROUTER_MARKER} Answer:"
    router.adapter().score_candidates(_request(prefix=prefix))
    body = router.completion_bodies[0]
    assert body["prompt"] == prefix
    assert "multimodal_data" not in body
    assert router.paths == ["/completion"]
