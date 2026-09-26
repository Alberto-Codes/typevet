"""Contract tests for image-conditioned llama.cpp candidate scoring (#142)."""

from __future__ import annotations

import base64
import json
from typing import Any

import httpx
import pytest

from typevet.adapters.outbound.llama_cpp_scoring import LlamaCppCandidateScoringAdapter
from typevet.domain.candidate_scoring_request import (
    CandidateScoringRequest,
    CandidateTokenSpec,
)
from typevet.domain.errors import (
    BackendHttpError,
    GenerationError,
    ScoringUnsupportedCapabilityError,
    TransportError,
)
from typevet.domain.media import MEDIA_MARKER, ImageInput

_ROUTER_MARKER = "<__media_NaJbOnuq8__>"
_PNG = b"\x89PNG\r\n\x1a\nsynthetic-bytes"
_PNG_B64 = base64.b64encode(_PNG).decode("ascii")


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


def _completion_body(
    *,
    tokens_evaluated: object | None = None,
    tokens_predicted: object | None = None,
) -> dict[str, object]:
    body: dict[str, object] = {
        "completion_probabilities": [
            {
                "top_logprobs": [
                    {"id": 101, "logprob": -0.5},
                    {"id": 202, "logprob": -1.2},
                ]
            }
        ]
    }
    if tokens_evaluated is not None:
        body["tokens_evaluated"] = tokens_evaluated
    if tokens_predicted is not None:
        body["tokens_predicted"] = tokens_predicted
    return body


def _props_body(
    *,
    vision: bool = True,
    marker: str | None = _ROUTER_MARKER,
) -> dict[str, object]:
    body: dict[str, object] = {"modalities": {"vision": vision, "audio": False}}
    if marker is not None:
        body["media_marker"] = marker
    return body


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
        props_status: int = 200,
        props_error: Exception | None = None,
        completion: dict[str, object] | None = None,
    ) -> None:
        self.props = props if props is not None else _props_body()
        self.props_status = props_status
        self.props_error = props_error
        self.completion = completion if completion is not None else _completion_body()
        self.paths: list[str] = []
        self.completion_bodies: list[dict[str, Any]] = []
        self.props_params: list[str | None] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.paths.append(request.url.path)
        if request.url.path.endswith("/props"):
            if self.props_error is not None:
                raise self.props_error
            self.props_params.append(request.url.params.get("model"))
            if self.props_status != 200:
                return httpx.Response(self.props_status, text="props unavailable")
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


@pytest.mark.contract
def test_text_request_keeps_string_prompt_and_never_probes_props() -> None:
    router = _Router()
    result = router.adapter().score_candidates(_request())
    assert [c.logprob for c in result.candidates] == [-0.5, -1.2]
    body = router.completion_bodies[0]
    assert body["prompt"] == "Answer:"
    assert "multimodal_data" not in body
    assert router.paths == ["/completion"]


@pytest.mark.contract
def test_media_request_sends_nested_prompt_object_with_raw_base64() -> None:
    router = _Router()
    router.adapter().score_candidates(_request(media=(_image(),)))
    body = router.completion_bodies[0]
    prompt = _prompt_object(body)
    assert prompt["multimodal_data"] == [_PNG_B64]
    assert prompt["prompt_string"].startswith(_ROUTER_MARKER)
    assert "multimodal_data" not in body


@pytest.mark.contract
def test_media_request_substitutes_the_router_marker() -> None:
    router = _Router()
    router.adapter().score_candidates(_request(media=(_image(),)))
    prompt_string = _prompt_object(router.completion_bodies[0])["prompt_string"]
    assert MEDIA_MARKER not in prompt_string
    assert prompt_string == f"{_ROUTER_MARKER}Answer:"


@pytest.mark.contract
def test_media_base64_carries_no_data_uri_prefix() -> None:
    router = _Router()
    router.adapter().score_candidates(_request(media=(_image(),)))
    encoded = _prompt_object(router.completion_bodies[0])["multimodal_data"][0]
    assert not encoded.startswith("data:")
    assert base64.b64decode(encoded) == _PNG


@pytest.mark.contract
def test_media_request_keeps_pre_sampling_fields() -> None:
    router = _Router()
    router.adapter().score_candidates(_request(media=(_image(),)))
    body = router.completion_bodies[0]
    assert body["n_predict"] == 0
    assert body["n_probs"] == 262144
    assert body["temperature"] == 0
    assert body["top_k"] == 0
    assert body["top_p"] == 1
    assert body["post_sampling_probs"] is False
    assert body["stream"] is False
    assert body["model"] == "gemma-mm"


@pytest.mark.contract
def test_media_request_disables_prompt_cache() -> None:
    router = _Router()
    router.adapter().score_candidates(_request(media=(_image(),)))
    body = router.completion_bodies[0]
    assert "cache_prompt" in body
    assert body["cache_prompt"] is False


@pytest.mark.contract
def test_media_request_probes_props_for_the_requested_model() -> None:
    router = _Router()
    router.adapter().score_candidates(_request(media=(_image(),)))
    assert router.paths[0] == "/props"
    assert router.props_params == ["gemma-mm"]


@pytest.mark.contract
def test_media_capability_probe_runs_once_per_model() -> None:
    router = _Router()
    adapter = router.adapter()
    adapter.score_candidates(_request(media=(_image(),)))
    adapter.score_candidates(_request(media=(_image(),)))
    assert router.paths.count("/props") == 1


@pytest.mark.contract
def test_media_request_on_text_only_model_is_refused() -> None:
    router = _Router(props=_props_body(vision=False))
    with pytest.raises(ScoringUnsupportedCapabilityError, match="image input"):
        router.adapter().score_candidates(_request(media=(_image(),)))
    assert router.completion_bodies == []


@pytest.mark.contract
def test_media_request_without_declared_modalities_is_refused() -> None:
    router = _Router(props={"media_marker": _ROUTER_MARKER})
    with pytest.raises(ScoringUnsupportedCapabilityError, match="image input"):
        router.adapter().score_candidates(_request(media=(_image(),)))


@pytest.mark.contract
def test_media_request_falls_back_to_documented_marker() -> None:
    router = _Router(props=_props_body(marker=None))
    router.adapter().score_candidates(_request(media=(_image(),)))
    prompt_string = _prompt_object(router.completion_bodies[0])["prompt_string"]
    assert prompt_string == f"{MEDIA_MARKER}Answer:"


@pytest.mark.contract
def test_props_http_error_maps_to_backend_http_error() -> None:
    router = _Router(props_status=503)
    with pytest.raises(BackendHttpError, match="HTTP 503"):
        router.adapter().score_candidates(_request(media=(_image(),)))


@pytest.mark.contract
def test_props_transport_error_maps_to_transport_error() -> None:
    router = _Router(props_error=httpx.ConnectError("refused"))
    with pytest.raises(TransportError, match=r"llama\.cpp request failed"):
        router.adapter().score_candidates(_request(media=(_image(),)))


@pytest.mark.contract
def test_props_non_object_body_is_rejected() -> None:
    router = _Router(props=["not", "an", "object"])
    with pytest.raises(GenerationError, match="props"):
        router.adapter().score_candidates(_request(media=(_image(),)))


@pytest.mark.contract
def test_prompt_token_count_reaches_the_caller() -> None:
    """``tokens_evaluated`` is the only signal that the image was attached."""
    router = _Router(
        completion=_completion_body(tokens_evaluated=315, tokens_predicted=1)
    )
    result = router.adapter().score_candidates(_request(media=(_image(),)))
    assert result.usage.input_tokens == 315
    assert result.usage.output_tokens == 1


@pytest.mark.contract
def test_text_request_also_reports_prompt_token_count() -> None:
    router = _Router(completion=_completion_body(tokens_evaluated=56))
    result = router.adapter().score_candidates(_request())
    assert result.usage.input_tokens == 56
    assert result.usage.output_tokens is None


@pytest.mark.contract
@pytest.mark.parametrize(
    "tokens_evaluated",
    [None, "many", -1],
    ids=["absent", "not-a-number", "negative"],
)
def test_unusable_prompt_token_count_reports_unknown(
    tokens_evaluated: object | None,
) -> None:
    router = _Router(completion=_completion_body(tokens_evaluated=tokens_evaluated))
    result = router.adapter().score_candidates(_request(media=(_image(),)))
    assert result.usage.input_tokens is None


@pytest.mark.contract
def test_multiple_images_encode_in_request_order() -> None:
    router = _Router()
    second = ImageInput(data=b"second-image-bytes", mime_type="image/webp")
    request = _request(media=(_image(), second))
    router.adapter().score_candidates(request)
    prompt = _prompt_object(router.completion_bodies[0])
    assert prompt["multimodal_data"] == [
        _PNG_B64,
        base64.b64encode(b"second-image-bytes").decode("ascii"),
    ]
    assert prompt["prompt_string"].count(_ROUTER_MARKER) == 2
